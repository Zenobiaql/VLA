import logging
import sys

import datasets
import torch
import transformers
from transformers import set_seed
from transformers import TrainerCallback
from transformers import LlamaTokenizer
from collections import OrderedDict
from safetensors import safe_open

sys.path.append('.')
from src import DataArguments, H4ArgumentParser, ModelArguments, SFTConfig, get_checkpoint, get_datasets
from src import get_VLA_dataset

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os

from peft import LoraConfig, TaskType, get_peft_model

from llm_backbone import Phi3InVisionActionFeatMask, MistralInVisionActionFeatMask
from llm_backbone import Codebook

logger = logging.getLogger(__name__)

def load_safetensors_weights(model, checkpoint_dir): 
    weights_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')] 
    for weights_file in weights_files: 
        weights_path = os.path.join(checkpoint_dir, weights_file) 
        with safe_open(weights_path, framework="pt", device='cpu') as f: 
            for key in f.keys():
                if 'embed_tokens' in key:
                    if key in model.state_dict().keys():
                        print('Load key: {}, Shape: {}'.format(key, model.state_dict()[key].shape))
                        model.state_dict()[key].copy_(f.get_tensor(key))
                    else:
                        print('Skip key {}'.format(key))
    return model


def main():
    try:
        print('MASTER_ADDR', os.environ['MASTER_ADDR'])
        print('MASTER_PORT', os.environ['MASTER_PORT'])
        print('NODE_RANK', os.environ['NODE_RANK'])
        print('LOCAL_RANK', os.environ['LOCAL_RANK'])
        print('RANK', os.environ['RANK'])
        print('WORLD_SIZE', os.environ['WORLD_SIZE'])
    except:
        pass

    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    """
    Load tokenizer
        The visual modality has 2048 (16384) tokens, and the action modality has 256 tokens, add them to the tokenizer
        Add special tokens for the visual and action modalities, 
    including <bots_i>, <eots_i>, <botp_i>, <eotp_i>, <bov_i>, <eov_i>, <boa_i>, <eoa_i>,
                <botp_o>, <eotp_o>, <bov_o>, <eov_o>, <boa_o>, <eoa_o>
    In total 16384 + vocab_size
    """
    
    if model_args.disable_auto_config:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    vocab_size = len(tokenizer)

    # add eos token when when calling tokenizer
    visual_action_tokens_to_add = ['<va' + str(i) + '>' for i in range(0, data_args.num_visual_action_tokens)]
    num_added_visual_action_tokens = tokenizer.add_special_tokens({'additional_special_tokens': visual_action_tokens_to_add})
    special_tokens = ['<bott_i>', '<eott_i>', # task text
                        '<bots_i>', '<eots_i>', # scene text
                        '<botp_i>', '<eotp_i>', # policy text
                        '<bov_i>', '<eov_i>', '<boa_i>', '<eoa_i>', # vision and action tokens
                        '<botp_o>', '<eotp_o>', # output policy text
                        '<bov_o>', '<eov_o>', '<boa_o>', '<eoa_o>'] # output vision and action tokens
    num_added_special_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # For SFT training, padding should be on the right (if overflow occurs)
    tokenizer.padding_side = data_args.padding_side

    # Load and pre-process the dataset
    train_dataset = get_VLA_dataset(data_args, tokenizer.eos_token, split='train')
    eval_dataset = get_VLA_dataset(data_args, tokenizer.eos_token, split='test')

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        # take a sample from the dataset (iteratable)
        if type(train_dataset) == datasets.IterableDataset:
            for i, example in enumerate(train_dataset.take(3)):
                logger.info(f"Sample {i}: {example['text']}")
        else:
            for i in range(3):
                logger.info(f"Sample {i}: {train_dataset[i]}")
    
    # input commonly ends by <eoa_i>, use <eoa_i> as the response template
    # when only text, input ends by <eotp_i>
    if data_args.only_text:
        response_template_id = tokenizer.convert_tokens_to_ids(['<eotp_i>'])
    else:
        response_template_id = tokenizer.convert_tokens_to_ids(['<eoa_i>'])

    data_collator = DataCollatorForCompletionOnlyLM(response_template_id, tokenizer=tokenizer)

    # Load pretrained model
    logger.info("*** Load pretrained model ***")
    # use float16 (V100 does not support bfloat16)
    torch_dtype = torch.float16 if training_args.fp16 else torch.float32

    model_kwargs = dict(
        # revision=model_args.model_revision,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        # trust_remote_code=True,
        use_cache=False if training_args.gradient_checkpointing else True
    )

    # Load Vision Action Codebook
    logger.info("*** Load Vision Action Codebook ***")
    va_embed = Codebook(model_args.va_ncodes, model_args.va_embedding_dim)
    state_dict = torch.load(model_args.va_checkpoint, map_location='cpu')['state_dict'] ################# check !!!
    new_state_dict = OrderedDict()
    for key in list(state_dict.keys()):
        if key == 'codebook.embeddings':
            new_state_dict['embeddings'] = state_dict[key]
            break
    load_info = va_embed.load_state_dict(new_state_dict, strict=True)
    print(load_info)
    va_embed.to(training_args.device)

    # Initialize LLM
    llm_checkpoint_path = model_args.model_name_or_path
    if training_args.resume_from_checkpoint is not None:
        logger.info(f"Checkpoint detected, loading weights at {training_args.resume_from_checkpoint}.")
        llm_checkpoint_path = training_args.resume_from_checkpoint
    if model_args.model_type == 'phi3':
        # configuration = Phi3Config.from_pretrained()
        model = Phi3InVisionActionFeatMask.from_pretrained(llm_checkpoint_path, 
                                                        tokenizer, va_embed, model_args.v_mask_ratio, **model_kwargs)
    elif model_args.model_type == 'mistral':
        # configuration = MistralConfig.from_pretrained(model_args.model_name_or_path)
        model = MistralInVisionActionFeatMask.from_pretrained(llm_checkpoint_path, 
                                                            tokenizer, va_embed, model_args.v_mask_ratio, **model_kwargs)
        
    # Load weights of embed_tokens
    if training_args.resume_from_checkpoint is not None:
        model = load_safetensors_weights(model, llm_checkpoint_path)
            
    # Initialize the Trainer
    class PrintCallback(TrainerCallback):
        def on_evaluation(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            # print whether this process should save the checkpoint
            print(f'Process {args.local_rank} should save checkpoint: {args.should_save}')

    # peft module
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        data_collator=data_collator,
        callbacks=[PrintCallback()] if training_args.debug else None,
        max_seq_length=training_args.max_seq_length,
        dataset_num_proc=data_args.preprocessing_num_workers,
        dataset_kwargs=training_args.dataset_kwargs,
    )

    """
    Training loop
    Check for last checkpoint
    """
    
    logger.info("*** Train ***")
    checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save model and create model card
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Evaluate
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()

"""
Predict
"""

import logging
import random
import sys

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed, MistralModel, PhiModel
from transformers import TrainerCallback

sys.path.append('.')
from src import DataArguments, H4ArgumentParser, ModelArguments, SFTConfig, get_checkpoint, get_datasets
from src import get_VLA_dataset_debug as get_VLA_dataset

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os
import json

logger = logging.getLogger(__name__)

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

    ###############
    # Setup logging
    ###############
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

    ################
    # Load tokenizer
    ################
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # vocab_size = len(transformers.AutoTokenizer.from_pretrained(model_args.base_model_name))
    # print('original_vocab_size', vocab_size)
    print('vocab_size', tokenizer.vocab_size)
    # print pad token id
    print('pad_token_id', tokenizer.pad_token_id)

    #######################
    # Load and pre-process the dataset
    #######################

    eval_dataset = get_VLA_dataset(data_args, tokenizer.eos_token, split='test', return_info=True)

    def preprocess_func(example):
        example['text'] = example['input'] + example['output']
        return example

    eval_dataset = eval_dataset.map(
        preprocess_func,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=['input'], # keep the output column
        desc="Preprocessing testing dataset",
    )

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        index = random.randint(0, len(eval_dataset))
        logger.info(f"Sample {index} from the training set:\n\n{eval_dataset[index]}")

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    # use float16 (V100 does not support bfloat16)
    torch_dtype = torch.float16 if training_args.fp16 else torch.float32

    model_kwargs = dict(
        # revision=model_args.model_revision,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        use_cache=False if training_args.gradient_checkpointing else True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    ).to(training_args.device)

    ###############
    # Do prediction
    ###############
    f = open(data_args.save_prediction_path, 'w')
    for i in range(10):
        index = random.randint(0, len(eval_dataset))
        input_text = eval_dataset[index]['text']
        print('input_text', input_text)
        input_ids = tokenizer(input_text, return_tensors='pt').input_ids
        input_ids = input_ids.to(training_args.device)
        with torch.no_grad():
            # output = model.generate(input_ids, max_length=2048, num_beams=5, early_stopping=True, output_scores=True)
            output = model.generate(input_ids, max_length=2048, num_beams=1, output_scores=True)
        output_text = tokenizer.decode(output[0], skip_special_tokens=False)
        # save the output_text
        ret = {}
        # ret['task_description'] = input_text.split('<eott_i>')[0].split('<bott_i>')[-1]
        # ret['scene_description'] = input_text.split('<eots_i>')[0].split('<bots_i>')[-1]
        ret['task_scene_description'] = input_text.split('<eots_i>')[0].split('<bots_i>')[-1]
        ret['input_clip_description'] = input_text.split('<eotp_i>')[0].split('<botp_i>')[-1]
        ret['input_video_tokens'] = [int(x[:-1]) for x in input_text.split('<eov_i>')[0].split('<bov_i>')[-1].split('<va') if x != '']
        ret['input_action_tokens'] = [int(x[:-1]) for x in input_text.split('<eoa_i>')[0].split('<boa_i>')[-1].split('<va') if x != '']
        ret['output_clip_description_pred'] = output_text.split('<eotp_o>')[0].split('<botp_o>')[-1]
        ret['output_clip_description_gt'] = eval_dataset[index]['output'].split('<eotp_o>')[0].split('<botp_o>')[-1]
        ret['output_video_tokens_pred'] = [int(x[:-1]) for x in output_text.split('<eov_o>')[0].split('<bov_o>')[-1].split('<va') if x != '']
        ret['output_video_tokens_gt'] = [int(x[:-1]) for x in eval_dataset[index]['output'].split('<eov_o>')[0].split('<bov_o>')[-1].split('<va') if x != '']
        ret['output_action_tokens_pred'] = [int(x[:-1]) for x in output_text.split('<eoa_o>')[0].split('<boa_o>')[-1].split('<va') if x != '']
        ret['output_action_tokens_gt'] = [int(x[:-1]) for x in eval_dataset[index]['output'].split('<eoa_o>')[0].split('<boa_o>')[-1].split('<va') if x != '']
        print('output_text', output_text)
        # save as jsonl file
        f.write(json.dumps(ret) + '\n')

if __name__ == "__main__":
    main()

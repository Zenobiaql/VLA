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
from src import get_VLA_dataset

# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os
import json
import time
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
from transformers import TextStreamer

logger = logging.getLogger(__name__)

def main():

    # parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    # model_args, data_args, training_args = parser.parse()

    parser = H4ArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse()

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    # logger.info(f"Training/evaluation parameters {training_args}")

    # distributed setup
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    ################
    # Load tokenizer
    ################
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    print('vocab_size', tokenizer.vocab_size)
    print('pad_token_id', tokenizer.pad_token_id)

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    # use float16 (V100 does not support bfloat16)
    torch_dtype = torch.float16

    model_kwargs = dict(
        # revision=model_args.model_revision,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        # trust_remote_code=True,
        # use_cache=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    model.eval()

    gradient_accumulation_steps = 1
    micro_batch_size_per_gpu = 1
    batch_size = gradient_accumulation_steps * micro_batch_size_per_gpu * world_size
    ds_config = {
        "fp16": {
            "enabled": True
        },
        "bf16": {
            "enabled": False
        },
        "zero_optimization": {
            "stage": 3,
            # "offload_param": {
            #     "device": "cpu",
            #     "pin_memory": True
            # },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 2e8,
            "stage3_prefetch_bucket_size": 2e8,
            "stage3_param_persistence_threshold": 1e8,
        },
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": 2000,
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size_per_gpu,
        "wall_clock_breakdown": False
    }
    dschf = HfDeepSpeedConfig(ds_config)

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    # ds_engine = deepspeed.init_inference(model, dtype=torch_dtype, 
    #                                      tensor_parallel={"tp_size": 1},
    #                                      replace_with_kernel_inject=True)
    ds_engine.module.eval()  # inference
    # streamer = TextStreamer(tokenizer)

    #######################
    # Load and pre-process the dataset
    #######################
    eval_dataset = get_VLA_dataset(data_args, tokenizer.eos_token, split='test', return_info=True)

    def preprocess_func(example):
        example['text'] = example['input']
        return example

    eval_dataset = eval_dataset.map(
        preprocess_func,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=['input'], # keep the output column
        desc="Preprocessing testing dataset",
    )

    if local_rank == 0:
        index = random.randint(0, len(eval_dataset))
        logger.info(f"Sample {index} from the training set:\n\n{eval_dataset[index]}")

    ###############
    # Do prediction
    ###############
    os.makedirs(os.path.dirname(data_args.save_prediction_path), exist_ok=True)
    f = open(data_args.save_prediction_path, 'a')
    for i in range(10):
        index = random.randint(0, len(eval_dataset))
        input_text = eval_dataset[index]['text']
        print('input_text', input_text)
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device=local_rank)
        start_time = time.time()
        with torch.no_grad():
            output = ds_engine.module.generate(input_ids, max_length=2048, num_beams=1,
                                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                               synced_gpus=True)
        
        output_text = tokenizer.decode(output[0], skip_special_tokens=False)
        if local_rank == 0:
            print('generate time', time.time() - start_time)
            print(output_text)
            # save the output_text
            ret = {}
            ret['task_description'] = input_text.split('<eott_i>')[0].split('<bott_i>')[-1]
            ret['scene_description'] = input_text.split('<eots_i>')[0].split('<bots_i>')[-1]
            # ret['task_scene_description'] = input_text.split('<eots_i>')[0].split('<bots_i>')[-1]
            ret['input_clip_description'] = input_text.split('<eotp_i>')[0].split('<botp_i>')[-1]

            ret['output_clip_description_pred'] = output_text.split('<eotp_o>')[0].split('<botp_o>')[-1]
            ret['output_clip_description_gt'] = eval_dataset[index]['output'].split('<eotp_o>')[0].split('<botp_o>')[-1]
            ret['output_clip_description_value_gt'] = eval_dataset[index]['gt_actions']

            ret['trajectory_id'] = eval_dataset[index]['trajectory_id']
            ret['view'] = eval_dataset[index]['view']

            ret['identical_token_ratio_video'], ret['identical_token_ratio_action'] = 0, 0

            ret['input_video_tokens'] = [int(x[:-1]) for x in input_text.split('<eov_i>')[0].split('<bov_i>')[-1].split('<va') if x != '']
            ret['output_video_tokens_pred'] = [int(x[:-1]) for x in output_text.split('<eov_o>')[0].split('<bov_o>')[-1].split('<va') if x != '']
            ret['output_video_tokens_gt'] = [int(x[:-1]) for x in eval_dataset[index]['output'].split('<eov_o>')[0].split('<bov_o>')[-1].split('<va') if x != '']

            ret['input_action_tokens'] = [int(x[:-1]) for x in input_text.split('<eoa_i>')[0].split('<boa_i>')[-1].split('<va') if x != '']
            ret['output_action_tokens_pred'] = [int(x[:-1]) for x in output_text.split('<eoa_o>')[0].split('<boa_o>')[-1].split('<va') if x != '']
            ret['output_action_tokens_gt'] = [int(x[:-1]) for x in eval_dataset[index]['output'].split('<eoa_o>')[0].split('<boa_o>')[-1].split('<va') if x != '']

            # print the ratio of identical tokens
            num_identical_tokens = 0
            for token_pred, token_gt in zip(ret['output_video_tokens_pred'], ret['output_video_tokens_gt']):
                if token_pred == token_gt:
                    num_identical_tokens += 1
            ret['identical_token_ratio_video'] = num_identical_tokens / len(ret['output_video_tokens_gt'])
            num_identical_tokens = 0
            for token_pred, token_gt in zip(ret['output_action_tokens_pred'], ret['output_action_tokens_gt']):
                if token_pred == token_gt:
                    num_identical_tokens += 1
            ret['identical_token_ratio_action'] = num_identical_tokens / len(ret['output_action_tokens_gt'])

            # print('output_text', output_text)
            # save as jsonl file
            f.write(json.dumps(ret) + '\n')

if __name__ == "__main__":
    main()

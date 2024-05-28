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
import mii.legacy as mii

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
    # Log on each process a small summary
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")


    ################
    # Load tokenizer
    ################
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    print('vocab_size', tokenizer.vocab_size)
    print('pad_token_id', tokenizer.pad_token_id)

    pipe = mii.pipeline(model_args.model_name_or_path)

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

    ###############
    # Do prediction
    ###############
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    os.makedirs(os.path.dirname(data_args.save_prediction_path), exist_ok=True)
    f = open(data_args.save_prediction_path, 'a')
    for i in range(10):
        index = random.randint(0, len(eval_dataset))
        input_text = eval_dataset[index]['text']
        
        start_time = time.time()
        
        output = pipe([input_text], max_length=1024)
        output_text = output[0].generated_text

        print('generate time', time.time() - start_time)

        # save the output_text
        if local_rank == 0:
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

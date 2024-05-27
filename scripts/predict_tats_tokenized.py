"""
Predict
"""

import random
import sys

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed, MistralModel, PhiModel
from transformers import TrainerCallback, TextStreamer
from time import sleep

sys.path.append('.')
from src import DataArguments, H4ArgumentParser, ModelArguments, SFTConfig, get_checkpoint, get_datasets
from src import get_VLA_dataset

import os
import json
import time

def main():

    parser = H4ArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse()

    ################
    # Load tokenizer
    ################
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # vocab_size = len(transformers.AutoTokenizer.from_pretrained(model_args.base_model_name))
    # print('original_vocab_size', vocab_size)
    print('vocab_size', tokenizer.vocab_size)
    # print pad token id
    print('pad_token_id', tokenizer.pad_token_id)
    tokenizer.padding_side = data_args.padding_side

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

    index = random.randint(0, len(eval_dataset))
    print(f"Sample {index} from the training set:\n\n{eval_dataset[index]}")

    #######################
    # Load pretrained model
    #######################
    # use float16 (V100 does not support bfloat16)
    torch_dtype = torch.float16

    model_kwargs = dict(
        # revision=model_args.model_revision,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        # trust_remote_code=True,
        use_cache=True
    )

    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    ).to(device)
    model.eval()
    streamer = TextStreamer(tokenizer)

    ###############
    # Do prediction
    ###############
    os.makedirs(os.path.dirname(data_args.save_prediction_path), exist_ok=True)
    f = open(data_args.save_prediction_path, 'w')
    for i in range(10):
        index = random.randint(0, len(eval_dataset))
        input_text = eval_dataset[index]['text']
        # print('input_text', input_text)
        input_ids = tokenizer(input_text, return_tensors='pt').input_ids
        input_ids = input_ids.to(device)
        start_time = time.time()
        with torch.no_grad():
            # output = model.generate(input_ids, max_length=2048, num_beams=5, early_stopping=True, output_scores=True)
            output = model.generate(input_ids, max_length=2048, num_beams=1, 
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                    streamer=streamer
                                    )
        print('generate time', time.time() - start_time)
        output_text = tokenizer.decode(output[0], skip_special_tokens=False)
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

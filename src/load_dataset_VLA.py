import json
import os
from datasets import Dataset, DatasetDict, IterableDataset
from torch.utils.data import DataLoader
import random
import numpy as np
import glob

def VLA_dataset_generator(shards, eos_token):
    '''
    each shard is a jsonl file, with each line containing a json object
    the json object contains the following fields:
    - trajectory_id: a integer that identifies the trajectory
    - view: a string that describes the view
    - start_frame: the start frame of the clip, -1 means it is 6 duplicate first frames
    - task_description: a string that describes the task, identical for clips with the same trajectory_id
    - scene_description: a string that describes the initial scene, identical for clips with the same trajectory_id and view
    - input_clip_description: a string that describes the frame difference in the input clip
    - output_clip_description: a string that describes the frame difference in the output clip
    - input_video_tokens: a 2D array of size 768 (256 * 3),
        256 * 3 is because each clip has 6 frames and downsamples by factor 2
    - output_video_tokens: a 2D array of size 768 (256 * 3),
    - input_action_tokens: a 2D array of size 42 (6 * 7),
    - output_action_tokens: a 2D array of size 42 (6 * 7),
    
    output:
    a generator that yields a dictionary with only the 'text' field

    text = '<bots_i>' + data['task_description'] + data['scene_description'] + '<eots_i>' + \
            '<botp_i>' + data['input_clip_description'] + '<eotp_i>' + \ 
            '<bov_i>' + ''.join([f'<va{str(x)}>' for x in data['input_video_tokens']]) + '<eov_i>' + \
            '<boa_i>' + ''.join([f'<va{str(x)}>' for x in data['input_action_tokens']]) + '<eoa_i>' + \
            '<botp_o>' + data['output_clip_description'] + '<eotp_o>' + \
            '<bov_o>' + ''.join([f'<va{str(x)}>' for x in data['output_video_tokens']]) + '<eov_o>' + \
            '<boa_o>' + ''.join([f'<va{str(x)}>' for x in data['output_action_tokens']) + '<eoa_o>' + eos_token
    '''

    for shard in shards:
        with open(shard, "r") as f:
            for line in f:
                instance_data = json.loads(line)
                text = '<bots_i>' + instance_data['task_description'] + instance_data['scene_description'] + '<eots_i>' + \
                        '<botp_i>' + instance_data['input_clip_description'] + '<eotp_i>' + \
                        '<bov_i>' + ''.join([f'<va{str(x)}>' for x in instance_data['input_video_tokens']]) + '<eov_i>' + \
                        '<boa_i>' + ''.join([f'<va{str(x)}>' for x in instance_data['input_action_tokens']]) + '<eoa_i>' + \
                        '<botp_o>' + instance_data['output_clip_description'] + '<eotp_o>' + \
                        '<bov_o>' + ''.join([f'<va{str(x)}>' for x in instance_data['output_video_tokens']]) + '<eov_o>' + \
                        '<boa_o>' + ''.join([f'<va{str(x)}>' for x in instance_data['output_action_tokens']]) + '<eoa_o>' + eos_token
                yield {"text": text}

def get_preprocessed_VLA_dataset(args, eos_token, split='train'):
    root = args.data_root
    shards = glob.glob(os.path.join(root, split, '*_stacked.jsonl'))
    shards = sorted(shards)
    ds = IterableDataset(VLA_dataset_generator, gen_kwargs={"shards": shards, "eos_token": eos_token})
    return ds
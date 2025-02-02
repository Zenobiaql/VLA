import json
import os
from datasets import Dataset, DatasetDict, IterableDataset, Dataset
from torch.utils.data import DataLoader
import random
import numpy as np
import glob

def VLA_dataset_generator(shards, eos_token, static_video_description, return_info, action_before_vision, wo_text, wo_vision, only_text):
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

    text = '<bott_i>' + data['task_description'] + '<eott_i>' + \
            '<bots_i>' + data['scene_description'] + '<eots_i>' + \
            '<botp_i>' + data['input_clip_description'] + '<eotp_i>' + \ 
            '<bov_i>' + ''.join([f'<va{str(x)}>' for x in data['input_video_tokens']]) + '<eov_i>' + \
            '<boa_i>' + ''.join([f'<va{str(x)}>' for x in data['input_action_tokens']]) + '<eoa_i>' + \
            '<botp_o>' + data['output_clip_description'] + '<eotp_o>' + \
            '<bov_o>' + ''.join([f'<va{str(x)}>' for x in data['output_video_tokens']]) + '<eov_o>' + \
            '<boa_o>' + ''.join([f'<va{str(x)}>' for x in data['output_action_tokens']) + '<eoa_o>' + eos_token
    length: 14 special tokens + 
            768 * 2 video tokens +
            42 * 2 action tokens +
            200 task description, scene description, input clip, output clip
            2 eos_token and bos_token (will be automatically added by the tokenizer)
            thus, 2048 sequence length is enough
    '''

    for shard in shards:
        with open(shard, "r") as f:
            for line in f:
                try:
                    instance_data = json.loads(line)

                    # text_input = '<bov_i>' + ''.join([f'<va{str(x)}>' for x in instance_data['input_video_tokens']]) + '<eov_i>' + '<boa_i><va0><eoa_i>'
                    # text_input += 'What should the robot arm do to ' + instance_data['task_description'] 
                    # text_input += "? Answer: <boa_o>"

                    # text_output = ''.join([f'<va{str(x)}>' for x in instance_data['output_action_tokens']]) + '<eoa_o>'

                        
                    if only_text: # For debugging: check if can train the language model correctly
                        if instance_data['input_clip_description'] == '': # sample a description for the input clip
                            instance_data['input_clip_description'] = random.choice(static_video_description)
                        text_input = '<bott_i>' + instance_data['task_description'] + '<eott_i>' + \
                                '<bots_i>' + instance_data['scene_description'] + '<eots_i>' + \
                                '<botp_i>' + instance_data['input_clip_description'] + '<eotp_i>'
                        text_output = '<botp_o>' + instance_data['output_clip_description'] + '<eotp_o>'
                    else: 
                        assert wo_text
                        if wo_text:
                            text_input = '<bott_i>' + instance_data['task_description'] + '<eott_i>'
                            text_output = ''
                        else:
                            if instance_data['input_clip_description'] == '': # sample a description for the input clip
                                instance_data['input_clip_description'] = random.choice(static_video_description)
                            text_input = '<bott_i>' + instance_data['task_description'] + '<eott_i>' + \
                                    '<bots_i>' + instance_data['scene_description'] + '<eots_i>' + \
                                    '<botp_i>' + instance_data['input_clip_description'] + '<eotp_i>'
                            text_output = '<botp_o>' + instance_data['output_clip_description'] + '<eotp_o>'
                            # if len(text_input) > 900 or len(text_output) > 800:
                            #     continue
                            if len(text_input) > 900:
                                text_input = '<bott_i>' + instance_data['task_description'] + '<eott_i>' + \
                                    '<bots_i>' + instance_data['scene_description'] + '<eots_i>' + \
                                    '<botp_i>' + '' + '<eotp_i>'
                            if len(text_output) > 800:
                                text_output = '<botp_o>' + '' + '<eotp_o>'
                        
                        # text_input += '<bov_i>' + ''.join([f'<va{str(x)}>' for x in instance_data['input_video_tokens']]) + '<eov_i>' + \
                        #             '<boa_i>' + ''.join([f'<va{str(x)}>' for x in instance_data['input_action_tokens']]) + '<eoa_i>'
                        text_input += '<bov_i>' + ''.join([f'<va{str(x)}>' for x in instance_data['input_video_tokens']]) + '<eov_i>' + '<boa_i><va0><eoa_i>'
                        text_output += '<boa_o>' + ''.join([f'<va{str(x)}>' for x in instance_data['output_action_tokens']]) + '<eoa_o>'
                    text_output += eos_token
                except:
                    continue

                # if return_info:
                #     yield {"input": text_input, "output": text_output, 
                #            "trajectory_id": instance_data['trajectory_id'], "view": instance_data['view'],
                #            "gt_actions": instance_data['gt_actions']}
                # else:
                #     yield {"input": text_input, "output": text_output}
                yield {"text": text_input + text_output}

def get_VLA_dataset(args, eos_token, split='train', return_info=False):
    if args.data_root is not None:
        root = args.data_root
        shards = glob.glob(os.path.join(root, split, '*.jsonl'))
    elif args.data_roots is not None:
        shards = []
        for root in args.data_roots:
            shards.extend(glob.glob(os.path.join(root, split, '*.jsonl')))
    else:
        assert False, 'data_root or data_roots must be provided'

    # len_shard = len(shards)
    # shards = shards[:len_shard // 2]
 
    if args.data_debug:
        shards = shards[:1]
    if args.dataset_type == 'dataset':
        ds = Dataset.from_generator(VLA_dataset_generator, gen_kwargs={"shards": shards, 
                                                            "eos_token": eos_token,
                                                            "static_video_description": args.static_video_description,
                                                            "return_info": return_info,
                                                            "action_before_vision": args.action_before_vision,
                                                            "wo_text": args.wo_text,
                                                            "wo_vision": args.wo_vision,
                                                            "only_text": args.only_text
                                                            })
    else: # iterable dataset
        ds = IterableDataset.from_generator(VLA_dataset_generator, gen_kwargs={"shards": shards, 
                                                                "eos_token": eos_token,
                                                                "static_video_description": args.static_video_description,
                                                                "return_info": return_info,
                                                                "action_before_vision": args.action_before_vision,
                                                                "wo_text": args.wo_text,
                                                                "wo_vision": args.wo_vision,
                                                                "only_text": args.only_text
                                                                })
        # ds.column_names = ['text']
    return ds
from transformers import AutoModelForCausalLM, AutoTokenizer
from alignment import get_VLA_dataset_legacy

from huggingface_hub import login
login(token='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')
import torch

device = 'cuda:0'

model_name_or_path = '/mnt/azureml/cr/j/d8c986f48e0042758991784fe953c9c3/exe/wd/data-rundong/VLA-experiments/Mistral-7B-8nodes-zero3/checkpoint-32000'
# model_name_or_path = 'mistralai/Mistral-7B-v0.1'
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

vocab_size = len(AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1"))
dataset = get_VLA_dataset_legacy(vocab_size, split='test')

def preprocess_func(example): 
    '''
    Format the example into a sequence format
    examples is a dict with the following keys:
    - text: text prompt of the manipulation task, in natural language
        since max sequence length is 2048+256=2304, its max number of tokens (for 4 input visuals + 4 output visuals)
        2304 - 2 (start&end) - 12 (special tokens) - (4+4)*256 - (4+4)*7 = 186
        (the begin of sequence token will be automatically added by the mistral tokenizer, but no for llama tokenizer)
    - task_description: task description in natural language
    - input_plan_description: input plan description in natural language
    - output_plan_description: output plan description in natural language
    - input_visual: input visual tokens for the manipulation task, in token format, e.g., <v1> <v2> <v3>
    - input_action: input action tokens for the manipulation task, in token format
    - output_visual: output visual tokens for the manipulation task, in token format
    - output_action: output action tokens for the manipulation task, in token format
    sequence format: bos (used for mistral, not used for phi) + bot_i + text + eot_i +
                    bov_i + input_visual + eov_i +
                    boa_i + input_action + eoa_i + 
                    bov_o + output_visual + eov_o +
                    boa_o + output_action + eoa_o + eos (padding will be automatically added later by the trainer)
    '''
    
    example['text'] = '<bot_i>' + example['task_description'] + example['input_plan_description'] + '<eot_i>' + \
                '<bov_i>' + ''.join(tokenizer.convert_ids_to_tokens(example['input_visual'])) + '<eov_i>' + \
                '<boa_i>' + ''.join(tokenizer.convert_ids_to_tokens(example['input_action'])) + '<eoa_i>' + \
                '<bot_o>' + example['output_plan_description'] + '<eot_o>' + \
                '<bov_o>' + ''.join(tokenizer.convert_ids_to_tokens(example['output_visual'])) + '<eov_o>' + \
                '<boa_o>' + ''.join(tokenizer.convert_ids_to_tokens(example['output_action'])) + '<eoa_o>' + \
                tokenizer.eos_token

    return example

train_dataset = train_dataset.map(
    preprocess_func,
    num_proc=data_args.preprocessing_num_workers,
    remove_columns=column_names,
    desc="Preprocessing training dataset",
)
eval_dataset = eval_dataset.map(
    preprocess_func,
    num_proc=data_args.preprocessing_num_workers,
    remove_columns=column_names,
    desc="Preprocessing testing dataset",
)

# create 6*256 random visual tokens
random_visual_tokens = torch.randint(0, 16384, (6, 256))
# convert into <vxx> format
random_visual_tokens = [f"<v{v}>" for v in random_visual_tokens.flatten().tolist()]
# create 6*7 random action tokens
random_action_tokens = torch.randint(0, 256, (6, 7))
# convert into <avxx> format
random_action_tokens = [f"<a{v}>" for v in random_action_tokens.flatten().tolist()]

prompt = "<bot_i>My favourite condiment is<eot_i><bov_i>" + "".join(random_visual_tokens) + "<eov_i><boa_i>" + "".join(random_action_tokens) + "<eoa_i>"

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=10000, do_sample=True)
out = tokenizer.batch_decode(generated_ids)[0]

print(out)
from transformers import AutoModelForCausalLM, AutoTokenizer
from alignment import DataArguments, H4ArgumentParser, ModelArguments, SFTConfig
from alignment import get_VLA_dataset_legacy
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='../data-rundong/VLA-experiments/Mistral-7B-8nodes-zero3/checkpoint-32000')
parser.add_argument('--base_model_name', type=str, default='mistralai/Mistral-7B-v0.1')
parser.add_argument('--data_root', type=str, default='/mnt/robotdata/bridge2_tokenized_legacy')
parser.add_argument('--num_visual_tokens', type=int, default=16384)
parser.add_argument('--num_action_tokens', type=int, default=256)
parser.add_argument('--num_input_frames', type=int, default=6)
parser.add_argument('--num_output_frames', type=int, default=1)
args = parser.parse_args()

def main():
	from huggingface_hub import login
	login(token='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')

	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

	vocab_size = len(AutoTokenizer.from_pretrained(args.base_model_name))
	eval_dataset = get_VLA_dataset_legacy(args, vocab_size, split='test')
	print(len(eval_dataset))

	def preprocess_func(example): 
		'''
		Format the example into a sequence format
		examples is a dict with the following keys:
		- text: text prompt of the manipulation task, in natural language
			since max sequence length is 2048, its max number of tokens is 2048 - 12 - 6*256 - 6*7 - 256 - 7 = 195
		- input_visual: input visual tokens for the manipulation task, in token format, e.g., <v1> <v2> <v3>
		- input_action: input action tokens for the manipulation task, in token format
		- output_visual: output visual tokens for the manipulation task, in token format
		- output_action: output action tokens for the manipulation task, in token format
		sequence format: bos + bot_i + text + eot_i +
						bov_i + input_visual + eov_i +
						boa_i + input_action + eoa_i + 
						bov_o + output_visual + eov_o +
						boa_o + output_action + eoa_o + eos (padding will be automatically added later by the trainer)
		'''
		example['text'] = '<bot_i>' + example['text'] + '<eot_i>' + \
					'<bov_i>' + ''.join(tokenizer.convert_ids_to_tokens(example['input_visual'])) + '<eov_i>' + \
					'<boa_i>' + ''.join(tokenizer.convert_ids_to_tokens(example['input_action'])) + '<eoa_i>' 

		return example

	eval_dataset = eval_dataset.map(
		preprocess_func,
		remove_columns=['input_visual', 'input_action', 'output_visual', 'output_action'],
		num_proc=12,
		desc="Preprocessing testing dataset",
	)

	device = 'cuda:0'

	# sample a prompt from the eval_dataset
	input_prompt = eval_dataset[0]['text']
	print(input_prompt)

	model_inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
	print(model_inputs)

	# model_name_or_path = 'mistralai/Mistral-7B-v0.1'
	model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)

	generated_ids = model.generate(**model_inputs, max_new_tokens=2500, do_sample=True)
	out = tokenizer.batch_decode(generated_ids)[0]

	print(out)

def generate_random_prompt():
	# create 6*256 random visual tokens
	random_visual_tokens = torch.randint(0, 16384, (6, 256))
	# convert into <vxx> format
	random_visual_tokens = [f"<v{v}>" for v in random_visual_tokens.flatten().tolist()]
	# create 6*7 random action tokens
	random_action_tokens = torch.randint(0, 256, (6, 7))
	# convert into <avxx> format
	random_action_tokens = [f"<a{v}>" for v in random_action_tokens.flatten().tolist()]

	prompt = "<bot_i>My favourite condiment is<eot_i><bov_i>" + "".join(random_visual_tokens) + "<eov_i><boa_i>" + "".join(random_action_tokens) + "<eoa_i>"

	return prompt

if __name__ == '__main__':
	main()
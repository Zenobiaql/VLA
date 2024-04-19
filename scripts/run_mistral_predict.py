from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import login
login(token='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')
import torch

device = 'cuda:0'

model_name_or_path = '/mnt/azureml/cr/j/068ff077946449488ff4cadffbe3a3a7/exe/wd/1node/checkpoint-250'
# model_name_or_path = 'mistralai/Mistral-7B-v0.1'
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

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
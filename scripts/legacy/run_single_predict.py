from transformers import AutoTokenizer, AutoModelForCausalLM
from alignment import get_VLA_dataset_legacy

pretrained_model_path = '/mnt/azureml/cr/j/d8c986f48e0042758991784fe953c9c3/exe/wd/data-rundong/VLA-experiments/Mistral-7B-8nodes-zero3/checkpoint-32000'

model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

vocab_size = len(AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1"))

dataset = get_VLA_dataset_legacy(vocab_size, split='test')

prompt = "This is an example script ."
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
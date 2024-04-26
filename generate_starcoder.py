import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from tqdm import tqdm

base_model = 'bigcode/starcoder2-7b'
model = AutoModelForCausalLM.from_pretrained(
base_model,
load_in_8bit=True,
torch_dtype=torch.float16,
device_map='auto')
tokenizer=AutoTokenizer.from_pretrained(base_model)

# Load the finetuned model
# Comment out these two commands to make use of the pretrained model
output_dir = "./10-707 Model Checkpoints/starcoder2_checkpoint-400"
model = PeftModel.from_pretrained(model, output_dir)

# Set the prompt
instructions = """\n
# Unit tests for the above function
import unittest

class Tests(unittest.TestCase):
"""
data = pd.read_parquet('human_eval.parquet')
data['function_signature'] = [func[func.find('def'):][:func[func.find('def'):].find('\n')+1] for func in data['prompt']]
functions = data['function_signature'] + data['canonical_solution']

# Generate tests for each code sample
i = 0
for function in tqdm(functions):
    prompt = function + instructions
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to('cuda')
    generate_ids = model.generate(inputs.input_ids, pad_token_id=tokenizer.eos_token_id, max_length=500)
    ans = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    ans = ans[len(function):]
    with open('starcoder_finetuned.txt', 'a') as f:
        f.write("Task_id: " + str(i) + "\n" + ans + "\n")
    f.close()
    i += 1
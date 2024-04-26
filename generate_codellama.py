import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from tqdm import tqdm

# Load the baseline model
base_model = 'codellama/CodeLlama-7b-hf'
model = AutoModelForCausalLM.from_pretrained(
base_model,
load_in_8bit=True,
torch_dtype=torch.float16,
device_map='auto')
tokenizer=AutoTokenizer.from_pretrained(base_model)

# Load the finetuned model
# Comment out these two commands to make use of the pretrained model
output_dir = "./10-707 Model Checkpoints/codellama_checkpoint-200"
model = PeftModel.from_pretrained(model, output_dir)

# Set the prompt
prompt_base = """
[INST] Your task is to write 5 tests to check the correctness of a function that solves a programming
problem.
The tests must be between [TESTS] and [/TESTS] tags.
You must write the comment "#Test case n:" on a separate line directly above each assert statement,
where n represents the test case number, starting from 1 and increasing by one for each subsequent
test case.
Function:
"""

# Generate tests for each code sample
data = pd.read_parquet('human_eval.parquet')
data['function_signature'] = [func[func.find('def'):][:func[func.find('def'):].find('\n')+1] for func in data['prompt']]
functions = data['function_signature'] + data['canonical_solution']

i = 0
for function in tqdm(functions):
    prompt = prompt_base + function + "[/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to('cuda')
    generate_ids = model.generate(inputs.input_ids, pad_token_id=tokenizer.eos_token_id, max_length=500)
    ans = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    ans = ans[len(prompt):]
    with open('generations_finetuned.txt', 'a') as f:
        f.write("Task_id: " + str(i) + "\n" + ans + "\n")
    f.close()
    i += 1
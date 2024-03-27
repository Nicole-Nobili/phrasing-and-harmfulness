import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from tqdm.auto import tqdm
from utils import load_model
tqdm.pandas()

# Generation function
@torch.no_grad()
def generate(model, tokenizer, messages):
    prompts = [f"A chat between a user and an AI assistant. The assistant answers the user's questions.\n\n### User: {message}\n### Assistant:" for message in messages]

    tokens = tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding=True).to(model.device)
    generated_ids = model.generate(**tokens, max_new_tokens=32, do_sample=True, top_p=1, temperature=0.1, pad_token_id=tokenizer.eos_token_id)

    return [i.split('Assistant: ')[-1] for i in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]

####################
# EVAL GENERATIONS #
####################

# Load data
eval_dataset = pd.read_csv('data/speech_acts.csv')

results_cols = []
for col in eval_dataset.columns:
    results_cols.append(col)
    results_cols.append('answer_' + col)

batch_size = 32

# Run generations
result_list = []

for hf_model in ['meta-llama/Llama-2-7b-hf', 'mistralai/Mistral-7B-v0.1']: 
    model_name = hf_model.split('/')[-1]
    print(f"Running {model_name}")
    adapters = []
    for ds in ['int', 'dec', 'imp', 'all']:
        for safety in ['s15']: # 's5'
            for rs in range(3):
                adapters.append(f"speech-acts/{model_name}-lora-{ds}-{safety}-rs-{rs+1}")

    for rs in range(3):
        adapters.append(f"speech-acts/{model_name}-lora-base-rs-{rs+1}")

    for adapter in tqdm(adapters):
        try:
            # Model loading
            model, tokenizer = load_model(hf_model, adapter, device='cuda:0')
            
            # New df to store model genrations
            results = pd.DataFrame(columns=['model_name'] + results_cols)
            
            # Run thrugh each column of the speech acts dataset
            for prompt_column in eval_dataset.columns:
                results[prompt_column] = eval_dataset[prompt_column]
                generations = []
                for b in range(len(eval_dataset) // batch_size + 1): 
                    generations += generate(model, tokenizer, eval_dataset[prompt_column].iloc[b*batch_size:(b+1)*batch_size].tolist())
                results["answer_" + prompt_column] = generations
            
            results["model_name"] = adapter.split('/')[-1]
            result_list.append(results)
            del model, tokenizer
        except Exception as e:
            print(f"Some problem occurred with: {adapter}\n{e}")
    
merged_results = pd.concat(result_list, ignore_index=True)
merged_results.to_csv(f"data/speech_acts_generations.csv", index=False)
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from tqdm.auto import tqdm

tqdm.pandas()

# Load model
def load_model(model_name, adapter_model="", dtype=torch.bfloat16, device='auto'):
    print("Loading the model...")
    if model_name == "": model_name = model_name

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device
    )
    peft_model = PeftModel.from_pretrained(model, adapter_model).merge_and_unload()
    del model

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if 'llama' or 'mistral' in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = 'left' 

    return peft_model, tokenizer

# Generation function
@torch.no_grad()
def generate(messages):
    prompts = [f"A chat between a user and an AI assistant. The assistant answers the user's questions.\n\n### User: {message}\n### Assistant:" for message in messages]

    tokens = tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding=True)['input_ids']
    generated_ids = model.generate(tokens.to(model.device), max_new_tokens=16, do_sample=True, top_p=1, temperature=0.1, pad_token_id=tokenizer.eos_token_id)

    return [i.split('Assistant: ')[-1] for i in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]

####################
# EVAL GENERATIONS #
####################

# Load data
eval_dataset = pd.read_csv('data/speech_acts.csv').drop(['Opinions', 'Presuppositions'], axis=1)

# create a new dataset with the same columns as the eval dataset and with another column called model name
# this dataset will be used to store the results of the model
results_cols = []
for col in eval_dataset.columns:
    results_cols.append(col)
    results_cols.append('answer_' + col)

results = pd.DataFrame(columns=['model_name'] + results_cols)

batch_size = 32

# Run generations
for hf_model in ['meta-llama/Llama-2-7b-hf', 'mistralai/Mistral-7B-v0.1']: 
    model_name = hf_model.split('/')[-1]
    print(f"Running {model_name}")
    adapters = []
    for ds in ['base', 'int', 'dec', 'imp', 'all']:
        for rs in range(3):
            adapters.append(f"speech-acts/{model_name}-lora-{ds}-rs-{rs+1}")

    result_list = []

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
                    generations += generate(eval_dataset[prompt_column].iloc[b*batch_size:(b+1)*batch_size])
                results["answer_" + prompt_column] = generations
            
            results["model_name"] = adapter.split('/')[-1]
            result_list.append(results)
            del model, tokenizer
        except Exception as e:
            print(f"Some problem occurred with: {adapter}")
    
    merged_results = pd.concat(result_list, ignore_index=True)
    merged_results.to_csv(f"data/eval/{model_name}.csv", index=False)
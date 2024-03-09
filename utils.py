import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(model_name, adapter_model="", dtype=torch.bfloat16, device='auto'):
    print("Loading the model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device
    )
    if adapter_model != "":
        model = PeftModel.from_pretrained(hf_model, adapter_model).merge_and_unload()
        del hf_model
    else:
        model = hf_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if 'llama' or 'mistral' in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = 'left' 

    return model, tokenizer

@torch.no_grad()
def run_conversion(model, tokenizer, prompts, message):
    encodeds = []
    for prompt in prompts:
        messages = [
            {"role": "user", "content": message.format(prompt)},
            {"role": "assistant", "content": "Assistant:"}
        ]
        encodeds.append(tokenizer.apply_chat_template(messages, tokenize=False)[:-5])

    encodeds = tokenizer.batch_encode_plus(encodeds, return_tensors='pt', padding=True)['input_ids']
    generated_ids = model.generate(encodeds.to(model.device), max_new_tokens=64, do_sample=True, temperature=0.5, pad_token_id=tokenizer.eos_token_id)

    return [i.split('Assistant: ')[-1] for i in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]
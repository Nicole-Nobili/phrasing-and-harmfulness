from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_model(model_name, adapter_model="", device='cpu', dtype=torch.bfloat16):
    print("Loading the model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype
    )
    peft_model = PeftModel.from_pretrained(model, adapter_model).merge_and_unload().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not getattr(tokenizer, "pad_token", None):
        print('Setting pad token to eos token')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = 'left' 

    return peft_model, tokenizer
from openai import OpenAI
from anthropic import Anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import torch
import os
import time
from dotenv import load_dotenv
load_dotenv()
tqdm.pandas()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
anyscale = OpenAI(
           base_url = "https://api.endpoints.anyscale.com/v1",
           api_key=os.environ['ANYSCALE_API_KEY'],
        )

#model = "meta-llama/Llama-2-7b-chat-hf"
#llama, tokenizer = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).cuda(), AutoTokenizer.from_pretrained(model)

gsm8k = pd.read_csv("gsm8k.csv")

def claude_answer(row):
    try:
        response = anthropic.messages.create(
            model="claude-3-opus-20240229",
            messages=[{
                "role": "user",
                "content": [{
                    "type": "text", 
                    "text": f"Answer the following questions as shown in the examples.\n\nExamples:\nQ: The profit from a business transaction is shared among 2 business partners, Mike and Johnson in the ratio 2:5 respectively. If Johnson got $2500, how much will Mike have after spending some of his share on a shirt that costs $200?\nA: According to the ratio, for every 5 parts that Johnson gets, Mike gets 2 parts Since Johnson got $2500, each part is therefore $2500/5 = $<<2500/5=500>>500 Mike will get 2*$500 = $<<2*500=1000>>1000 After buying the shirt he will have $1000-$200 = $<<1000-200=800>>800 left #### 800\n\nQ: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nA: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10\n\nQ: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?\nA: In one hour, there are 3 sets of 20 minutes. So, Joy can read 8 x 3 = <<8*3=24>>24 pages in an hour. It will take her 120/24 = <<120/24=5>>5 hours to read 120 pages. #### 5\n\nAnswer this question as shown in the examples. Separate the final answer with ####.\nQ: {row}"
                    }]
                }],
            temperature=0,
            max_tokens=512
        )
        output = response.content[0].text
    except:
        output = "Error"
    
    time.sleep(0.5)
    return output

def gpt4_answer(row):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "user",
                    "content": f"Answer the following questions as shown in the examples.\n\nExamples:\nQ: The profit from a business transaction is shared among 2 business partners, Mike and Johnson in the ratio 2:5 respectively. If Johnson got $2500, how much will Mike have after spending some of his share on a shirt that costs $200?\nA: According to the ratio, for every 5 parts that Johnson gets, Mike gets 2 parts Since Johnson got $2500, each part is therefore $2500/5 = $<<2500/5=500>>500 Mike will get 2*$500 = $<<2*500=1000>>1000 After buying the shirt he will have $1000-$200 = $<<1000-200=800>>800 left #### 800\n\nQ: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nA: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10\n\nQ: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?\nA: In one hour, there are 3 sets of 20 minutes. So, Joy can read 8 x 3 = <<8*3=24>>24 pages in an hour. It will take her 120/24 = <<120/24=5>>5 hours to read 120 pages. #### 5\n\nAnswer this question as shown in the examples. Separate the final answer with ####.\nQ: {row}"
                }
            ],
            temperature=0,
            max_tokens=512
        )
        output = response.choices[0].message.content
    except:
        output = "Error"
    
    time.sleep(0.5)
    return output

def llama_answer(row):
    prompt = tokenizer.apply_chat_template([
        {
            "role": "user", 
            "content": f"Answer the following questions as shown in the examples.\n\nExamples:\nQ: The profit from a business transaction is shared among 2 business partners, Mike and Johnson in the ratio 2:5 respectively. If Johnson got $2500, how much will Mike have after spending some of his share on a shirt that costs $200?\nA: According to the ratio, for every 5 parts that Johnson gets, Mike gets 2 parts Since Johnson got $2500, each part is therefore $2500/5 = $<<2500/5=500>>500 Mike will get 2*$500 = $<<2*500=1000>>1000 After buying the shirt he will have $1000-$200 = $<<1000-200=800>>800 left #### 800\n\nQ: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nA: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10\n\nQ: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?\nA: In one hour, there are 3 sets of 20 minutes. So, Joy can read 8 x 3 = <<8*3=24>>24 pages in an hour. It will take her 120/24 = <<120/24=5>>5 hours to read 120 pages. #### 5\n\nAnswer this question as shown in the examples. Separate the final answer with ####.\nQ: {row}"
        }], tokenize=False)
    tokens = tokenizer(prompt, return_tensors="pt").to('cuda')
    try:
        output = llama.generate(
            **tokens, 
            max_new_tokens=512,
            temperature=0.0, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
        output = tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        output = f"Error - {e}"

    return output

def anyscale_answer(row, model):
    response = anyscale.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Answer the following questions as shown in the examples.\n\nExamples:\nQ: The profit from a business transaction is shared among 2 business partners, Mike and Johnson in the ratio 2:5 respectively. If Johnson got $2500, how much will Mike have after spending some of his share on a shirt that costs $200?\nA: According to the ratio, for every 5 parts that Johnson gets, Mike gets 2 parts Since Johnson got $2500, each part is therefore $2500/5 = $<<2500/5=500>>500 Mike will get 2*$500 = $<<2*500=1000>>1000 After buying the shirt he will have $1000-$200 = $<<1000-200=800>>800 left #### 800\n\nQ: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nA: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10\n\nQ: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?\nA: In one hour, there are 3 sets of 20 minutes. So, Joy can read 8 x 3 = <<8*3=24>>24 pages in an hour. It will take her 120/24 = <<120/24=5>>5 hours to read 120 pages. #### 5\n\nAnswer this question as shown in the examples. Separate the answer with ####.\nQ: {row}"
            }
        ],
        temperature=0,
        max_tokens=512
    )
    output = response.choices[0].message.content

    return output

#gsm8k["imperative_answer_gpt4"] = gsm8k["imperative"].progress_apply(gpt4_answer)
#gsm8k["interrogative_answer_gpt4"] = gsm8k["interrogative"].progress_apply(gpt4_answer)

#gsm8k["imperative_answer_claude"] = gsm8k["imperative"].progress_apply(claude_answer)
#gsm8k["interrogative_answer_claude"] = gsm8k["interrogative"].progress_apply(claude_answer)

#gsm8k["imperative_answer_llama"] = gsm8k["imperative"].progress_apply(llama_answer)
#gsm8k["interrogative_answer_llama"] = gsm8k["interrogative"].progress_apply(llama_answer)


gsm8k["imperative_answer_llama7b"] = gsm8k["imperative"].progress_apply(anyscale_answer, model="meta-llama/Llama-2-7b-chat-hf")
gsm8k["interrogative_answer_llama7b"] = gsm8k["interrogative"].progress_apply(anyscale_answer, model="meta-llama/Llama-2-7b-chat-hf")

gsm8k["imperative_answer_llama13b"] = gsm8k["imperative"].progress_apply(anyscale_answer, model="meta-llama/Llama-2-13b-chat-hf")
gsm8k["interrogative_answer_llama13b"] = gsm8k["interrogative"].progress_apply(anyscale_answer, model="meta-llama/Llama-2-13b-chat-hf")

gsm8k["imperative_answer_llama70b"] = gsm8k["imperative"].progress_apply(anyscale_answer, model="meta-llama/Llama-2-70b-chat-hf")
gsm8k["interrogative_answer_llama70b"] = gsm8k["interrogative"].progress_apply(anyscale_answer, model="meta-llama/Llama-2-70b-chat-hf")

gsm8k.to_csv("gsm8k_llama.csv", index=False)
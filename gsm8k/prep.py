from datasets import load_dataset
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import re
import random
import os
tqdm.pandas()

gsm8k = pd.DataFrame(load_dataset("gsm8k", "main")["train"]).sample(200, random_state=42).reset_index(drop=True)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def int2imp(row):
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
            "role": "user",
            "content": f"You convert questions into instructions keeping the same meaning. Modify the sentence as least as possible.\n\nExamples:\nQ: The profit from a business transaction is shared among 2 business partners, Mike and Johnson in the ratio 2:5 respectively. If Johnson got $2500, how much will Mike have after spending some of his share on a shirt that costs $200?\nI: The profit from a business transaction is shared among 2 business partners, Mike and Johnson in the ratio 2:5 respectively. Compute how much Mike will have after spending some of his share on a shirt that costs $200 if Johnson got $2500.\n\nQ: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nI: Weng earns $12 an hour for babysitting. Calculate how much she earned yesterday by doing 50 minutes of babysitting.\n\nQ: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?\t\nI: Joy can read 8 pages of a book in 20 minutes. Calculate how many hours it will take her to read 120 pages.\n\nConvert this question as shown in the examples.\nQ: {row}"
            }
        ],
        temperature=0,
        max_tokens=128,
        top_p=0.3,
        frequency_penalty=0,
        presence_penalty=0
        )
    return response.choices[0].message.content

def imp2int(row):
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
            "role": "user",
            "content": f"You convert instructions into questions keeping the same meaning. Modify the sentence as least as possible.\n\nExamples:\nI: The profit from a business transaction is shared among 2 business partners, Mike and Johnson in the ratio 2:5 respectively. Compute how much Mike will have after spending some of his share on a shirt that costs $200 if Johnson got $2500.\nQ: The profit from a business transaction is shared among 2 business partners, Mike and Johnson in the ratio 2:5 respectively. If Johnson got $2500, how much will Mike have after spending some of his share on a shirt that costs $200?\n\nI: Weng earns $12 an hour for babysitting. Calculate how much she earned yesterday by doing 50 minutes of babysitting.\nQ: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n\nI: Joy can read 8 pages of a book in 20 minutes. Calculate how many hours it will take her to read 120 pages.\nQ: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?\t\n\nConvert this question as shown in the examples.\nI: {row}"
            }
        ],
        temperature=0,
        max_tokens=128,
        top_p=0.3,
        frequency_penalty=0,
        presence_penalty=0
        )
    return response.choices[0].message.content

gsm8k["imperative"] = gsm8k["question"].progress_apply(lambda x: int2imp(x) if x.endswith("?") else x)
gsm8k["interrogative"] = gsm8k["question"].progress_apply(lambda x: imp2int(x) if x.endswith(".") else x)

# IMP -> DEC
pattern = r'\b(determine|compute|calculate)\b'
replacements = [r'I need to \1', r'I have to \1']

def imp2dec(x):
    def replace_with_pattern(match):
        replacement_pattern = random.choice(replacements)
        return replacement_pattern.replace(r'\1', match.group().lower())

    return re.sub(pattern, replace_with_pattern, x, flags=re.IGNORECASE)

gsm8k["declarative"] = gsm8k["imperative"].apply(imp2dec)

gsm8k.to_csv("gsm8k.csv", index=False)
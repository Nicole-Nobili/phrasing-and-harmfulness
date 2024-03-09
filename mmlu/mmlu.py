from openai import OpenAI
from datasets import load_dataset

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

tqdm.pandas()
#%%
dataset = load_dataset("cais/mmlu", "all")
filtered_dataset = dataset.filter(lambda example: len(example["question"]) < 128)
#%%
sampled_dataset = filtered_dataset.shuffle(seed=42)['auxiliary_train'][:100]
questions = np.array(sampled_dataset["question"])
choices = np.array(sampled_dataset["choices"])
answers = np.array(sampled_dataset["answer"])

data = {"question": questions, "choice1": choices[:, 0], "choice2": choices[:, 1], "choice3": choices[:, 2], "choice4": choices[:, 3], 'answer': answers}
df = pd.DataFrame(data)

df.to_csv("mmlu_sample.csv", index=False)
# %%
client = OpenAI(api_key="...")

content = """Given a sentences and a set of possible answers, transform the sentence into a plain question.

Here are some examples:
SENTENCE: Some birds are
CHOICES: (A) people (B) creatures (C) fish (D) solar
CONVERTED: What are some birds?

SENTENCE: A reptile's body temperature
CHOICES: (A) will sync with their climate (B) will keep stable under any circumstances (C) reacts as other warm blooded animals temperature would (D) plunges rapidly in warm climates
CONVERTED: What does a reptile's body temperature do?

SENTENCE: If this fell on you, you would probably die
CHOICES: (A) a leaning tower (B) a balloon (C) a feather (D) a towel
CONVERTED: Which one, if fallen on you, would probably kill you?

You don't have to answer, just convert the SENTENCE.
"""
# %%
def convert_mmlu(row, model="gpt-4-turbo-preview"):

    response = client.chat.completions.create(
    model=model,
    messages=[
        {
        "role": "system",
        "content": content
        },
        {
        "role": "user",
        "content": f"SENTENCE: {row['question']}\nCHOICES: (A) {row['choice1']} (B) {row['choice2']} (C) {row['choice3']} (D) {row['choice4']}\nCONVERTED:"
        }
    ],
    temperature=0.3,
    max_tokens=64,
    top_p=0.5,
    seed=42
    )

    return response.choices[0].message.content
# %%
df['Converted'] = df.progress_apply(convert_mmlu, axis=1)
df['Interrogative'] = df['Converted'].apply(lambda x: x.split("CONVERTED: ")[-1]).apply(lambda x: x if x[-1] == '?' else x + '?')
# %%
int2dec_sys = """You are a very diligent AI assistant. Do the following conversions:

User: What do national parks have rules for?
Assistant: National parks have rules for...

User: What is a disperser able to do?
Assistant: A disperser is able to perform...

User: When do nocturnal predators hunt?
Assistant: Nocturnal predators hunt during..."""

def int2dec(prompt, model="gpt-4-turbo-preview"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
            "role": "system",
            "content": int2dec_sys
            },
            {
            "role": "user",
            "content": f"User: {prompt}"
            }
        ],
        temperature=0.3,
        max_tokens=128,
        top_p=0.5,
        seed=42
    )

    return response.choices[0].message.content

int2imp_sys = """You are a very diligent AI assistant. Convert questions into instructions:

User: What do national parks have rules for?
Assistant: Tell me which of followings are rules of national parks.

User: What is a disperser able to do?
Assistant: Explain what a disperser can do.

User: When do nocturnal predators hunt?
Assistant: Tell me when nocturnal predators hunt."""

def int2imp(prompt, model="gpt-4-turbo-preview"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
            "role": "system",
            "content": int2imp_sys
            },
            {
            "role": "user",
            "content": f"User: {prompt}"
            }
        ],
        temperature=0.3,
        max_tokens=128,
        top_p=0.5,
        seed=42
    )

    return response.choices[0].message.content
# %%
df['Declarative'] = df['Interrogative'].progress_apply(int2dec)
df['Imperative'] = df['Interrogative'].progress_apply(int2imp)
# %%
df.rename(columns={"question": "Original"})
df.drop('Converted', axis=1).to_csv("mmlu_conv.csv", index=False)
# %%

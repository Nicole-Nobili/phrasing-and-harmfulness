# %%
from datasets import load_dataset
import re
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

dataset = load_dataset("allenai/WildChat")['train']
# %%
# Sample 10.000 examples 
filtered = dataset.remove_columns(['conversation_id', 'timestamp', 'openai_moderation', 'detoxify_moderation', 'redacted']).filter(lambda example: example['model'] == 'gpt-4' and example['language'] == 'English' and example['toxic'] == False)
sample = filtered.shuffle(seed=42).select(range(10000))
# %%
# Extract 'content' from the first dict in 'conversation' and create a new column
sample = sample.map(lambda example: {'prompt': example['conversation'][0]['content']})
# %%
def split_into_sentences(example, idx):
    sentences = re.split(r'(?<=[.!?])(?=\s+[A-Z])|(?<=[.!?])\s+', example['prompt'])
    return {'sentences': [{'id': idx, 'sentence': sentence} for sentence in sentences]}

sample = sample.map(split_into_sentences, with_indices=True)

# Flatten the list of dictionaries into a DataFrame
sentences_df = pd.DataFrame([item for sublist in sample['sentences'] for item in sublist])
mask = sentences_df['sentence'] == ''
sentences_df = sentences_df.loc[~mask].dropna()
# %%
import numpy as np
m = 2

filtered_df = sentences_df.copy()
sent_count = sentences_df.groupby('id').size()
filtered_df = sentences_df.groupby('id').apply(lambda x: pd.concat([x.iloc[:m], x.iloc[-m:]]) if len(x) > 2*m else x).reset_index(drop=True)
filtered_df['sentence'] = filtered_df['sentence'].apply(lambda x: x[:500] if len(x) > 500 else x)

for i in sent_count.index:
    if sent_count.loc[i] > 2*m:
        idx = filtered_df.loc[filtered_df['id'] == i, 'sentence'].index[2]
        filtered_df.loc[idx, 'sentence'] += '\n[...]\n'

grouped_sentences = filtered_df.groupby('id')['sentence'].apply('\n'.join).reset_index()
# %%

### Run conversion
from anthropic import Anthropic
import os

api_key = os.environ['ANTHROPIC_API_KEY']
client = Anthropic(api_key=api_key)

template = "You have to recognize whether a given sentence is either interrogative (INT), declarative (DEC), or imperative (IMP). If all of the previous categories don't fit, you output NAN.\nExamples:\nUser: Write a full book.\nAssistant: IMP\nUser: How would this alternate version play out?\nAssistant: INT\nUser: I think there’s a simpler explanation.\nAssistant DEC\nUser: [...] On a scale of 1 to 10, how funny is the joke?\nAssistant: INT\nUser: I would like to have the map of Europe [...].\nAssistant DEC\nUser: [...] I need help on this.\nAssistant DEC\nEven if only a part of the sentence is related to a given type (INT - DEC - IMP), please return the actual type.\n\nUser: {sentence}"

def get_speech_act(sentence):
    try:
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2,
            temperature=0,

            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": template.format(sentence=sentence)}]
                }
            ]
        )
        return message.content[0].text
    except Exception as e:
        return 'ERR'

test = grouped_sentences.copy().sample(1000, replace=False, random_state=42)
test['response'] = test['sentence'].progress_apply(get_speech_act)
test['speech_act'] = test['response'].apply(lambda x: x.split('\n')[0])

test.to_csv('wildchat_speechacts.csv', index=False)
# %%

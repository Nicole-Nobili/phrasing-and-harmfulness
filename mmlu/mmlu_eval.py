# %% 
from openai import OpenAI
from anthropic import Anthropic
import argparse
import pandas as pd
from utils import load_model
from tqdm.auto import tqdm
import google.generativeai as genai

tqdm.pandas()
# %% 
def find_answer(row, model_name='gpt4', api_key=None, model=None):
    """
    Finds the most likely answer to a given question among the provided choices using a specified model.
    
    Parameters:
    - question: A string containing the question, declarative, or imperative.
    - choices: A list of strings containing the possible choices.
    - model_name: A string specifying which model to use ('hf', 'gpt4', 'claude').

    Returns:
    - The index of the most likely answer among the choices.
    """

    global answers

    for stype in ['Original', 'Interrogative', 'Declarative', 'Imperative']:

        prompt = f"""Question: {row[stype]}\n\nWhich one of the four choices completes the question correctly, (A), (B), (C) or (D)?\n\nChoices:\n(A) {row['Choice1']}\n(B) {row['Choice2']}\n(C) {row['Choice3']}\n(D) {row['Choice4']}\n\nAnswer only with the right letter.\nAnswer:"""

        if model_name == 'gpt4':
            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                "role": "user",
                "content": prompt
                }
            ],
            temperature=0,
            max_tokens=4,
            seed=42
            )

            output = response.choices[0].message.content
        
        elif model_name == 'claude3':
            client = Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4,
                temperature=0,

                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
            )
            print(message.content)
            output = message.content.text

        elif model_name =='gemini':
            genai.configure(api_key=api_key)

            # Set up the model
            generation_config = {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 4,
            }

            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ]

            try:
                model = genai.GenerativeModel(model_name="gemini-1.0-pro", generation_config=generation_config, safety_settings=safety_settings)

                convo = model.start_chat(history=[
                ])

                convo.send_message(prompt)
                output = convo.last.text
            except:
                output = "Error"
            
        else:
            prompt = f"""A chat between a user and an AI assistant. The assistant answers the user's questions.\n\n### User: Which one of the four choices completes the question correctly, (A), (B), (C) or (D)?\n\nQuestion: {row[stype]}\nChoices:\n(A) {row['Choice1']}\n(B) {row['Choice2']}\n(C) {row['Choice3']}\n(D) {row['Choice4']}\n\nAnswer only with the right letter.\nAssistant:"""
            tokens = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
            output = model.generate(tokens, max_new_tokens=4, temperature=0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            output = tokenizer.decode(output[0], skip_special_tokens=True)

        answers[f"{stype} Answer"].append(output)


parser = argparse.ArgumentParser(description="Find the most likely answer to a given question using a specified model.")
parser.add_argument("--model", type=str, default="gpt4", help="The model to use ('hf', 'gpt4', 'claude3').")
parser.add_argument("--api_key", type=str, help="The API key for the model (if required).")

args = parser.parse_args()

# Load the model and tokenizer
if 'llama' in args.model.lower():
    base_model = "meta-llama/Llama-2-7b-hf"
if 'mistral' in args.model.lower():
    base_model = "mistralai/Mistral-7B-v0.1"

if args.model not in ['gpt4', 'claude3', 'gemini']:
    if args.model == base_model:
        model, tokenizer = load_model(args.model)
    else:
        model, tokenizer = load_model(base_model, adapter_model=args.model)
else:
    model = None

answers = {
    f'Original Answer': [],
    f'Interrogative Answer': [],
    f'Declarative Answer': [],
    f'Imperative Answer': []
}
# %% 
# Load the data
df = pd.read_csv('mmlu_conv_hand.csv')
model_name = args.model.split('/')[-1]

# Find the most likely answer for each question
df.progress_apply(find_answer, axis=1, model_name=args.model, api_key=args.api_key, model=model)
df = pd.concat([df, pd.DataFrame(answers)], axis=1)
df.to_csv(f'mmlu_{model_name}.csv', index=False)

import pandas as pd
import json
import sys
import termios
import tty
from sklearn.metrics import confusion_matrix
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# For Windows
# import msvcrt

# For Unix/Linux
# import termios, tty, sys

def getch():
    """
    Read a single character from standard input.
    For Windows, this function uses msvcrt.getch().
    For Unix/Linux, it uses termios and tty to get a character without requiring Enter to be pressed.
    """
    # For Windows
    # return msvcrt.getch().decode()

    # For Unix/Linux
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

correction = pd.read_json('human_eval.json')
df = pd.read_csv('wildchat_speechacts.csv').dropna().reset_index(drop=True).iloc[:len(correction)]
df['correct'] = correction['key_press']
# Create an empty list to store the entries and key presses
entries_and_presses = []

print(f"DF length: {len(df)}\nCorrection length: {(df['correct'] == 'n').sum()}")

labels = []

print("Correct the following sentences by classifying them as:\nInterrogative (INT) - press 1\nDeclarative (DEC) - press 2\nImperative (IMP) - press 3\nOther (NAN) - press 4")
for i in range(len(df)):
    if df.iloc[i]['correct'] == 'y':
        labels.append(df.iloc[i]['speech_act'])
    else:
        sent_id = df.iloc[i]['id']

        sent = df.iloc[i]['sentence']
        pred = df.iloc[i]['speech_act']
        print(f"\n### ID {sent_id} ###\n\n{sent}\n\n-- {pred} --\n\n")
        key_press = getch()

        # Check if the key press is valid
        while key_press not in ['1', '2', '3', '4', 'q']:
            print("Invalid input. Please try again.\n")
            key_press = getch()

        # Add the entry and key press to the list
        if key_press == 'q':
            break
        elif key_press == '1':
            labels.append('INT')
        elif key_press == '2':
            labels.append('DEC')
        elif key_press == '3':
            labels.append('IMP')
        elif key_press == '4':
            labels.append('NAN')


def convert_label(label):
    if label not in ['INT', 'DEC', 'IMP', 'NAN']:
        return 'NAN'
    else:
        return label

df['label'] = labels
df['speech_act'] = df['speech_act'].apply(str.strip).apply(convert_label)

print(df['label'].value_counts().apply(lambda x: str(np.round(x / len(df) * 100, 1))+'%'))

all_df = pd.read_csv('wildchat_speechacts.csv').dropna().reset_index(drop=True)
all_df['speech_act'] = all_df['speech_act'].apply(str.strip).apply(convert_label)
print(all_df['speech_act'].value_counts().apply(lambda x: str(np.round(x / len(all_df) * 100, 1))+'%'))

df.to_csv('human_eval_df.csv', index=False)
print("Corrections saved in 'human_eval_df.csv' file.")
import pandas as pd
import json
import sys
import termios
import tty

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

df = pd.read_csv('wildchat_speechacts.csv').dropna().reset_index(drop=True)

# Create an empty list to store the entries and key presses
entries_and_presses = []
ids_already_seen = []

start_id = 0

print("Press 'y' if the sentence is an instruction, 'n' if it is not, or 'q' to save and exit.")
for i in range(len(df)):
    sent_id = df.iloc[i]['id']
    if sent_id not in ids_already_seen:
        sent = df.iloc[i]['sentence']
        pred = df.iloc[i]['speech_act']
        print(f"\n### ID {sent_id} ### {i} ###\n\n{sent}\n\n-- {pred} --\n\n(y / n)")
        key_press = getch()

        # Check if the key press is valid
        while key_press not in ['y', 'n', 'q']:
            print("Invalid input. Please try again. (y / n)\n")
            key_press = getch()

        # Add the entry and key press to the list
        if key_press == 'q':
            break
        else:
            if key_press == 'y':
                ids_already_seen.append(sent_id)
            entries_and_presses.append({'id': str(sent_id), 'entry': sent, 'key_press': key_press})

# Save the entries and key presses to a JSON file
with open('human_eval.json', 'w') as json_file:
    json.dump(entries_and_presses, json_file, indent=4)

print("Entries and key presses saved to human_eval.json")
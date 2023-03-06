import os
import requests

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), "data", "tiny_shakespeare.txt")

# download if not exists
if not os.path.exists(input_file_path):
    print("File does not exist, downloading...")
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

# read the file
with open(input_file_path, 'r') as f:
    text = f.read()

# print some information
print(f"Corpus length: {len(text)} \n")

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size} ({''.join(chars)})\n")

for i, line in enumerate(text.split('\n')[:50]):
    print(f"{i:02d}  {line}")
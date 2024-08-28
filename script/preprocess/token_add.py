import os
import random
from itertools import product
from transformers import BertTokenizer

# fix random seed
def set_seed(seed_value=42):
    random.seed(seed_value)
set_seed(42)


###################################################################################################


current_directory = os.getcwd()
vocab_path = './data/saved_weights/pretrained_dnabert/vocab.txt'


###################################################################################################


# create tokens
bases = 'ATGC'
tokens = []
for i in range(7):
    for combination in product(bases, repeat=6 - 1):
        token = ''.join(combination[:i] + ('-',) + combination[i:])
        tokens.append(token)

print(f"Generated {len(tokens)} tokens.")

# write down vocab file
with open(vocab_path, 'r') as file:
    existing_vocab = file.read().splitlines()

updated_vocab = existing_vocab + [token for token in tokens if token not in existing_vocab]

with open(vocab_path, 'w') as file:
    file.write('\n'.join(updated_vocab))

print("Updated vocabulary.")

tokenizer = BertTokenizer(vocab_file=vocab_path)

print("Tokenizer has been updated with new tokens.")
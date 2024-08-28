import os
import argparse
import random
import matplotlib.pyplot as plt
import json
from datasets import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fix random seed
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

plt.style.use('dark_background')


###################################################################################################


'''
tsv index : 'target DNA seq' 'sgRNA seq' 'Mismatch' 'Cleavege reads' 'label'
'''


###################################################################################################


parser = argparse.ArgumentParser(description='Extension, cross valication fold')
parser.add_argument('--extension', type=str, default='None', choices=['None', '100'])
parser.add_argument('--task', type=str, default='clf', choices=['clf', 'regr'])
args = parser.parse_args()
task = args.task
if args.extension in ['None']:
    extension = args.extension
else:
    extension = int(args.extension)

k_mer = 3



def loacte_mismatch(seq1, seq2):
    locate = ['-']*24
    for i, (base1, base2) in enumerate(zip(seq1, seq2)):
        if base1 != base2:
            locate[i] = '*'
    return '-'.join(locate)


###################################################################################################


# input information

sgrna = 'PDCD1_site_9'
dna_seq     = '-CACTGATCCAGGGCCTGACTGGG'
sgrna_seq   = '-GGGGGTTCCAGGGCCTGTCTGGG'
label = 0
read = 1622
fig_path = f'data/fig/{sgrna}_regression_6.png'
title = f'DNABERT Attention weights of {sgrna}\n{loacte_mismatch(dna_seq, sgrna_seq)}'


###################################################################################################


current_directory = os.getcwd()
sgRNA_json_path = f'data/datasets/origin/CHANGEseq_sgRNA_seqname.json'

with open(sgRNA_json_path, 'r') as file:
    sgRNAs_json = json.load(file)
sgRNAs_name = sgRNAs_json["sgRNAs_name"]

# separate datasets by 10
targets_fold_list = np.array_split(sgRNAs_name, 10)
targets_fold_list = [[str(target) for target in subarray] for subarray in targets_fold_list]

def train_test_fold(targets, targets_fold_list):
    test_list = targets
    train_list = [target for subarray in targets_fold_list for target in subarray if target not in test_list]
    return train_list, test_list

for i, sgrna_list in enumerate(targets_fold_list):
    if sgrna in sgrna_list:
        fold = i + 1

targets = targets_fold_list[fold-1]
train_list, test_list = train_test_fold(targets, targets_fold_list)

finetuned_offtarget_path = f'data/saved_weights/ot_ft/ot_ft_{str(extension)}_{str(task)}_{fold}'


###################################################################################################


def seq_to_kmer(sequence, kmer=k_mer):
    kmers = [sequence[i:i+kmer] for i in range(len(sequence)-kmer+1)]
    merged_sequence = ' '.join(kmers)
    return merged_sequence

def list_to_datasets_clf(dna_seq=dna_seq, sgrna_seq=sgrna_seq, label=label):
    datasets_dict = {'DNA' : [], 'sgRNA' : [], 'label' : []}
    dna_seq = seq_to_kmer(dna_seq)
    sgrna_seq = seq_to_kmer(sgrna_seq)
    datasets_dict['DNA'].append(dna_seq)
    datasets_dict['sgRNA'].append(sgrna_seq)
    datasets_dict['label'].append(label)
    datasets = Dataset.from_dict(datasets_dict)
    return datasets

def list_to_datasets_regr(dna_seq=dna_seq, sgrna_seq=sgrna_seq, read=read):
    datasets_dict = {'DNA' : [], 'sgRNA' : [], 'label' : []}
    dna_seq = seq_to_kmer(dna_seq)
    sgrna_seq = seq_to_kmer(sgrna_seq)
    read = np.log(read+1)
    datasets_dict['DNA'].append(dna_seq)
    datasets_dict['sgRNA'].append(sgrna_seq)
    datasets_dict['label'].append(read)
    datasets = Dataset.from_dict(datasets_dict)
    return datasets

###################################################################################################


def visualize_clf(datasets, tokens, model, fig_path, title):
    
    # convert Dataset to torch type
    datasets = TensorDataset(
        torch.tensor(datasets['input_ids']),
        torch.tensor(datasets['attention_mask']),
        torch.tensor(datasets['token_type_ids']),
        torch.tensor(datasets['label'])
    )
    
    data_loader = DataLoader(datasets, batch_size=32)
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logit = outputs.logits
            attentions = outputs.attentions # touple 12 len
            
    logit = logit.cpu()
    prediction = np.argmax(torch.softmax(logit, dim=1).cpu().numpy(), axis=1)[0]
    print(f'!!!!!!!!!! TRUE LABEL = {label} : PRED LABEL = {prediction} !!!!!!!!!!')
    
    # Visualize DNABERT Attention weights
    fig, ax1 = plt.subplots(figsize=(30, 15))
    ax2 = ax1.twinx()
    
    ax1.set_yticks(np.arange(len(tokens)))
    ax2.set_yticks(np.arange(len(tokens)))
    ax1.set_ylim(-1, len(tokens))
    ax2.set_ylim(-1, len(tokens))
    ax1.set_yticklabels(tokens[::-1])
    ax2.set_yticklabels(tokens[::-1])
    ax1.set_xticks(np.arange(12))
    ax1.set_xlim(-0.1, 12.1)
    
    # Draw lines connecting tokens
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            for layer in range(len(attentions)):
                attention = attentions[layer][0]
                plt.plot([layer, layer+0.9], [i, j], color='green', alpha=attention.mean(0)[i, j].item())
    plt.title(title, fontsize=30)
    plt.grid(False)
    # plt.show()
    fig.savefig(fig_path)


def visualize_regr(datasets, tokens, model, fig_path, title):
    
    # convert Dataset to torch type
    datasets = TensorDataset(
        torch.tensor(datasets['input_ids']),
        torch.tensor(datasets['attention_mask']),
        torch.tensor(datasets['token_type_ids']),
        torch.tensor(datasets['label'])
    )
    
    data_loader = DataLoader(datasets, batch_size=32)
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, token_type_ids, reads = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logit = outputs.logits
            attentions = outputs.attentions # touple 12 len
            
    prediction = logit.squeeze().cpu().numpy()
    print(f'!!!!!!!!!! TRUE READ COUNT = {reads[0]} : PRED READ COUNT = {prediction} !!!!!!!!!!')
    
    # Visualize DNABERT Attention weights
    fig, ax1 = plt.subplots(figsize=(40, 20))
    ax2 = ax1.twinx()
    
    ax1.set_yticks(np.arange(len(tokens)))
    ax2.set_yticks(np.arange(len(tokens)))
    ax1.set_ylim(-1, len(tokens))
    ax2.set_ylim(-1, len(tokens))
    ax1.set_yticklabels(tokens[::-1])
    ax2.set_yticklabels(tokens[::-1])
    ax1.set_xticks(np.arange(13))
    ax1.set_xlim(-0.1, 12)
    
    # Draw lines connecting tokens
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            for layer in range(len(attentions)):
                attention = attentions[layer][0]
                plt.plot([layer, layer+0.9], [i, j], color='green', alpha=attention.mean(0)[i, j].item())
    plt.title(title, fontsize=40)
    plt.grid(False)
    # plt.show()
    fig.savefig(fig_path)



###################################################################################################


def clf_main():
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetuned_offtarget_path)
    
    # definition func for tokenizer
    if extension == 'None':
        max_length = 2*(24 - k_mer + 1) + 3
    elif extension == 100:
        max_length = (24 - k_mer + 1) + (100 - k_mer + 1) + 3
    def tokenize_function(examples):
        return tokenizer(examples['DNA'], examples['sgRNA'], padding='max_length', truncation=True, max_length=max_length)
    
    # load fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_offtarget_path,
        num_labels=2,
        output_hidden_states=False,
        output_attentions=True,
        ignore_mismatched_sizes=True
    ).to(device)
    model.to(device)
    model.eval()
    
    datasets = list_to_datasets_clf(dna_seq, sgrna_seq, label)
    datasets = datasets.map(tokenize_function, batched=True)
    
    tokens = tokenizer.convert_ids_to_tokens(datasets['input_ids'][0])
    
    visualize_clf(datasets, tokens, model, fig_path, title)
        

def regr_main():
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetuned_offtarget_path)
    
    # definition func for tokenizer
    if extension == 'None':
        max_length = 2*(24 - k_mer + 1) + 3
    elif extension == 100:
        max_length = (24 - k_mer + 1) + (100 - k_mer + 1) + 3
    def tokenize_function(examples):
        return tokenizer(examples['DNA'], examples['sgRNA'], padding='max_length', truncation=True, max_length=max_length)
    
    # load fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_offtarget_path,
        num_labels=1,
        output_hidden_states=False,
        output_attentions=True,
        ignore_mismatched_sizes=True
    ).to(device)
    model.to(device)
    model.eval()
    
    datasets = list_to_datasets_regr(dna_seq, sgrna_seq, read)
    datasets = datasets.map(tokenize_function, batched=True)
    
    tokens = tokenizer.convert_ids_to_tokens(datasets['input_ids'][0])
    
    visualize_regr(datasets, tokens, model, fig_path, title)



if __name__ == '__main__':
    if task == 'clf':
        clf_main()
    elif task == 'regr':
        regr_main()
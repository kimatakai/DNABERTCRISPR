import os
import argparse
import csv
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
parser.add_argument('--fold', type=int, default=1, choices=range(1, 11))
args = parser.parse_args()
task = args.task
fold = args.fold
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
title = f'DNABERT Attention weights of {sgrna}'


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

targets = targets_fold_list[fold-1]
train_list, test_list = train_test_fold(targets, targets_fold_list)

finetuned_offtarget_path = f'data/saved_weights/dnabert/ot_ft_{str(task)}_{fold}'


###################################################################################################


def seq_to_kmer(sequence, kmer=k_mer):
    kmers = [sequence[i:i+kmer] for i in range(len(sequence)-kmer+1)]
    merged_sequence = ' '.join(kmers)
    return merged_sequence

def list_to_datasets_clf(data):
    datasets_dict = {'DNA' : [], 'sgRNA' : [], 'label' : []}
    for row in data:
        dna_seq = seq_to_kmer(row[3])
        sgrna_seq = seq_to_kmer(row[4])
        label = int(row[8])
        datasets_dict['DNA'].append(dna_seq)
        datasets_dict['sgRNA'].append(sgrna_seq)
        datasets_dict['label'].append(label)
    datasets = Dataset.from_dict(datasets_dict)
    return datasets

def list_to_datasets_regr(data):
    datasets_dict = {'DNA' : [], 'sgRNA' : [], 'label' : []}
    for row in data:
        dna_seq = seq_to_kmer(row[3])
        sgrna_seq = seq_to_kmer(row[4])
        reads = int(row[7])
        datasets_dict['DNA'].append(dna_seq)
        datasets_dict['sgRNA'].append(sgrna_seq)
        datasets_dict['label'].append(reads)
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
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(40, 20))
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
                plt.plot([layer, layer+0.9], [i, j], color='green', alpha=attention.mean(0)[i, j].item(), lw=3)
    plt.title(title, fontsize=30)
    plt.grid(False)
    # plt.show()
    fig.savefig(fig_path)


def visualize_regr(datasets, tokens, model, save_path, title):
    
    # convert Dataset to torch type
    datasets = TensorDataset(
        torch.tensor(datasets['input_ids']),
        torch.tensor(datasets['attention_mask']),
        torch.tensor(datasets['token_type_ids']),
        torch.tensor(datasets['label'])
    )
    
    data_loader = DataLoader(datasets, batch_size=32)
    
    attention_weights = {i : [] for i in range(12)}
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, token_type_ids, reads = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            attentions = outputs.attentions # touple 12 len
            for i, attention in enumerate(attentions):
                attention = torch.mean(attention, dim=0)
                attention_weights[i].append(attention)
    
    for i in range(len(attention_weights)):
        attention_weights_i = torch.cat(attention_weights[i], dim=0)
        attention_weights_i = attention_weights_i.cpu().numpy()
        attention_weights_i = np.mean(attention_weights_i, axis=0)
        attention_weights[i] = attention_weights_i
    
    print(attention_weights[0].shape)
    
    # Visualize DNABERT Attention weights
    plt.style.use('dark_background')
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
            for layer in range(len(attention_weights)):
                attention = attention_weights[layer]
                plt.plot([layer, layer+0.9], [i, j], color='green', alpha=attention[i, j].item(), lw=3)
    plt.title(title, fontsize=40)
    plt.grid(False)
    # plt.show()
    fig.savefig(save_path)
    
    return attention_weights


def visualize_diff_weights(weights1, weights2, tokens, save_path, title):
    
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(22.5, 11.25))
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
            for layer in range(len(weights1)):
                attention1 = weights1[layer]
                attention2 = weights2[layer]
                diff_attention = attention1 - attention2
                color = 'red' if diff_attention[i, j] > 0 else 'blue'
                alpha = abs(diff_attention[i, j].item())*3
                plt.plot([layer, layer+0.9], [i, j], color=color, alpha=alpha, lw=3)
                
    # Set title with Arial font and partially italicize it
    title_font = {'fontname': 'Arial', 'fontsize': 40, 'fontweight': 'normal'}
    plt.title(title, **title_font)
    
    plt.grid(False)
    # plt.show()
    fig.savefig(save_path, format='tiff')
    



###################################################################################################


# open tsv file
def read_tsv(sgrna_name):
    with open(f'data/datasets/tsvdata/{sgrna_name}.tsv', 'r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        data = [row for row in reader]
    return data

# data dict
data_dict = {}
for sgrna_name in test_list:
    if not sgrna_name in data_dict:
        data_dict[sgrna_name] = read_tsv(sgrna_name)


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
    
    for sgrna_name in test_list:
        list_data = data_dict[sgrna_name]
        # split datasets
        list_data_pos = [row for row in list_data if int(row[8])==1]
        list_data_neg = [row for row in list_data if int(row[8])==0]
        
        # convert datasets object and tokenize
        datasets_pos = list_to_datasets_regr(list_data_pos)
        datasets_neg = list_to_datasets_regr(list_data_neg)
        datasets_pos = datasets_pos.map(tokenize_function, batched=True)
        datasets_neg = datasets_neg.map(tokenize_function, batched=True)
        
        # index tokens
        tokens = ['[CLS]'] + [f'DNA_{i+1}' for i in range(len(sgrna_seq)-k_mer+1)] + ['[SEP]'] + [f'sgRNA_{i+1}' for i in range(len(sgrna_seq)-k_mer+1)] + ['[SEP]']
        
        # positive datasets visualize
        save_path = f'data/fig/{sgrna_name}_regression_positive.png'
        title = f'Attention weights of {sgrna_name.replace("_", " ")} on label 1'
    
        attention_weights_pos = visualize_regr(datasets_pos, tokens, model, save_path, title)
        
        # negative datasets visualize
        save_path = f'data/fig/{sgrna_name}_regression_negative.png'
        title = f'Attention weights of {sgrna_name.replace("_", " ")} on label 0'
    
        attention_weights_neg = visualize_regr(datasets_neg, tokens, model, save_path, title)

        # difference visualize
        save_path = f'data/fig/{sgrna_name}_regression_difference.tiff'
        title = f'Difference attention weights of $\it{{{sgrna_name.replace("_", " ")}}}$'
        visualize_diff_weights(attention_weights_pos, attention_weights_neg, tokens, save_path, title)


if __name__ == '__main__':
    if task == 'clf':
        clf_main()
    elif task == 'regr':
        regr_main()
import os
import argparse
import random
import csv
import json
from datasets import Dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
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


###################################################################################################


'''
tsv index : 'target DNA seq' 'sgRNA seq' 'Mismatch' 'Cleavege reads' 'label'
'''


###################################################################################################


parser = argparse.ArgumentParser(description='Extension, cross valication fold')
parser.add_argument('--extension', type=str, default='None', choices=['None', '100'])
parser.add_argument('--fold', type=int, default=1, choices=range(1, 11))
args = parser.parse_args()
fold = args.fold
if args.extension in ['None']:
    extension = args.extension
else:
    extension = int(args.extension)

k_mer = 3


###################################################################################################


current_directory = os.getcwd()
tsvdir_path = f'./data/datasets/tsvdata/'
sgRNA_json_path = f'data/datasets/origin/CHANGEseq_sgRNA_seqname.json'
pretrained_model_path = f'data/saved_weights/pretrained_dnabert{str(k_mer)}'
finetuned_mismatch_path = f'data/saved_weights/pair_ft/pair_ft_{str(extension)}_{str(fold)}'

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


###################################################################################################


def load_tsv(name):
    datapath = tsvdir_path + f'{name}.tsv'
    with open(datapath, 'r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        data = [row for row in reader]
    return data

def seq_to_kmer(sequence, kmer=k_mer):
    kmers = [sequence[i:i+kmer] for i in range(len(sequence)-kmer+1)]
    merged_sequence = ' '.join(kmers)
    return merged_sequence

def list_to_datasets(data):
    random.shuffle(data)
    datasets_dict = {'targetDNA' : [], 'sgRNA' : [], 'label' : []}
    for row in data:
        if extension == 'None':
            targetdna = seq_to_kmer(row[3])
            targetdna_ = row[3]
        elif extension == 100:
            targetdna = seq_to_kmer(row[9])
            targetdna_ = row[9]
        sgrna = seq_to_kmer(row[4])
        sgrna_ = row[4]
        datasets_dict['targetDNA'].append(targetdna)
        datasets_dict['sgRNA'].append(sgrna)
        mismatch = int(row[6])
        datasets_dict['label'].append(return_mismatch_positions(targetdna_, sgrna_))
    datasets = Dataset.from_dict(datasets_dict)
    return datasets

def return_mismatch_positions(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("The sequences must be of the same length.")
    
    mismatch_list = []
    for base1, base2 in zip(seq1, seq2):
        if base1 == '-' and base2 == '-':
            mismatch_list.append(0)
        elif base1 == '-' or base2 == '-':
            mismatch_list.append(1)
        elif base1 == 'N' or base2 == 'N':
            mismatch_list.append(0)
        else:
            mismatch_list.append(0 if base1 == base2 else 1)
    
    return mismatch_list


###################################################################################################


def evalate_func(datasets, model, device='cuda'):
    true_labels = datasets['label']
    
    tensor_dataset = TensorDataset(
        torch.tensor(datasets['input_ids']),
        torch.tensor(datasets['attention_mask']),
        torch.tensor(true_labels, dtype=torch.float)
    )
    data_loader = DataLoader(tensor_dataset, batch_size=32)
    
    model.to(device)
    model.eval()
    
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu()
            all_logits.append(logits)
            all_labels.append(labels.cpu())
    
    # Logitsを確率に変換し、最終的な予測を行う
    all_labels = torch.cat(all_labels, dim=0)
    print(all_labels.shape)
    all_logits = torch.cat(all_logits, dim=0)
    print(all_logits.shape)
    probabilities = torch.sigmoid(all_logits)
    print(probabilities[0])
    predictions = (probabilities > 0.5).int()
    print(predictions[0])
    print(predictions.shape)
    
    # 真のラベルと予測ラベルの評価
    accuracy = accuracy_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions, average='macro')
    precision = precision_score(all_labels, predictions, average='macro')
    f1 = f1_score(all_labels, predictions, average='macro')
    
    return {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1}


###################################################################################################


def main():
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetuned_mismatch_path)
    
    # definition func for tokenizer
    if extension == 'None':
        # max_length = 48 - k_mer + 2
        max_length = 2*(24 - k_mer + 1) + 3
    elif extension == 100:
        max_length = (24 - k_mer + 1) + (100 - k_mer + 1) + 3
    def tokenize_function(examples):
        return tokenizer(examples['targetDNA'], examples['sgRNA'], padding='max_length', truncation=True, max_length=max_length)
    
    # load fine-tuned model
    model = (AutoModelForSequenceClassification.from_pretrained(finetuned_mismatch_path, num_labels=24, problem_type="multi_label_classification").to(device))
    model.to(device)
    model.eval()
    
    remove_list = []
    
    # test
    for name in test_list:
        if name not in remove_list:
            print(f'name : {name}')

            # load tsv data
            data = load_tsv(name)

            # processing datasets
            datasets = list_to_datasets(data)
            datasets = datasets.map(tokenize_function, batched=True)

            results = evalate_func(datasets, model)
            for metric in results:
                print(f'{metric} : {results[metric]}')


if __name__ == '__main__':
    main()
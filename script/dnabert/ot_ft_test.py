import os
import argparse
import random
import csv
import json
from datasets import Dataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, accuracy_score, recall_score, precision_score, f1_score, mean_squared_error, r2_score
from scipy.stats import pearsonr
import statsmodels.api as sm
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
parser.add_argument('--task', type=str, default='clf', choices=['clf', 'regr'])
parser.add_argument('--fold', type=int, default=1, choices=range(1, 11))
args = parser.parse_args()
task = args.task
fold = args.fold

k_mer = 3


###################################################################################################


current_directory = os.getcwd()
tsvdir_path = f'./data/datasets/tsvdata/'
sgRNA_json_path = f'data/datasets/origin/CHANGEseq_sgRNA_seqname.json'
pretrained_model_path = f'data/saved_weights/pretrained_dnabert3'
finetuned_offtarget_path = f'data/saved_weights/dnabert/ot_ft_{str(task)}_{fold}'
result_save_dir_path = f'data/result/dnabert/'
result_save_path = result_save_dir_path + f'result_dnabert_{task}_{fold}.json'
result_mean_save_path = result_save_dir_path + f'result_mean_dnabert_{task}_{fold}.json'


os.makedirs(result_save_dir_path, exist_ok=True)

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

# list -> dataset object
def list_to_datasets(data):
    random.shuffle(data)
    datasets_dict = {'targetDNA' : [], 'sgRNA' : [], 'mismatch' : [], 'reads' : [], 'label' : []}
    for row in data:
        targetdna = seq_to_kmer(row[3])
        sgrna = seq_to_kmer(row[4])
        datasets_dict['targetDNA'].append(targetdna)
        datasets_dict['sgRNA'].append(sgrna)
        datasets_dict['mismatch'].append(int(row[6]))
        datasets_dict['reads'].append(float(row[7]))
        datasets_dict['label'].append(int(row[8]))
    datasets = Dataset.from_dict(datasets_dict)
    return datasets

def list_to_datasets_regr(data):
    random.shuffle(data)
    datasets_dict = {'targetDNA' : [], 'sgRNA' : [], 'mismatch' : [], 'label' : [], 'cleavage' : []}
    for row in data:
        targetdna = seq_to_kmer(row[3])
        sgrna = seq_to_kmer(row[4])
        datasets_dict['targetDNA'].append(targetdna)
        datasets_dict['sgRNA'].append(sgrna)
        datasets_dict['mismatch'].append(int(row[6]))
        datasets_dict['label'].append(np.log2(float(row[7])+1))
        datasets_dict['cleavage'].append(int(row[8]))
    datasets = Dataset.from_dict(datasets_dict)
    return datasets


###################################################################################################


def clf_evaluate_func(datasets, model):
    
    # extract true label
    true_label = datasets['label']
    true_label_np = torch.IntTensor(true_label).cpu().numpy()
    
    # convert Dataset to torch type
    datasets = TensorDataset(
        torch.tensor(datasets['input_ids']),
        torch.tensor(datasets['attention_mask']),
        torch.tensor(datasets['token_type_ids']),
        torch.tensor(datasets['label'])
    )
    data_loader = DataLoader(datasets, batch_size=32)
    all_logits = []
    
    # prediction
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            state = outputs.hidden_states[-1][:, 0, :]
            logits = outputs.logits
            all_logits.append(logits)
            # all_embedding.append(state)
    all_logits = torch.cat(all_logits, dim=0)
    # all_embedding = torch.cat(all_embedding, dim=0)
    # print(f'All embeddings shape : {all_embedding.shape}')
    
    # logits -> prob
    probabilities = torch.softmax(all_logits, dim=1)
    probabilities_np = probabilities[:, 1].cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    predictions = np.argmax(probabilities, axis=1)
    print(f'true label [0]  : {true_label_np[0]}')
    print(f'pred label [0]  : {predictions[0]}')
    print(f'prob [0]        : {probabilities[0]}')
    print(f'true label [1]  : {true_label_np[1]}')
    print(f'pred label [1]  : {predictions[1]}')
    print(f'prob [1]        : {probabilities[1]}')
    print(f'true label [1000]  : {true_label_np[1000]}')
    print(f'pred label [1000] : {predictions[1000]}')
    print(f'prob [1000]     : {probabilities[1000]}')
    print(true_label_np[0:100])
    print(predictions[0:100])
    
    # print(all_embedding[0])
    
    # metrics
    accuracy = accuracy_score(true_label_np, predictions)
    recall = recall_score(true_label_np, predictions)
    precision = precision_score(true_label_np, predictions)
    f1 = f1_score(true_label_np, predictions)
    auc_score = roc_auc_score(true_label_np, probabilities_np)
    p, r, _ = precision_recall_curve(true_label_np, probabilities_np)
    prauc_score = auc(r, p)
    mcc = matthews_corrcoef(true_label_np, predictions)
    
    return {'accuracy':accuracy, 'recall':recall, 'precision':precision, 'f1':f1, 'ROC-AUC':auc_score, 'PR-AUC':prauc_score, 'mcc':mcc}


def return_r2(true_reads, predictions):
    X = sm.add_constant(predictions)
    model = sm.OLS(true_reads, X).fit()
    r2 = model.rsquared
    return r2

def regr_evaluate_func(datasets, model):
    
    # extract true label
    true_reads = datasets['label']
    true_reads_np = torch.FloatTensor(true_reads).cpu().numpy()
    
    # convert Dataset to torch type
    datasets = TensorDataset(
        torch.tensor(datasets['input_ids']),
        torch.tensor(datasets['attention_mask']),
        torch.tensor(datasets['token_type_ids']),
        torch.tensor(datasets['label'])
    )
    data_loader = DataLoader(datasets, batch_size=32)
    all_preds = []
    
    # prediction
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, token_type_ids, reads = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            # state = outputs.hidden_states[-1][:, 0, :]
            preds = outputs.logits.squeeze().cpu().numpy()
            
            # all_labels.append(labels)
            # all_embedding.append(state)
            if preds.ndim == 0:
                preds = np.expand_dims(preds, axis=0)
            if preds.ndim > 0:
                all_preds.append(preds)
    all_preds = np.concatenate(all_preds, axis=0)
    print(all_preds.shape)
    print(true_reads_np.shape)
    
    print(f'true label [0]  : {true_reads_np[0]}')
    print(f'pred label [0]  : {all_preds[0]}')
    print(f'true label [1]  : {true_reads_np[1]}')
    print(f'pred label [1]  : {all_preds[1]}')
    print(f'true label [1000]  : {true_reads_np[1000]}')
    print(f'pred label [1000] : {all_preds[1000]}')
    
    # metrics
    mse = mean_squared_error(true_reads_np, all_preds)
    pearson_corr, _ = pearsonr(true_reads_np, all_preds)
    r2_sk = r2_score(true_reads_np, all_preds)
    r2_statis = return_r2(true_reads_np, all_preds)
    
    return {'MSE': mse, 'Pearson Correlation': pearson_corr, 'R^2 Scikit': r2_sk, 'R^2 Statis' : r2_statis}



###################################################################################################


def main():
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetuned_offtarget_path)
    
    # definition func for tokenizer
    max_length = 2*(24 - k_mer + 1) + 3
    def tokenize_function(examples):
        return tokenizer(examples['targetDNA'], examples['sgRNA'], padding='max_length', truncation=True, max_length=max_length)
    
    # load fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_offtarget_path,
        num_labels=2,
        output_hidden_states=True,
        ignore_mismatched_sizes=True
    ).to(device)
    model.to(device)
    model.eval()
    
    results_dict = {'accuracy':[], 'recall':[], 'precision':[], 'f1':[], 'ROC-AUC':[], 'PR-AUC':[], 'mcc':[]}
    results_mean_dict = {}
    
    # test
    for name in test_list:
        print(f'name : {name}')

        # load tsv data
        data = load_tsv(name)

        # processing datasets
        datasets = list_to_datasets(data)
        datasets = datasets.map(tokenize_function, batched=True)
            
        num_negative = len([i for i in datasets['label'] if i == 0])
        num_positive = len([i for i in datasets['label'] if i == 1])
        label_weights = torch.tensor([1, num_negative/num_positive], dtype=torch.float32)
        print(num_negative, num_positive, label_weights)
            
        print(datasets['input_ids'][0])

        results = clf_evaluate_func(datasets, model)
        for metric in results:
            print(f'{metric} : {results[metric]}')
            results_dict[metric].append(results[metric])
    
    for metric in results:
        results_mean_dict[metric] = sum(results_dict[metric])/len(results_dict[metric])
    
    with open(result_save_path, 'w') as file:
        json.dump(results_dict, file, ensure_ascii=False, indent=4)
    
    with open(result_mean_save_path, 'w') as file:
        json.dump(results_mean_dict, file, ensure_ascii=False, indent=4)
        

def main_regr():
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetuned_offtarget_path)
    
    # definition func for tokenizer
    max_length = 2*(24 - k_mer + 1) + 3
    def tokenize_function(examples):
        return tokenizer(examples['targetDNA'], examples['sgRNA'], padding='max_length', truncation=True, max_length=max_length)
    
    # load fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_offtarget_path,
        num_labels=1,
        output_hidden_states=True,
        ignore_mismatched_sizes=True
    ).to(device)
    model.to(device)
    model.eval()
    
    results_dict = {'MSE': [], 'Pearson Correlation': [], 'R^2 Scikit': [], 'R^2 Statis': []}
    results_mean_dict = {}
    
    # test
    for name in test_list:
        print(f'name : {name}')

        # load tsv data
        data = load_tsv(name)

        # processing datasets
        datasets = list_to_datasets_regr(data)
        datasets = datasets.map(tokenize_function, batched=True)
            
        print(datasets['input_ids'][0])

        results = regr_evaluate_func(datasets, model)
        for metric in results:
            print(f'{metric} : {results[metric]}')
            results_dict[metric].append(float(results[metric]))
    
    for metric in results:
        results_mean_dict[metric] = float(sum(results_dict[metric])/len(results_dict[metric]))
    
    with open(result_save_path, 'w') as file:
        json.dump(results_dict, file, ensure_ascii=False, indent=4)
    
    with open(result_mean_save_path, 'w') as file:
        json.dump(results_mean_dict, file, ensure_ascii=False, indent=4)


###################################################################################################

if __name__ == '__main__':
    if task == 'clf':
        main()
    elif task == 'regr':
        main_regr()
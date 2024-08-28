import os
import argparse
import random
import csv
import json
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

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
finetuned_mismatch_path = f'data/saved_weights/pair_ft/pair_ft'
finetuned_offtarget_path = f'data/saved_weights/dnabert/ot_ft_{str(task)}_{fold}'

os.makedirs(finetuned_offtarget_path, exist_ok=True)

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


# setting hyper parameter
epochs = 3
if task == 'clf':
    batch_size = 64
elif task == 'regr':
    batch_size = 256
learning_rate = 2e-5
logging_steps = 1000



# compute metrics for classification
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

class CustomTrainer(Trainer):
    def __init__(self, *args, weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.weight is not None:
            # Ensure weights are on the same device as logits
            weight = self.weight.to(logits.device)
            if self.args.fp16:
                weight = weight.half()
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# compute metrics for regression
def compute_metrics_regr(pred):
    labels = pred.label_ids
    preds = pred.predictions[0]
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    return {"mse": mse, "mae": mae}

class CustomTrainerRegr(Trainer):
    def __init__(self, *args, weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = nn.MSELoss()  # Use MSELoss for regression
        if self.args.fp16:
            logits = logits.half()
            labels = labels.half()
        loss = loss_fct(logits.view(-1), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


###################################################################################################


def main_clf():
    
    print(f'Off-Target Prediction Fine-Tuning task {task} fold {fold}')
    
    # load datasets
    data = []
    for name in train_list:
        data += load_tsv(name)
    
    # processing datasets
    datasets = list_to_datasets(data)

    print(datasets['targetDNA'][0])
    print(datasets['sgRNA'][0])
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetuned_mismatch_path)
    
    # definition func for tokenizer
    max_length = 2*(24 - k_mer + 1) + 3
    def tokenize_function(examples):
        return tokenizer(examples['targetDNA'], examples['sgRNA'], padding='max_length', truncation=True, max_length=max_length)
    datasets = datasets.map(tokenize_function, batched=True)
    
    print(f'targetDNA [0]       : {datasets["targetDNA"][0]}')
    print(f'sgRNA [0]           : {datasets["sgRNA"][0]}')
    print(f'input_ids [0]       : {datasets["input_ids"][0]}')
    print(f'token_type_ids [0]  : {datasets["token_type_ids"][0]}')
    
    # training arguments
    training_args = TrainingArguments(
        output_dir=finetuned_offtarget_path,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        # gradient_accumulation_steps=1,
        lr_scheduler_type="cosine_with_restarts",
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=False,
        log_level="error",
        save_strategy="no",  # This disables checkpoint saving
    )
    
    # load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_mismatch_path,
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)
    
    num_negative = len([i for i in datasets['label'] if i == 0])
    num_positive = len([i for i in datasets['label'] if i == 1])
    label_weights = torch.tensor([1, num_negative/num_positive], dtype=torch.float32)
    print(num_negative, num_positive, label_weights)
    
    trainer = CustomTrainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        compute_metrics = compute_metrics,
        train_dataset = datasets,
        weight = label_weights
    )

    # fine tuning
    trainer.train()
    
    # save fine-tuned model
    tokenizer.save_pretrained(finetuned_offtarget_path)
    model.save_pretrained(finetuned_offtarget_path)
    print(f'Complete saving offtarget prediction task fine-tuned model for fold {fold}')


def main_regr():
    
    print(f'Off-Target Prediction Fine-Tuning task {task} fold {fold}')
    
    # load datasets
    data = []
    for name in train_list:
        data += load_tsv(name)
    
    # processing datasets
    datasets = list_to_datasets_regr(data)

    print(datasets['targetDNA'][0])
    print(datasets['sgRNA'][0])
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetuned_mismatch_path)
    
    # definition func for tokenizer
    max_length = 2*(24 - k_mer + 1) + 3
    def tokenize_function(examples):
        return tokenizer(examples['targetDNA'], examples['sgRNA'], padding='max_length', truncation=True, max_length=max_length)
    datasets = datasets.map(tokenize_function, batched=True)
    
    print(f'targetDNA [0]       : {datasets["targetDNA"][0]}')
    print(f'sgRNA [0]           : {datasets["sgRNA"][0]}')
    print(f'input_ids [0]       : {datasets["input_ids"][0]}')
    print(f'token_type_ids [0]  : {datasets["token_type_ids"][0]}')
    
    # training arguments
    training_args = TrainingArguments(
        output_dir=finetuned_offtarget_path,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        # gradient_accumulation_steps=1,
        lr_scheduler_type="cosine_with_restarts",
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=False,
        log_level="error",
        save_strategy="no",  # This disables checkpoint saving
        # fp16=True,
        # fp16_opt_level="O1"
    )
    
    # load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_mismatch_path,
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(device)
    
    trainer = CustomTrainerRegr(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        compute_metrics = compute_metrics_regr,
        train_dataset = datasets
    )

    # fine tuning
    trainer.train()
    
    # save fine-tuned model
    tokenizer.save_pretrained(finetuned_offtarget_path)
    model.save_pretrained(finetuned_offtarget_path)
    print(f'Complete saving offtarget prediction task fine-tuned model for fold {fold}')


###################################################################################################

if __name__ == '__main__':
    if task == 'clf':
        main_clf()
    elif task == 'regr':
        main_regr()

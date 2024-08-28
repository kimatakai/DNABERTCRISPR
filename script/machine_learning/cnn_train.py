
import os
import argparse
import json
import numpy as np

import cnn_module


# fix random seed
def set_seed(seed_value=42):
    np.random.seed(seed_value)
set_seed(42)


###################################################################################################


'''
name chr start dna sgrna strand mismatch reads label expanded-dna
'''


###################################################################################################


parser = argparse.ArgumentParser(description='Extension, cross valication fold')
parser.add_argument('--task', type=str, default='clf', choices=['clf', 'regr'])
parser.add_argument('--fold', type=int, default=1, choices=range(1, 11))
args = parser.parse_args()
task = args.task
fold = args.fold
print(f'!!!!!!!!!! task : {task}, fold : {fold} !!!!!!!!!!')


###################################################################################################


current_directory = os.getcwd()
tsvdir_path = f'data/datasets/tsvdata/'
sgRNA_json_path = f'data/datasets/origin/CHANGEseq_sgRNA_seqname.json'
model_save_dir_path = f'data/saved_weights/cnn'
model_save_path = model_save_dir_path + f'/cnn_{task}_{fold}.weights.h5'


os.makedirs(model_save_dir_path, exist_ok=True)

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


def train_clf(encodings, labels):
    model_class = cnn_module.clf_class(inputlen=len(encodings[0]))
    model_class.train(encodings, labels, model_save_path)

def train_regr(encodings, labels, reads):
    model_class = cnn_module.regr_class(inputlen=len(encodings[0]))
    model_class.train(encodings, labels, reads, model_save_path)


###################################################################################################



def main():
    # cnn training
    listdata = cnn_module.return_traindata(train_list)
    encodings = cnn_module.return_encoding(listdata)
    labels = cnn_module.return_labels(listdata)
    reads = cnn_module.return_reads(listdata)
    
    if task == 'clf':
        train_clf(encodings, labels)
    elif task == 'regr':
        train_regr(encodings, labels, reads)
    

if __name__ == '__main__':
    main()
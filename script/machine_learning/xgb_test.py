
import os
import argparse
import json
import numpy as np

import xgb_module



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
model_save_dir_path = f'data/saved_weights/xgb'
model_save_path = model_save_dir_path + f'/xgb_{task}_{fold}.pkl'
result_save_dir_path = f'data/result/xgb/'
result_save_path = result_save_dir_path + f'result_xgb_{task}_{fold}.json'
result_mean_save_path = result_save_dir_path + f'result_mean_xgb_{task}_{fold}.json'


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


def main():
    results_dict = {}
    results_mean_dict = {}
    for name in test_list:
        print(f'!!!!!!!!!! {name} !!!!!!!!!!')
        listdata = xgb_module.load_tsv(name)
        encodings = xgb_module.return_encoding(listdata)
        labels = xgb_module.return_labels(listdata)
        reads = xgb_module.return_reads(listdata)
        
        if task == 'clf':
            results = xgb_module.test_clf(encodings, labels, model_save_path)
            for metrics in results:
                if metrics not in results_dict:
                    results_dict[metrics] = []
                results_dict[metrics].append(results[metrics])
        
        if task == 'regr':
            results = xgb_module.test_regr(encodings, reads, model_save_path)
            for metrics in results:
                if metrics not in results_dict:
                    results_dict[metrics] = []
                results_dict[metrics].append(float(results[metrics]))
        
        for result in results:
            print(f'{result} : {results[result]}')
    
    for metrics in results_dict:
        if metrics != 'name':
            sample_sam = sum(results_dict[metrics])
            sample_len = len(results_dict[metrics])
            results_mean_dict[metrics] = float(sample_sam/sample_len)
    
    with open(result_save_path, 'w') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    with open(result_mean_save_path, 'w') as f:
        json.dump(results_mean_dict, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    main()
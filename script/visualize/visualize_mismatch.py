import os
import csv
import json
import numpy as np


###################################################################################################


'''
tsv index : 'target DNA seq' 'sgRNA seq' 'Mismatch' 'Cleavege reads' 'label'
'''



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

fold = 7

train_list, test_list = train_test_fold(targets_fold_list[fold-1], targets_fold_list)




###################################################################################################


# mismatch locate count
def loacte_mismatch(seq1, seq2):
    locate = ['-']*24
    for i, (base1, base2) in enumerate(zip(seq1, seq2)):
        if base1 != base2:
            locate[i] = '*'
    # return '-'.join(locate)
    return locate

def whether_same_mismatch(locate, seq1, seq2):
    locate_ = ['-']*24
    for i, (base1, base2) in enumerate(zip(seq1, seq2)):
        if locate[i] == '-':
            if base1 != base2:
                return 'ng'
            else:
                locate_[i] = '-'
        elif locate[i] == '*':
            if base1 == base2:
                return 'ng'
            else:
                locate_[i] = '*'
        else:
            continue
    if locate_ == locate:
        return 'ok'
    else:
        return 'ng'
    
def return_max_sample(data):
    max_reads = 0
    max_index = 0
    for i, row in enumerate(data):
        reads = int(row[7])
        if reads > max_reads:
            max_index = i
            max_reads = reads
        else:
            pass
    return max_index


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


# most reads sample
for sgrna_name in test_list:
    data = data_dict[sgrna_name]
    max_index = return_max_sample(data)
    # print(data[max_index])


# concatenate each data
all_data = []
for sgrna_name in test_list:
    data = data_dict[sgrna_name]
    for row in data:
        all_data.append(row)
        if int(row[6])==0:
            print(row)

# sample data
sample_seq1 = '-GAGACCCTGCTCAAGGGCCGAGG'
sample_seq2 = '-GGTTTCACCGAGACCTCAGTAGG'
sample_mm_locate = loacte_mismatch(sample_seq1, sample_seq2)

# check same mismatch
same_mm_index = []
for i, row in enumerate(all_data):
    seq1 = row[3]
    seq2 = row[4]
    mm_flag = whether_same_mismatch(sample_mm_locate, seq1, seq2)
    if mm_flag == 'ok':
        same_mm_index.append(i)

print(all_data[0])

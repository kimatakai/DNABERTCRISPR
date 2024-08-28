import os
import csv
import json
import numpy as np


###############################################################################


'''
tsv index : 'name', 'chrom', 'chromStart', 'strand', 'target_sequence', 'offtarget_sequence', 'distance', 'CHANGEseq_reads', 'label'
'''


###############################################################################


current_directory = os.getcwd()
all_tsv_path = 'data/datasets/origin/dataset.tsv'
sgRNA_json_path = 'data/datasets/origin/CHANGEseq_sgRNA_seqname.json'


###############################################################################


class Extension_class:
    def __init__(self, sgrna_name):
        self.dirpath = f'./data/datasets/tsvdata'
        self.datapath = self.dirpath + f'/{sgrna_name}.tsv'
    
    def recreate_tsv(self, each_data):
        processed_data = []
        for row in each_data:
            processed_data.append(row)
        self.write_tsv(processed_data)
    
    def write_tsv(self, data):
        os.makedirs(self.dirpath, exist_ok=True)
        with open(self.datapath, 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerows(data)
    
    
    
###############################################################################


def main():
    # read all tsv data
    with open(all_tsv_path, 'r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        all_tsv = [row for row in reader]
    
    # read json data
    with open(sgRNA_json_path, 'r') as file:
        sgRNAs_json = json.load(file)
    sgRNAs_name = sgRNAs_json["sgRNAs_name"]
    
    # each sgRNA
    for name in sgRNAs_name:
        each_data = [row for row in all_tsv if row[0]==name]
        Extension_cls = Extension_class(name)
        Extension_cls.recreate_tsv(each_data)



if __name__ == '__main__':
    main()

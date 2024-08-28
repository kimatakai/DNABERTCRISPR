
import os
import csv
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import statsmodels.api as sm
import joblib



###################################################################################################


'''
name chr start dna sgrna strand mismatch reads label expanded-dna
'''


###################################################################################################


current_directory = os.getcwd()
tsvdir_path = f'./data/datasets/tsvdata/'


###################################################################################################


def load_tsv(name):
    filename = tsvdir_path + f'{name}.tsv'
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        data = [row for row in reader]
    return data

def return_traindata(name_list):
    train_data = []
    for name in name_list:
        data = load_tsv(name)
        train_data += data
    return train_data

base_index = {'A':0, 'T':1, 'G':2, 'C':3}
def create_onehot(seq1, seq2, n_mismatch):
    seqlen = len(seq1)
    onehot = np.zeros((seqlen*4*4), dtype=np.int8)
    for i, (base1, base2) in enumerate(zip(seq1, seq2)):
        if base1 == '-' or base2 == '-' or base1 == 'N' or base2 == 'N':
            pass
        else:
            onehot[4*4*i+4*base_index[base1]+base_index[base2]] = 1
    onehot = np.append(onehot, n_mismatch)
    return onehot

def return_encoding(listdata):
    encodings = []
    for row in listdata:
        onehot = create_onehot(row[3], row[4], int(row[6]))
        encodings.append(onehot)
    return np.array(encodings, dtype=np.int8)

def return_labels(listdata):
    labels = []
    for row in listdata:
        labels.append(row[8])
    return np.array(labels, dtype=np.int8)

def return_reads(listdata):
    reads = []
    for row in listdata:
        reads.append(row[7])
    return np.log2(np.array(reads, np.float32)+1)

def prob_to_labels(predicted_probs):
    labels = []
    for prob in predicted_probs:
        if prob >= 0.5:
            labels.append(int(1))
        else:
            labels.append(int(0))
    return np.array(labels)

def return_r2_stats(reads_true, reads_pred):
    X = sm.add_constant(reads_pred)
    model = sm.OLS(reads_true, X).fit()
    r2 = model.rsquared
    return r2

def return_r2_sklearn(reads_true, reads_pred):
    r2 = r2_score(reads_true, reads_pred)
    return r2


###################################################################################################


def train_clf(encodings, labels, save_path):
    print(f'Start training XGB for classification task')
    # split data
    encodings_train, encodings_eval, labels_train, labels_eval = train_test_split(
        encodings,
        labels,
        test_size=0.1,
        stratify=labels,
        random_state=42,
    )
    # XGBoost parametor
    params = {
        'max_depth': 10,
        'eta': 0.1,
        'n_estimators': 1000,
        'verbosity': 1,
        'seed': 42,
        'n_thread': 12,
    }
    # labels weights
    weight_positive = np.sum(labels_train == 0) / len(labels_train)
    weight_negative = np.sum(labels_train == 1) / len(labels_train)
    sample_weights = np.where(labels_train == 0, weight_negative, weight_positive)
    
    # model train
    model = xgb.XGBClassifier(**params)
    model.fit(
        encodings_train, 
        labels_train, 
        sample_weight = sample_weights,
        eval_metric="logloss", 
        early_stopping_rounds=10,
        eval_set=[(encodings_eval, labels_eval)], 
        verbose=True)
    
    print('!!!!!!!!!! save model !!!!!!!!!!')
    joblib.dump(model, save_path)


def test_clf(encodings, labels, save_path):
    model = joblib.load(save_path)
    
    true_labels = labels
    probabilities = np.array(model.predict_proba(encodings))
    probabilities = probabilities[:, 1]
    predictions = model.predict(encodings)
    
    # metrics
    accuracy = accuracy_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    auc_score = roc_auc_score(true_labels, probabilities)
    p, r, _ = precision_recall_curve(true_labels, probabilities)
    prauc_score = auc(r, p)
    mcc = matthews_corrcoef(true_labels, predictions)
    
    return {'accuracy':accuracy, 'recall':recall, 'precision':precision, 'f1':f1, 'ROC-AUC':auc_score, 'PR-AUC':prauc_score, 'mcc':mcc}


###################################################################################################


def train_regr(encodings, labels, reads, save_path):
    print(f'Start training XGB for regression task')
    
    # split data
    encodings_train, encodings_eval, labels_train, labels_eval, reads_train, reads_eval = train_test_split(
        encodings,
        labels,
        reads,
        test_size=0.1,
        random_state=42,
    )
    
    # XGBoost parameters for regression
    params = {
        'max_depth': 10,
        'eta': 0.1,
        'n_estimators': 1000,
        'verbosity': 1,
        'seed': 42,
        'n_thread': 12,
        # 'objective': 'reg:squarederror'  # Use squared error for regression
    }
    
    # model train
    model = xgb.XGBRegressor(**params)
    model.fit(
        encodings_train, 
        reads_train, 
        eval_set=[(encodings_eval, reads_eval)], 
        verbose=True
    )
    
    print('!!!!!!!!!! save model !!!!!!!!!!')
    joblib.dump(model, save_path)


def test_regr(encodings, reads, save_path):
    model = joblib.load(save_path)
    
    reads_true = reads
    reads_pred = model.predict(encodings)
    
    # metrics
    r2_scikit = return_r2_sklearn(reads_true, reads_pred)
    r2_stats = return_r2_stats(reads_true, reads_pred)
    mse = mean_squared_error(reads_true, reads_pred)
    pearson_corr, _ = pearsonr(reads_true, reads_pred)
            
    return {'R^2 Scikit': r2_scikit, 'R^2 Statis' : r2_stats, 'MSE': mse, 'Pearson Correlation': pearson_corr}

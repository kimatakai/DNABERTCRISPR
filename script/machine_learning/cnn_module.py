
import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import statsmodels.api as sm


# fix random seed
def set_seed(seed_value=42):
    np.random.seed(seed_value)
set_seed(42)


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

def return_r2_stats(true_reads, reads_pred):
    X = sm.add_constant(reads_pred)
    model = sm.OLS(true_reads, X).fit()
    r2 = model.rsquared
    return r2

def return_r2_sklearn(reads_true, reads_pred):
    r2 = r2_score(reads_true, reads_pred)
    return r2




###################################################################################################


# setting hyper parameter
epochs = 500
# setting callback information
reduce_learning = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, verbose=1, mode='auto', \
    min_delta=0.02, cooldown=0, min_lr=0) 
eary_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02, patience=9, verbose=1, mode='auto')
callbacks = [reduce_learning, eary_stopping]


class clf_class:
    def __init__(self, inputlen):
        self.inputlen = inputlen
        self.device = self.get_device()
        self.model = self.build_model()
        self.batch_size = 64
    
    def get_device(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            return "/GPU:0"
        else:
            return "/CPU:0"
        
    def build_model(self):
        with tf.device(self.device):
            # model definition
            input_layer = Input(shape=(self.inputlen, 1))  # CNN expects a 3D input
            conv1       = Conv1D(16, kernel_size=3, activation='relu', kernel_regularizer=l2(1e-5))(input_layer)
            conv1       = BatchNormalization()(conv1)
            conv1       = MaxPooling1D(pool_size=2)(conv1)
            conv1       = Dropout(0.5)(conv1)
            conv2       = Conv1D(32, kernel_size=3, activation='relu', kernel_regularizer=l2(1e-5))(conv1)
            conv2       = BatchNormalization()(conv2)
            conv2       = MaxPooling1D(pool_size=2)(conv2)
            conv2       = Dropout(0.5)(conv2)
            conv3       = Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(1e-5))(conv2)
            conv3       = BatchNormalization()(conv3)
            conv3       = MaxPooling1D(pool_size=2)(conv3)
            conv3       = Dropout(0.5)(conv3)
            flat        = Flatten()(conv3)
            dense       = Dense(64, activation='relu', kernel_regularizer=l2(1e-5))(flat)
            dense       = BatchNormalization()(dense)
            dense       = Dropout(0.5)(dense)
            output      = Dense(1, activation="sigmoid")(dense)
            model       = Model(inputs=[input_layer], outputs=[output])
            # compile and train the model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

    def train(self, input, labels, save_path):
        print(f'Start training CNN for classification')
        with tf.device(self.device):
            print(self.device)
            print(input.shape)
            input_train, input_validation, labels_train, labels_validation = \
                train_test_split(input, labels, test_size=0.1, stratify=labels, random_state=42)
            weight_positive = np.sum(labels_train == 0) / len(labels_train)
            weight_negative = np.sum(labels_train == 1) / len(labels_train)
            sample_weights = np.where(labels_train == 0, weight_negative, weight_positive)
            model = self.build_model()
            model.summary()
            model.fit(input_train, labels_train, sample_weight=sample_weights, batch_size=self.batch_size, epochs=epochs, verbose=1, \
                validation_data=(input_validation, labels_validation), callbacks=callbacks)
            model.save_weights(save_path)
    
    def test(self, input, true_labels, save_path):
        print(f'Test CNN for classification')
        with tf.device(self.device):
            model = self.build_model()
            model.load_weights(save_path)
            labels_prob = np.array(model.predict(input))
            labels_pred = prob_to_labels(labels_prob)
            
            # metrics
            accuracy = accuracy_score(true_labels, labels_pred)
            recall = recall_score(true_labels, labels_pred)
            precision = precision_score(true_labels, labels_pred)
            f1 = f1_score(true_labels, labels_pred)
            auc_score = roc_auc_score(true_labels, labels_pred)
            p, r, _ = precision_recall_curve(true_labels, labels_pred)
            prauc_score = auc(r, p)
            mcc = matthews_corrcoef(true_labels, labels_pred)

            return {'accuracy':accuracy, 'recall':recall, 'precision':precision, 'f1':f1, 'ROC-AUC':auc_score, 'PR-AUC':prauc_score, 'mcc':mcc}


class regr_class:
    def __init__(self, inputlen):
        self.inputlen = inputlen
        self.device = self.get_device()
        self.model = self.build_model()
        self.batch_size = 256
    
    def get_device(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            return "/GPU:0"
        else:
            return "/CPU:0"
        
    def build_model(self):
        with tf.device(self.device):
            # model definition
            input_layer = Input(shape=(self.inputlen, 1))  # CNN expects a 3D input
            conv1       = Conv1D(32, kernel_size=3, activation='relu', kernel_regularizer=l2(1e-5))(input_layer)
            conv1       = BatchNormalization()(conv1)
            conv1       = MaxPooling1D(pool_size=2)(conv1)
            conv1       = Dropout(0.5)(conv1)
            conv2       = Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(1e-5))(conv1)
            conv2       = BatchNormalization()(conv2)
            conv2       = MaxPooling1D(pool_size=2)(conv2)
            conv2       = Dropout(0.5)(conv2)
            conv3       = Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(1e-5))(conv2)
            conv3       = BatchNormalization()(conv3)
            conv3       = MaxPooling1D(pool_size=2)(conv3)
            conv3       = Dropout(0.5)(conv3)
            flat        = Flatten()(conv3)
            dense       = Dense(64, activation='relu', kernel_regularizer=l2(1e-5))(flat)
            dense       = BatchNormalization()(dense)
            dense       = Dropout(0.5)(dense)
            output      = Dense(1, activation="linear", name="output")(dense)
            model       = Model(inputs=[input_layer], outputs=[output])
            # compile and train the model
            model.compile(loss='mean_squared_error', optimizer='adam')
            return model

    def train(self, input, labels, reads, save_path):
        print(f'Start training CNN for regression')
        with tf.device(self.device):
            input_train, input_validation, labels_train, labels_validation, reads_train, reads_validation = \
                train_test_split(input, labels, reads, test_size=0.1, stratify=labels, random_state=42)
            model = self.build_model()
            model.summary()
            model.fit(input_train, reads_train, batch_size=self.batch_size, epochs=epochs, verbose=1, \
                validation_data=(input_validation, reads_validation), callbacks=callbacks)
            model.save_weights(save_path)
    
    def test(self, input, reads_true, save_path):
        print(f'Test CNN for regression')
        with tf.device(self.device):
            model = self.build_model()
            model.load_weights(save_path)
            reads_pred = np.array(model.predict(input), np.float32).flatten()
            
            # metrics
            r2_scikit = return_r2_sklearn(reads_true, reads_pred)
            r2_stats = return_r2_stats(reads_true, reads_pred)
            mse = mean_squared_error(reads_true, reads_pred)
            pearson_corr, _ = pearsonr(reads_true, reads_pred)
            
            return {'R^2 Scikit': r2_scikit, 'R^2 Statis' : r2_stats, 'MSE': mse, 'Pearson Correlation': pearson_corr}



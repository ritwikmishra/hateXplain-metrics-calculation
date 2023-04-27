
# CUDA_VISIBLE_DEVICES=0 python temp.py --split 1 --bias_in_fc True --add_cls_sep_tokens False --epochs 10 --encoder_frozen False --encoder_name bert-base-cased --data_path data/ --checkpoint_path . --message "" --dummy False --run_ID 27 --drop_out 0.40 --bert_lr 5e-7 --ft_lr 1e-6 --max_len 300
"""# Header file"""
import os 
import math

import string
import re, pickle

import psutil
import time, datetime, pytz
# !pip install emoji
import emoji, transformers
from tqdm import tqdm

import pandas as pd
import numpy as np
import json
import torch, sys
import torch.nn as nn
from torch.utils.data import Dataset, random_split, TensorDataset, DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, AdamW, RobertaTokenizer, RobertaModel

from transformers import XLMRobertaTokenizer, XLMRobertaModel


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
# import networkx as nx
# from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from collections import Counter

IST = pytz.timezone('Asia/Kolkata') # your time zone here

"""### Cache location

# Variables
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on {}".format(device))




#argparser
import argparse

parser = argparse.ArgumentParser(description='Hatespeech model training')
parser.add_argument('--split', type=int, help='data split, 1/2/3')
parser.add_argument('--bias_in_fc', type=str, help='bias in fc, True/False', default=True)
parser.add_argument('--add_cls_sep_tokens', type=str, help='add_cls_sep_tokens or not, True/False', default=False)
parser.add_argument('--epochs', type=int, help='no. of epochs to train the model', default=10)
parser.add_argument('--encoder_frozen', type=str, help='encoder layers frozen or not', default=False)
parser.add_argument('--encoder_name', type=str, help='name of encoder, bert-base-cased/xlm-roberta-base')
parser.add_argument('--data_path', type=str, help='data files path')
parser.add_argument('--checkpoint_path', type=str, help='path where checkpoints to be saved')
parser.add_argument('--message', type=str, help='notes if done something diff from prev experiment ', default='NO MESSAGE')
parser.add_argument('--dummy', type=str, help='is this a dummy experiment or not', default=False)
parser.add_argument('--drop_out', type=float, default=0.4)
parser.add_argument('--bert_lr', type=float, default=5e-07)
parser.add_argument('--ft_lr', type=float, default=1e-6)
parser.add_argument('--keep_k', type=int, default=0)
parser.add_argument('--max_len', type=int, default=100)

parser.add_argument('--run_ID', type=int, help='experiment run ID')


args = parser.parse_args()


# randon seed for model or randon function
SEED = 63571
import os, random
os.environ['PYTHONHASHSEED'] = str(SEED)
# Torch RNG
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# Python RNG
np.random.seed(SEED)
random.seed(SEED)

# language
language = "en"

# Parameters for dataset
dataset_parameter = {
                     'header' : 0,
                     'batch_size' : 8,
                     'max_seq_len' : args.max_len,
                     'dataset_split' : [0.7, 0.15, 0.15],
                     'sentence_column' :   'text',         #'commentText',
                     'label_column' : 'label',            #'label',
                     'num_classes':2
                    }


# transformer model parameters



bert_model_parameter = {'name' : 'XLM-Roberta',
                   'hugging_face_name' : args.encoder_name, #'bert-base-cased', #'xlm-roberta-base'
                   'tokenizer' : args.encoder_name,
                   'second_last_layer_size' : 768,
                   'config' : None,
                   'sub_word_starting_char': '▁',
                   'tokenizer_cls_id': None,
                   'tokenizer_sep_id': None,
                   'tokenizer_pad_id': None,
                   'token_ids_way': 3,
                   'word_level_mean_way' : 5,
                   'freeze_pre_trained_model_weight' : True
                   }

model_parameter = bert_model_parameter

# Optimizer, loss function and other training paramters
training_parameter = {
                 
                      'hidden_layers'     : [128, 32],
                      'activation_function'   : 'leaky_relu',  #choose one out of ['relu', 'prelu', 'elu', 'leaky_relu']
                      'optimizer_parameter' : {'name' : 'AdamW',
                                               'lr' : 1e-5},
                      'loss_func_parameter' : {'name' : 'NLLLoss',
                                               'class_weight' : [1, 1]},
                      'epoch' : 10,
                     }

tokenizer = AutoTokenizer.from_pretrained(bert_model_parameter['tokenizer'], return_tensors="pt",max_length=100, padding='max_length')
bert = AutoModel.from_pretrained(bert_model_parameter['hugging_face_name'])

bert_model_parameter['tokenizer_cls_id'], _, bert_model_parameter['tokenizer_sep_id'], bert_model_parameter['tokenizer_pad_id'] = tokenizer("i", return_tensors="pt", max_length=4, padding='max_length')['input_ids'][0].tolist()


import stanza
import pandas as pd
from transformers import AutoTokenizer


def tokenize_my_sent(text):
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    tokenized_text_i = []
    doc = nlp(text)
    for sents in doc.sentences:
        for word in sents.words:

            if len(tokenizer.tokenize(word.text)) > 0:
                tokenized_text_i.append(word.text)
    return ' '.join(tokenized_text_i)


nlp = stanza.Pipeline('en',processors='tokenize', download_method=2)  # put desired language here
add_cls_sep_token_ids = False
if args.add_cls_sep_tokens=='True':
    add_cls_sep_token_ids = True


def tokenize_word_ritwik(text):
    input_ids, attention_mask, word_break, word_break_len, tag_one_hot, tag_attention = [],[],[],[], [] , []

    # assert len(text) == len(tags)
    # onehot_encoding = pd.get_dummies(dataset_parameter['tag_set'])
    for text_i in text:
        input_ids_i, attention_mask_i, word_break_i, word_break_len_i, tag1hot_i, tag_att_i = [model_parameter['tokenizer_cls_id']] if add_cls_sep_token_ids else [], [], [], 0, [], [0] if add_cls_sep_token_ids else []

        # tokenized_text_i = []
        # doc = nlp(text_i)
        # for sents in doc.sentences:
        # 	for word in sents.words:
        # 		tokenized_text_i.append(word.text)
        tokenized_text_i = text_i.split() # since humne pehle filtering kardi hai, yahan pe dobara stanza se pass karne ki zarurat ni hai

        for i, word in enumerate(tokenized_text_i):
            input_ids_per_word = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
     
            if len(input_ids_per_word) > 1:
                if model_parameter['token_ids_way']==1:
                    input_ids_per_word = [input_ids_per_word[0]]
                    word_break_i.append(1)
                elif model_parameter['token_ids_way']==2:
                    input_ids_per_word = [input_ids_per_word[-1]]
                    word_break_i.append(1)
                elif model_parameter['token_ids_way']==3:
                    word_break_i.append(len(input_ids_per_word))
                else:
                    raise "Error ajeeb"
            else:
                word_break_i.append(1)
            input_ids_i += input_ids_per_word

            # tag1hot_i.append(onehot_encoding[tag].tolist())
            # tag_att_i.append(1)

        # -------------- truncation start --------------

        input_ids_i = input_ids_i[:dataset_parameter['max_seq_len']+ (1 if add_cls_sep_token_ids else 0)] # notice I added 1 to input ids only because baaki saari lists empty thi initially whereas input ids wali list empty nhi thi i.e. it contained cls token id
        word_break_i = word_break_i[:dataset_parameter['max_seq_len']]
        word_break_len_i = len(word_break_i) # number of words
        # --------------  truncation end  --------------
        # print(input_ids_i)
        # input()
        if add_cls_sep_token_ids:
            input_ids_i.append(model_parameter['tokenizer_sep_id'])
        attention_mask_i += [1]*len(input_ids_i)

        # # -------------- padding start --------------
        input_ids_i = input_ids_i + [model_parameter['tokenizer_pad_id']]*(dataset_parameter['max_seq_len']+(2 if add_cls_sep_token_ids else 0) - len(input_ids_i))
        attention_mask_i = attention_mask_i + [0]*(dataset_parameter['max_seq_len']+(2 if add_cls_sep_token_ids else 0) - len(attention_mask_i))
        word_break_i = word_break_i + [0]*(dataset_parameter['max_seq_len'] - len(word_break_i))
    
        # # --------------  padding end  --------------

        input_ids.append(input_ids_i)
        attention_mask.append(attention_mask_i)
        word_break.append(word_break_i)
        word_break_len.append(word_break_len_i)

 
    # print('attention_mask.shape: ', attention_mask.shape)

 
    return input_ids, attention_mask, word_break, word_break_len





bias_in_fc = True
if args.bias_in_fc=='False':
    bias_in_fc = False



class MODEL(nn.Module):
    def __init__(self, model):
        """
        Initialize model and define last layers of fine-tuned model
        bert: BertModel from Huggingface
        return: None
        """
        super(MODEL, self).__init__()
        self.bert_model = model
        self.device = 'cuda:0'
        
        # word level with or without demo weights initialization
        if bert_model_parameter['word_level_mean_way'] == 1:
            self.dense = nn.Linear(768, training_parameter['hidden_layers'][0], bias=False)

        if model_parameter['word_level_mean_way'] == 2:
            self.register_parameter(name='model_adjacency_weight',
                                     param=torch.nn.Parameter(torch.rand(model_parameter['second_last_layer_size'],
                                                                       requires_grad=True)))
        elif model_parameter['word_level_mean_way'] == 3:
            self.register_parameter(name='word_level_emb_horizontal_weights',
                                     param=torch.nn.Parameter(torch.rand(dataset_parameter['max_seq_len'],
                                                                       requires_grad=True)))
        elif model_parameter['word_level_mean_way'] == 4:
            self.register_parameter(name='word_level_emb_batch_weights',
                                     param=torch.nn.Parameter(torch.rand(dataset_parameter['max_seq_len'], model_parameter['second_last_layer_size'],
                                                                       requires_grad=True)))
        elif model_parameter['word_level_mean_way'] == 5:
            self.flat_dense = nn.Linear(dataset_parameter['max_seq_len']*model_parameter['second_last_layer_size'], model_parameter['second_last_layer_size'])
            
        self.activation = training_parameter["activation_function"]
        self.relu1 = nn.ReLU()

        self.fc1 = nn.Linear(model_parameter['second_last_layer_size'], 128)
        self.dropout1 = nn.Dropout(0.1)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

        if args.encoder_frozen == 'True':
            for param in self.bert_model.parameters():
                param.requires_grad = False
    
    # def get_input_embeddings(self):
    #     return self.bert_model.embeddings.word_embeddings
    #     return self.dummy_fun
    
    # def dummy_fun(self,input_ids):
    #     input_ids = torch.tensor(input_ids)
    #     if len(input_ids.shape) != 2:
    #         input_ids = input_ids.unsqueeze(0)
    #     text = [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(x)) for x in input_ids]
    #     input_ids, attention_mask, word_break, word_break_len = tokenize_word_ritwik(text)
    #     input_ids, attention_mask, word_break, word_break_len = torch.tensor(input_ids), torch.tensor(attention_mask),  torch.tensor(word_break), torch.tensor(word_break_len)
    #     input_ids, attention_mask, word_level, word_level_len = input_ids.to(device), attention_mask.to(device),  word_break.to(device), word_break_len.to(device)
    #     token_level_embedding = self.bert_model.embeddings.word_embeddings(input_ids)
    #     return id

    def forward(self, input_ids, attention_mask, token_type_ids=None, output_hidden_states=None):
        """
        Pass input data from all layer of model

        input_id                  :   (list) Encoding vectors (INT) from BertTokenizerFast
        attention_mask            :   (list) Mask vector (INT [0,1]) from Bert BertTokenizerFast
        average_adjacency_matrix  :   (list of list) Average adj matrix (float (0:1)) defined with stanza dependancy graph and its degree multiplication
        word_level                :   (list) Contain number of sub words broken from parent word (INT)
        word_level_len            :   (INT) Define the length of parent sentence without any tokenization

        return: (float [0,1]) Last output from fine tuned model
        """
        # if type(input_ids[0])!=list and ():
        #     input_ids = [input_ids]
        input_ids = torch.tensor(input_ids)
        if len(input_ids.shape) != 2:
            input_ids = input_ids.unsqueeze(0)
        
        text = [tokenize_my_sent(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(x))) for x in input_ids]
        input_ids, attention_mask, word_break, word_break_len = tokenize_word_ritwik(text)
        input_ids, attention_mask, word_break, word_break_len = torch.tensor(input_ids), torch.tensor(attention_mask),  torch.tensor(word_break), torch.tensor(word_break_len)
        input_ids, attention_mask, word_level, word_level_len = input_ids.to(device), attention_mask.to(device),  word_break.to(device), word_break_len.to(device)

        model_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        token_level_embedding = model_output.last_hidden_state

        # word level embeddings initialized
        word_level_embedding = torch.zeros(input_ids.shape[0], dataset_parameter['max_seq_len'], bert_model_parameter['second_last_layer_size'])
        # print(word_level_embedding.shape)
        # input('wait herenn')

        # iterate all text in one batch
        for batch_no in range(0, input_ids.shape[0]):
            start, end = 0, 0
            for word_break_counter in range(0, word_level_len[batch_no]):
                start = end
                end = start + word_level[batch_no][word_break_counter]
                word_level_embedding[batch_no][word_break_counter] = torch.mean(token_level_embedding[batch_no][start:end], 0, True)
        word_level_embedding = word_level_embedding.to(device)
        if args.keep_k:
            word_level_embedding[:,args.keep_k:,:] = torch.tensor(0.0).to(word_level_embedding.device) # embedding masking. We are keeping the embeddings for first k (=4) words, rest are set to 0.
            # word_level_embedding[:,0:1,:] = torch.tensor(0.0).to(word_level_embedding.device)

        if bert_model_parameter['word_level_mean_way']==1:
            word_level_embedding_flat = torch.flatten(word_level_embedding, start_dim=1)
            output = torch.mean(word_level_embedding, 1)
        elif bert_model_parameter['word_level_mean_way']==2:
            word_level_embedding_flat = torch.flatten(word_level_embedding, start_dim=1)
            word_level_embedding_mean = word_level_embedding * self.word_level_emb_vertical_weights
            output = torch.mean(word_level_embedding_mean, 1)
        elif bert_model_parameter['word_level_mean_way']==3:
            word_level_embedding_flat = torch.flatten(word_level_embedding, start_dim=1)
            word_level_embedding_mean =  word_level_embedding.permute(0,2,1) * self.word_level_emb_horizontal_weights
            output = torch.mean(word_level_embedding_mean.permute(0,2,1), 1)
        elif bert_model_parameter['word_level_mean_way']==4:
            word_level_embedding_flat = torch.flatten(word_level_embedding, start_dim=1)
            word_level_embedding_mean = word_level_embedding * self.word_level_emb_batch_weights
            output = torch.mean(word_level_embedding_mean, 1)
        elif bert_model_parameter['word_level_mean_way']==5:
            word_level_embedding_flat = torch.flatten(word_level_embedding, start_dim=1)
            output = model.flat_dense(word_level_embedding_flat)
        first_fc_layer_emb = output #its size will be 768 in any mean-type way

        x = self.relu1(output)
        x = self.fc1(output)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = self.fc2(x)    
        x = self.log_softmax(x)
        # return x, first_fc_layer_emb, word_level_embedding_flat
        return transformers.modeling_outputs.SequenceClassifierOutput({'logits':x})
        # return output, dense_layer_emb,  word_level_embedding_flat #also returning word_level_embedding to use it in calculating relevance at word-level





    def predict(self, text):
        # processed_text = preprocess_text(text)

        text = tokenize_my_sent(text)
  


        # input_ids, attention_mask, word_break, word_break_len = tokenize_word_ritwik([processed_text])
        input_ids, attention_mask, word_break, word_break_len = tokenize_word_ritwik2([text])

        input_ids, attention_mask, word_break, word_break_len = torch.tensor(input_ids), torch.tensor(attention_mask),  torch.tensor(word_break), torch.tensor(word_break_len)


      #put all feature variable on device
        input_ids, attention_mask, word_break, word_break_len = input_ids.to(device), attention_mask.to(device),  word_break.to(device), word_break_len.to(device)

      #get prediction from model
        with torch.no_grad():
            logits, first_fc_layer_emb, word_level_embedding_flat = self.forward(input_ids, attention_mask, word_break, word_break_len)


        label_proba = np.exp(logits.detach().cpu().numpy()[0])
        return label_proba, first_fc_layer_emb, word_level_embedding_flat

model = MODEL(bert)
device = 'cuda:0'
model = model.to(device)

#saving checkpoints
checkpoint_file = 'runID-'+str(args.run_ID)+'-'+'checkpoint.pth'
# check if there is a checkpoint to load from
start_epoch = 1
if os.path.exists(args.checkpoint_path+'/trained_models/'+checkpoint_file):

    checkpoint = torch.load(args.checkpoint_path+'/trained_models/'+checkpoint_file)
    print(model.load_state_dict(checkpoint['model_state_dict']))

from ferret import Benchmark
from ferret.explainers.lime import LIMEExplainer
from ferret.explainers.shap import SHAPExplainer

bench = Benchmark(model, tokenizer, explainers=[SHAPExplainer(model, tokenizer),LIMEExplainer(model, tokenizer)])
explanations = bench.explain("I will kill Hitler", target=1)
print(bench.get_dataframe(explanations))



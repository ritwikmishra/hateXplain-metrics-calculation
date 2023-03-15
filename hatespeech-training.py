
"""# Header file"""
import os 
import math

import string
import re

import psutil
import time
# !pip install emoji
import emoji
from tqdm import tqdm

import pandas as pd
import numpy as np
import json
import torch
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


"""### Cache location

# Variables
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on {}".format(device))



#argparser
import argparse

parser = argparse.ArgumentParser(description='Hatespeech model training')
parser.add_argument('--split', type=int, help='data split, 1/2/3')
parser.add_argument('--bias_in_fc', type=bool, help='bias in fc, True/False', default=True)
parser.add_argument('--add_cls_sep_tokens', type=bool, help='add_cls_sep_tokens or not, True/False', default=False)
parser.add_argument('--epochs', type=int, help='no. of epochs to train the model', default=10)
parser.add_argument('--encoder_frozen', type=bool, help='encoder layers frozen or not', default=False)
parser.add_argument('--encoder_name', type=str, help='name of encoder, bert-base-cased/xlm-roberta-base', default='xlm-roberta-base')
parser.add_argument('--data_path', type=str, help='data files path')
parser.add_argument('--checkpoint_path', type=str, help='path where checkpoints to be saved')
parser.add_argument('--run_ID', type=int, help='experiment run ID')


args = parser.parse_args()



# randon seed for model or randon function
SEED = 63571
np.random.seed(SEED)

# language
language = "en"

# Parameters for dataset
dataset_parameter = {
                     'header' : 0,
                     'batch_size' : 8,
                     'max_seq_len' : 100,
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

"""# Data Preprocessor"""

def remove_punctuation(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)

def remove_extra_space(text):
    return re.sub(' +', ' ', text)

def remove_username(text):
    return re.sub('@[\w]+','',text)

def remove_mentions(text):
    return re.sub('#[\w]+','',text)

def remove_url(text):
    return re.sub(('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'), '', text)

def remove_rt(text):
    return re.sub('RT :','',text)

def remove_newline(text):
    return re.sub('(\r|\n|\t)','',text)

# !pip install demoji
import demoji

def remove_emoji(text):
    return demoji.replace(text, repl="")

def remove_emoji2(text):
    regrex_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U0001F1F2-\U0001F1F4"  # Macau flag
                                u"\U0001F1E6-\U0001F1FF"  # flags
                                u"\U0001F600-\U0001F64F"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U0001F1F2"
                                u"\U0001F1F4"
                                u"\U0001F620"
                                u"\u200c"
                                u"\u200d"
                                u"\u2640-\u2642"
                                "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'',text)



def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_dots_from_shortforms(text):
    text = text.split('.') #t.v --> tv
    return ("").join(text)

def remove_special_char(text):
    return re.sub('\W+',' ', text)

def preprocess_text(text):
    # text = remove_emoji(text) # not including remove_emoji() in preprocessing as ROBERTa can handle emojis too..   
    text = remove_username(text)
    text = remove_emoji(text)
    text = remove_emoji2(text)
    text = remove_html_tags(text).strip()
    text = remove_rt(text)
    text = remove_url(text)
    text = remove_dots_from_shortforms(text)
    text = remove_special_char(text)
    text = text.lower()
    text = remove_mentions(text) 
    text = remove_newline(text)
    text = remove_extra_space(text)
    return text

"""# Model and tokenizer import"""


tokenizer = AutoTokenizer.from_pretrained(bert_model_parameter['tokenizer'])
bert = AutoModel.from_pretrained(bert_model_parameter['hugging_face_name'])

bert_model_parameter['tokenizer_cls_id'], _, bert_model_parameter['tokenizer_sep_id'], bert_model_parameter['tokenizer_pad_id'] = tokenizer("i", return_tensors="pt", max_length=4, padding='max_length')['input_ids'][0].tolist()

def tokenizer_word_length(text):
    """
        This function will calculate length of input ids from tokenizer
            
        Input ids are the token ids which are calculated from tokenizers.
        Here we are calculate ids of each word and then appending it in the final ids list, 
        in the mean if user want to include only first or last or all ids then this can be done with varible model_parameter['token_ids_way']
                    
        text: (list of list) list of text, first dimension is sample size and second is string or text
            
        return: (1D list) length of input ids
    """
    tokenized_len = []
    
    for row in text:
        space_seq_text = row.split(" ")
        text_input_ids = [bert_model_parameter['tokenizer_cls_id']]        
        for word in space_seq_text:
            word_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            text_input_ids += word_token_ids
        text_input_ids += [bert_model_parameter['tokenizer_sep_id']]
        tokenized_len.append(len(text_input_ids))
        
    return tokenized_len


import stanza
import pandas as pd
from transformers import AutoTokenizer

nlp = stanza.Pipeline('en',processors='tokenize', download_method=2)  # put desired language here
add_cls_sep_token_ids = args.add_cls_sep_tokens


def filter_for_max_len(sent_list, label_list):
	sents2, labels2 = [], []
	for s, l in tqdm(zip(sent_list, label_list), total=len(sent_list), ncols=150, desc='filtering'):
		tokenized_text_i = []
		doc = nlp(s)
		for sents in doc.sentences:
			for word in sents.words:
				tokenized_text_i.append(word.text)
		s = ' '.join(tokenized_text_i)
		if len(tokenizer(s,return_tensors="pt")['input_ids'][0]) <= dataset_parameter['max_seq_len']:
			sents2.append(s)
			labels2.append(l)
	return sents2, labels2

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



#this is for model.predict() fn only!
def tokenize_word_ritwik2(text):
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
		#input_ids_i = input_ids_i[:dataset_parameter['max_seq_len']+ (1 if add_cls_sep_token_ids else 0)] # notice I added 1 to input ids only because baaki saari lists empty thi initially whereas input ids wali list empty nhi thi i.e. it contained cls token id
		word_break_i = word_break_i[:dataset_parameter['max_seq_len']]
		word_break_len_i = len(word_break_i) # number of words
		# --------------  truncation end  --------------
		# print(input_ids_i)
		# input()
		if add_cls_sep_token_ids:
			input_ids_i.append(model_parameter['tokenizer_sep_id'])
		attention_mask_i += [1]*len(input_ids_i)

		# # -------------- padding start --------------
		# input_ids_i = input_ids_i + [model_parameter['tokenizer_pad_id']]*(dataset_parameter['max_seq_len']+(2 if add_cls_sep_token_ids else 0) - len(input_ids_i))
		# attention_mask_i = attention_mask_i + [0]*(dataset_parameter['max_seq_len']+(2 if add_cls_sep_token_ids else 0) - len(attention_mask_i))
		# word_break_i = word_break_i + [0]*(dataset_parameter['max_seq_len'] - len(word_break_i))
	
		# # --------------  padding end  --------------

		input_ids.append(input_ids_i)
		attention_mask.append(attention_mask_i)
		word_break.append(word_break_i)
		word_break_len.append(word_break_len_i)

	return input_ids, attention_mask, word_break, word_break_len



"""# Dataloader"""

class Dataset_loader(Dataset):
    def __init__(self, data_file_path):
        """
            Load 
                - preprocessed data from "Raw data" and "Stanza library" section and                 
                - adjacency matrix made from dependency relation from stanza
            Process
                - tokenize the dataset into encoding, mask and labels
                - word break list and its length
                - tensor dataloader of all variables/features
            
            Max length is +2 in tokenizer because tokenizer will pad start and ending of text, we are only considering length of sentace

            !!! In-future all data related functions will be integrated in the dataloader class !!!
            return: None
        """
        # hatexplain dataset 
        with open(data_file_path) as fp:
            data = json.load(fp)

        keep_ids = []
        for ele in data:
            keep_ids.append(ele['annotation_id'])

        df = pd.read_json(args.data_path+'dataset.json')
        df = df.T

        df = df[df["post_id"].isin(keep_ids)]

        df['text'] = df['post_tokens'].apply(lambda l : " ".join(l))
        df['label_list'] = df['annotators'].apply(lambda ele : [d['label'] for d in ele])
        df['final_label'] = df['label_list'].apply(lambda label_list : max(label_list,key=label_list.count))
        df['label'] = df['final_label'].apply(lambda label : 0 if label=='normal' else 1)


        df = df[[dataset_parameter['sentence_column'], dataset_parameter['label_column']]]
        df[dataset_parameter['sentence_column']] = df[dataset_parameter['sentence_column']].apply(lambda x: preprocess_text(x))


        df['length'] = tokenizer_word_length(df[dataset_parameter['sentence_column']])
        df = df[df.length <dataset_parameter['max_seq_len']]
        
        text = list(df.iloc[:,0])
        label = list(df.iloc[:,1])


        text, label = filter_for_max_len(text, label)

        

        input_ids, attention_mask, word_break, word_break_len = tokenize_word_ritwik(text)


        # to preprocess data and convert all data to torch tensor
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        word_break = torch.tensor(word_break)
        word_break_len = torch.tensor(word_break_len)
        label = torch.tensor(label)
                        
        self.length = len(label)        
        self.dataset = TensorDataset(input_ids, attention_mask, word_break, word_break_len, label)
        
        
    def __len__(self):
        """
            This will return length of dataset
            
            return: (int) length of dataset 
        """
        return self.length

    def __getitem__(self, id):
        """
            Give encoding, mark and label at ith index given by user as id
            
            id: (int) 
            
            return: (list) custom vector at specified index (vector, mask, labels)
        """
        return self.dataset[id] 


def word_break_list(text):
    """
        Generate word break array if nth word in text is break into k piece then nth element 
        in resultant array with be k otherwise 1
        
        For example text is "He cannot be trusted", 
        tokenized text will be "He can ##not be trust ##ed", cannot & trusted broke into 2 sub words
        Its word breakage list will be [1, 2, 1, 2]
        
        dataset: (str) Single text/sentence
        
        return: (list) word breakage list
    """
    tokenized_text =    tokenizer(text, 
                                    max_length=dataset_parameter['max_seq_len'],
                                    padding=True,
                                    truncation=True)
    breakage = []
    counter=0
    for token in tokenized_text:
        #tokenized word have '_' in starting
        if token[:2]!="_":     
            counter+=1
        breakage.append(counter)
    freq = {}
    for i in breakage:
        if i in freq:
            freq[i] = freq[i]+1
        else:
            freq[i] = 1
    freq = list(freq.values())
    return freq    

def dataloader_creator(dataset):
    """
        Generate dataloader of torch class with batch size and sampler defined
        
        dataset: (torch.data.Dataset class) Dataset on which dataloader will be created
        
        return: (torch.data.Dataloader)
    """
    dataset_sampler = RandomSampler(dataset) 
    dataset_dataloader = DataLoader(  dataset=dataset, 
                                      sampler=dataset_sampler, 
                                      batch_size=dataset_parameter['batch_size'])
    return dataset_dataloader
    
def dataloader_spliter(dataset):
    """
        Split the dataset into 3 parts train, test and validation, create dataloader for each part
        
        dataset: (torch.data.Dataset class) Dataset on which split and dataloader will be created
        
        return: (torch.data.Dataloader) 3 Dataloader of train, test and validation
    """
    train_split, validataion_split = math.floor(dataset_parameter['dataset_split'][0]*len(dataset)), math.floor(dataset_parameter['dataset_split'][1]*len(dataset))
    test_split = len(dataset)-train_split-validataion_split
    train_set, val_set, test_set = random_split(dataset, (train_split, validataion_split, test_split))
    
    train_set = dataloader_creator(train_set)
    val_set = dataloader_creator(val_set)
    test_set = dataloader_creator(test_set)
    
    return train_set, val_set, test_set


def dataloader_spliter(dataset):
    """
        Split the dataset into 3 parts train, test and validation, create dataloader for each part
        
        dataset: (torch.data.Dataset class) Dataset on which split and dataloader will be created
        
        return: (torch.data.Dataloader) 3 Dataloader of train, test and validation
    """
    train_split, validataion_split = math.floor(dataset_parameter['dataset_split'][0]*len(dataset)), math.floor(dataset_parameter['dataset_split'][1]*len(dataset))
    test_split = len(dataset)-train_split-validataion_split
    train_set, val_set, test_set = random_split(dataset, (train_split, validataion_split, test_split))
    
    train_set = dataloader_creator(train_set)
    val_set = dataloader_creator(val_set)
    test_set = dataloader_creator(test_set)
    
    return train_set, val_set, test_set


encoder_layers='unfreezed'
if args.encoder_frozen==True:
	encoder_layers = 'freezed'

bias_in_fc = 'bias-in-fc'
if args.bias_in_fc==False:
	bias_in_fc = 'no-bias-in-fc'

cls_token = 'no-cls-token'
if args.add_cls_sep_tokens==True:
	cls_token = 'cls-token'


model_name =  args.encoder_name +'-'+ encoder_layers+'-'+bias_in_fc+'-'+cls_token+'-'+'dataSplit'+str(args.split)


train_dataset = Dataset_loader(args.data_path + 'train_split'+str(args.split)+'.json')
test_dataset = Dataset_loader(args.data_path +'test_split'+str(args.split)+'.json')
val_dataset = Dataset_loader(args.data_path +'val_split'+str(args.split)+'.json')

train_dataloader  = DataLoader(  dataset=train_dataset, batch_size=dataset_parameter['batch_size'])
test_dataloader  = DataLoader(  dataset=test_dataset, batch_size=dataset_parameter['batch_size'])
val_dataloader  = DataLoader(  dataset=val_dataset, batch_size=dataset_parameter['batch_size'])

# train_dataloader, val_dataloader, test_dataloader  = dataloader_spliter(dataset)


"""
# Model architecture """

class MODEL(nn.Module):
    def __init__(self, model):      
        """
            Initialize model and define last layers of fine tuned model
            
            bert: BertModel from Huggingface
            
            return: None
        """
        super(MODEL, self).__init__()
        self.bert_model = model       
    

        # word level with or without demo weights initialization
        if bert_model_parameter['word_level_mean_way']==1:
          self.dense = nn.Linear(768, training_parameter['hidden_layers'][0], bias=False)

        if model_parameter['word_level_mean_way']==2:
                self.register_parameter(name='model_adjacency_weight', 
                                        param=torch.nn.Parameter(torch.rand(model_parameter['second_last_layer_size'],
                                                                            requires_grad=True)))     
        elif model_parameter['word_level_mean_way']==3:
                self.register_parameter(name='word_level_emb_horizontal_weights', 
                                        param=torch.nn.Parameter(torch.rand(dataset_parameter['max_seq_len'],
                                                                            requires_grad=True)))    
        elif model_parameter['word_level_mean_way']==4:
                self.register_parameter(name='word_level_emb_batch_weights', 
                                        param=torch.nn.Parameter(torch.rand(dataset_parameter['max_seq_len'], model_parameter['second_last_layer_size'],
                                                                            requires_grad=True)))    
        elif model_parameter['word_level_mean_way']==5:
                self.flat_dense = nn.Linear(dataset_parameter['max_seq_len']*model_parameter['second_last_layer_size'], model_parameter['second_last_layer_size'])
            

        self.activation = training_parameter["activation_function"]
        self.dropout = nn.Dropout(0.1)      
        self.relu =  nn.ReLU()    
        self.fc1 = nn.Linear(model_parameter['second_last_layer_size'], 128, bias=args.bias_in_fc)
        self.fc2 = nn.Linear(128,2, bias=args.bias_in_fc)
        self.log_softmax = nn.LogSoftmax(dim=1)


        if args.encoder_frozen==False:
	        for param in self.bert_model.parameters():
	            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask, word_level, word_level_len): 
        """
            Pass input data from all layer of model
            
            input_id                  :   (list) Encoding vectors (INT) from BertTokenizerFast
            attention_mask            :   (list) Mask vector (INT [0,1]) from Bert BertTokenizerFast
            average_adjacency_matrix  :   (list of list) Average adj matrix (float (0:1)) defined with stanza dependancy graph and its degree multiplication
            word_level                :   (list) Contain number of sub words broken from parent word (INT)
            word_level_len            :   (INT) Define the length of parent sentence without any tokenization

            
            return: (float [0,1]) Last output from fine tuned model
        """




        #token level embeddings
        model_output = self.bert_model(input_ids=input_ids, 
                                  attention_mask=attention_mask)
        token_level_embedding = model_output.last_hidden_state



        # #word level embeddings initialized 

        word_level_embedding = torch.zeros(input_ids.shape[0], dataset_parameter['max_seq_len'], bert_model_parameter['second_last_layer_size'])
     


        #iterate all text in one batch 
        for batch_no in range(0, input_ids.shape[0]):

        #copy first or starting padding
            start, end = 0, 0

            for word_break_counter in range (0, word_level_len[batch_no]):
                    start = end


                    end = start+word_level[batch_no][word_break_counter]

                    word_level_embedding[batch_no][word_break_counter] = torch.mean(token_level_embedding[batch_no][start:end], 0, True)

        word_level_embedding = word_level_embedding.to(device)

        # word_level_embedding[:, 1:] =0.0


        
        # word_level_embedding = token_level_embedding

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
     
        x = self.fc1(output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)    
      
        x = self.log_softmax(x)

        return x, first_fc_layer_emb, word_level_embedding_flat
        # return output, dense_layer_emb,  word_level_embedding_flat #also returning word_level_embedding to use it in calculating relevance at word-level





    def predict(self, text):
#         processed_text = preprocess_text(text)
  


#         input_ids, attention_mask, word_break, word_break_len = tokenize_word_ritwik([processed_text])
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
model = model.to(device)


def train(epoch):
    """
        In this model will be trained on train_dataloader which is defined earlier

        epoch: epoch for verbose

        return: loss and accuracy on whole training data (mean taken on batch)
    """
    running_batch_loss = 0.0
    running_batch_accuracy = 0.0
    
    model.train()
    with tqdm(train_dataloader, unit="batch") as tepoch:
      
        for input_ids, attention_mask,  word_level, word_level_len, label in tepoch:
            tepoch.set_description(f"Training Epoch {epoch}")
            
            input_ids, attention_mask,  word_level, word_level_len, label = input_ids.to(device), attention_mask.to(device),  word_level.to(device), word_level_len.to(device), label.to(device)
             
            optimizer.zero_grad()
            
            preds, first_fc_layer_emb, word_level_embedding_flat = model(input_ids, attention_mask,  word_level, word_level_len)
            # print('pred: ', preds)
            # print('label: ', label)

            loss = loss_fn(preds, label)

            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            running_batch_loss += batch_loss
            
            batch_accuracy = (((preds.max(1)[1]==label).sum()).item())/label.size(0)
            running_batch_accuracy += batch_accuracy
            
            tepoch.set_postfix(loss=batch_loss, accuracy=(batch_accuracy))

            del input_ids, attention_mask,  word_level, word_level_len, label, preds                 
            
    return running_batch_loss/len(train_dataloader), running_batch_accuracy/len(train_dataloader)
       
def validate(epoch):
    """
        In this model will be validation without changing any parameter of model on val_dataloader

        epoch: epoch for verbose

        return: loss and accuracy on whole training data (mean taken on batch)
    """

    predictions = []
    true_labels = []

    running_batch_loss = 0.0
    running_batch_accuracy = 0.0
    
    model.eval()
    with tqdm(val_dataloader, unit="batch") as tepoch:
        
        for input_ids, attention_mask, word_level, word_level_len, label in tepoch:
            tepoch.set_description(f"Validation Epoch {epoch}")
            
            input_ids, attention_mask,  word_level, word_level_len, label = input_ids.to(device), attention_mask.to(device),  word_level.to(device), word_level_len.to(device), label.to(device)
            
            with torch.no_grad():   
                preds, first_fc_layer_emb, word_level_embedding_flat = model(input_ids, attention_mask, word_level, word_level_len)
                loss = loss_fn(preds, label)
            
                batch_loss = loss.item()
                running_batch_loss += batch_loss
            
                batch_accuracy = (((preds.max(1)[1]==label).sum()).item())/label.size(0)
                running_batch_accuracy += batch_accuracy

                preds = preds.detach().cpu().numpy()
                predictions.append(preds)

                label = label.detach().cpu().numpy()
                true_labels.append(label)
                
            tepoch.set_postfix(loss=batch_loss, accuracy=(batch_accuracy))

            del input_ids, attention_mask,  word_level, word_level_len, label, preds     
    predictions = np.concatenate(predictions, axis=0)    
    predictions = np.argmax(predictions, axis = 1)

    true_labels = np.concatenate(true_labels, axis=0) 

            
    return running_batch_loss/len(test_dataloader), running_batch_accuracy/len(test_dataloader), predictions, true_labels

def test(test_dataloader):
    """
        Test model on test dataset

        return: loss and accuracy on whole training data (mean taken on batch)
    """
    running_batch_loss = 0.0
    running_batch_accuracy = 0.0

    predictions = []
    true_labels = []
    
    model.eval()
    with tqdm(test_dataloader, unit="batch") as tepoch:
        
        for input_ids, attention_mask, word_level, word_level_len, label in tepoch:
            tepoch.set_description(f"Testing on test dataset")
            
            input_ids, attention_mask, word_level, word_level_len, label = input_ids.to(device), attention_mask.to(device),  word_level.to(device), word_level_len.to(device), label.to(device)
            
            with torch.no_grad():
                preds, first_fc_layer_emb, word_level_embedding_flat = model(input_ids, attention_mask,  word_level, word_level_len)
                loss = loss_fn(preds, label)

                batch_loss = loss.item()
                running_batch_loss += batch_loss
            
                batch_accuracy = (((preds.max(1)[1]==label).sum()).item())/label.size(0)
                running_batch_accuracy += batch_accuracy

                preds = preds.detach().cpu().numpy()
                predictions.append(preds)

                label = label.detach().cpu().numpy()
                true_labels.append(label)

            del input_ids, attention_mask, word_level, word_level_len, label, preds
            
            tepoch.set_postfix(loss=batch_loss, accuracy=(batch_accuracy))
    
    predictions = np.concatenate(predictions, axis=0)    
    predictions = np.argmax(predictions, axis = 1)

    true_labels = np.concatenate(true_labels, axis=0) 

    return running_batch_loss/len(test_dataloader), running_batch_accuracy/len(test_dataloader), predictions, true_labels







"""# Train model"""

# optimizer for model
optimizer = AdamW(model.parameters(), lr = training_parameter['optimizer_parameter']['lr'])

# as we are using NLL loss function we can use class weights for imbalance dataset
weights = torch.tensor(training_parameter['loss_func_parameter']['class_weight'], dtype=torch.float)
weights = weights.to(device)

# initialize loss
loss_fn  = nn.NLLLoss(weight=weights)


all_train_loss, all_val_loss = [], []
all_train_accuracy, all_val_accuracy = [], []


if not os.path.exists(args.checkpoint_path+'/reports'):
    os.makedirs(args.checkpoint_path+'/reports')


if not os.path.exists(args.checkpoint_path+'/plots'):
    os.makedirs(args.checkpoint_path+'/plots')


if not os.path.exists(args.checkpoint_path+'/trained_models'):
    os.makedirs(args.checkpoint_path+'/trained_models')

for epoch in range(1, args.epochs+1):
	train_loss, train_accuracy = train(epoch)
	val_loss, val_accuracy, val_predictions, val_true_labels  =  validate(epoch)
	# print('val_predictions: ', val_predictions)
	# print()
	# print('val_true_labels: ', val_true_labels)


	val_report = classification_report(val_true_labels, val_predictions)

	report_path = args.checkpoint_path+'/reports/'+model_name+'classification_report.txt'
	with open(report_path, 'a') as f:
		if epoch==1:
			f.write(f"=====================RUN ID:  {args.run_ID}=======================\n")
		f.write(f"EPOCH {epoch}/{args.epochs}\n")
		f.write(f"validation Loss: {val_loss}\n")
		f.write(val_report)
		f.write('\n\n')

	print(f'Training Loss: {train_loss:.3f}, Accuracy : {train_accuracy:.3f}')
	print(f'Validate Loss: {val_loss:.3f}, Accuracy : {val_accuracy:.3f}')
	print()


	state = {}
	state['state_dict']= model.state_dict()
	state['training_parameter'] = training_parameter
	
	torch.save(state, args.checkpoint_path+'/trained_models/hateSpeechModel-'+model_name+'_epoch_'+str(epoch)+'.pth.t7')

	all_train_loss.append(train_loss)
	all_train_accuracy.append(train_accuracy)
	all_val_loss.append(val_loss)
	all_val_accuracy.append(val_accuracy)



print('Done training model: ', model_name)



# test_loss, test_accuracy, prediction, true_labels = test(test_dataloader)


# # test_loss, test_accuracy, predictions, true_labels
# print('MODEL:'+ model_name+", "+"Classification report on Test data")
# report = classification_report(true_labels, prediction)
# print(confusion_matrix(true_labels, prediction))
# print(report)






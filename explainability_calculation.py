# 

import pandas as pd
import json
from tqdm.notebook import tqdm

# !pip install more_itertools
# !pip install sentencepiece
# !pip install transformers
# !pip install ekphrasis

import more_itertools as mit
import os

from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, AdamW, RobertaTokenizer, RobertaModel

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

import pandas as pd
import json
from tqdm.notebook import tqdm
import more_itertools as mit
import os

from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, AdamW, RobertaTokenizer, RobertaModel

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

# The important key here is the 'bert_token'. Set it to True for Bert based models and False for Others.
import pandas as pd
import json
from tqdm.notebook import tqdm
import more_itertools as mit
import os

from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, AdamW, RobertaTokenizer, RobertaModel

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

dict_data_folder={
      '2':{'data_file':'dataset.json','class_label':'classes_two.npy'},
      '3':{'data_file':'dataset.json','class_label':'classes.npy'}
}

# We need to load the dataset with the labels as 'hatespeech', 'offensive', and 'normal' (3-class). 

params = {}
params['num_classes']=2
params['data_file']=dict_data_folder[str(params['num_classes'])]['data_file']
params['class_names']=dict_data_folder[str(params['num_classes'])]['class_label']

def get_annotated_data(params):
    #temp_read = pd.read_pickle(params['data_file'])
    with open(params['data_file'], 'r') as fp:
        data = json.load(fp)
    dict_data=[]
    for key in data:
        temp={}
        temp['post_id']=key
        temp['text']=data[key]['post_tokens']
        final_label=[]
        for i in range(1,4):
            temp['annotatorid'+str(i)]=data[key]['annotators'][i-1]['annotator_id']
#             temp['explain'+str(i)]=data[key]['annotators'][i-1]['rationales']
            temp['target'+str(i)]=data[key]['annotators'][i-1]['target']
            temp['label'+str(i)]=data[key]['annotators'][i-1]['label']
            final_label.append(temp['label'+str(i)])

        final_label_id=max(final_label,key=final_label.count)
        temp['rationales']=data[key]['rationales']
        if data[key]['post_id']=='13851720_gab':
          print('Inside get_annotated_data()')
          print("len(temp['rationales'][0]): ", len(temp['rationales'][0]))

            
        if(params['class_names']=='classes_two.npy'):
            if(final_label.count(final_label_id)==1):
                temp['final_label']='undecided'
            else:
                if(final_label_id in ['hatespeech','offensive']):
                    final_label_id='toxic'
                else:
                    final_label_id='non-toxic'
                temp['final_label']=final_label_id

        
        else:
            if(final_label.count(final_label_id)==1):
                temp['final_label']='undecided'
            else:
                temp['final_label']=final_label_id

        
        
        
        dict_data.append(temp)    
    temp_read = pd.DataFrame(dict_data)  
    return temp_read  
# data_all_labelled=get_annotated_data(params)

def get_annotated_data(params):
    #temp_read = pd.read_pickle(params['data_file'])
    with open(params['data_file'], 'r') as fp:
        data = json.load(fp)
    dict_data=[]
    for key in data:
        temp={}
        temp['post_id']=key
        temp['text']=data[key]['post_tokens']
        final_label=[]
        for i in range(1,4):
            temp['annotatorid'+str(i)]=data[key]['annotators'][i-1]['annotator_id']
#             temp['explain'+str(i)]=data[key]['annotators'][i-1]['rationales']
            temp['target'+str(i)]=data[key]['annotators'][i-1]['target']
            temp['label'+str(i)]=data[key]['annotators'][i-1]['label']
            final_label.append(temp['label'+str(i)])

        final_label_id=max(final_label,key=final_label.count)
        temp['rationales']=data[key]['rationales']
            
        if(params['class_names']=='classes_two.npy'):
            if(final_label.count(final_label_id)==1):
                temp['final_label']='undecided'
            else:
                if(final_label_id in ['hatespeech','offensive']):
                    final_label_id='toxic'
                else:
                    final_label_id='non-toxic'
                temp['final_label']=final_label_id

        else:
            if(final_label.count(final_label_id)==1):
                temp['final_label']='undecided'
            else:
                temp['final_label']=final_label_id

        
        dict_data.append(temp)    
    temp_read = pd.DataFrame(dict_data)  
    return temp_read  






data_all_labelled=get_annotated_data(params)

params_data={
    'include_special':False,  #True is want to include <url> in place of urls if False will be removed
    'bert_tokens':True, #True /False
    'type_attention':'softmax', #softmax
    'set_decay':0.1,
    'majority':2,
    'max_length':128,
    'variance':10,
    'window':4,
    'alpha':0.5,
    'p_value':0.8,
    'method':'additive',
    'decay':False,
    'normalized':False,
    'not_recollect':True,
}




if(params_data['bert_tokens']):
    print('Loading BERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', do_lower_case=False)
else:
    print('Loading Normal tokenizer...')
    tokenizer=None

# Load the whole dataset and get the tokenwise rationales


import re


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    #corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

def custom_tokenize(sent,tokenizer,max_length=512):
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    try:

        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = False, # Add '[CLS]' and '[SEP]'
                            #max_length = max_length,
                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            #max_length = 128,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.

    except ValueError:
        encoded_sent = tokenizer.encode(
                            ' ',                      # Sentence to encode.
                            add_special_tokens = False, # Add '[CLS]' and '[SEP]'
                            max_length = max_length,
                    
                       )
          ### decide what to later

    return encoded_sent

# Load the whole dataset and get the tokenwise rationales



text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    #corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

def custom_tokenize(sent,tokenizer,max_length=512):
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    try:

        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = False, # Add '[CLS]' and '[SEP]'
                            #max_length = max_length,
                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            #max_length = 128,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.

    except ValueError:
        encoded_sent = tokenizer.encode(
                            ' ',                      # Sentence to encode.
                            add_special_tokens = False, # Add '[CLS]' and '[SEP]'
                            max_length = max_length,
                    
                       )
          ### decide what to later

    return encoded_sent


def ek_extra_preprocess(post_id, text,params,tokenizer):
    remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
    word_list=text_processor.pre_process_doc(text)

        

    if(params['include_special']):
        pass
    else:
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 


    if(params['bert_tokens']):
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        sub_word_list = custom_tokenize(sent,tokenizer)

        return sub_word_list
    else:
        word_list=[token for token in word_list if token not in string.punctuation]
        
        return 


def returnMask(row,params,tokenizer):
    
    text_tokens=row['text']
    
    
    
    ##### a very rare corner case
    if(len(text_tokens)==0):
        text_tokens=['dummy']
        print("length of text ==0")
    #####
    
    
    mask_all= row['rationales']
    post_id = row['post_id']
    
    mask_all_temp=mask_all
    count_temp=0


    while(len(mask_all_temp)!=3):
        mask_all_temp.append([0]*len(text_tokens))
    
    word_mask_all=[]
    word_tokens_all=[]
    
    for mask in mask_all_temp:
        if(mask[0]==-1):
            mask=[0]*len(mask)
        
        
        list_pos=[]
        mask_pos=[]
        
        flag=0


        for i in range(0,len(mask)):
            if(i==0 and mask[i]==0):
                list_pos.append(0)
                mask_pos.append(0)

 
            if(flag==0 and mask[i]==1):
                mask_pos.append(1)
                list_pos.append(i)

                flag=1
                
            elif(flag==1 and mask[i]==0):
                flag=0
                mask_pos.append(0)
                list_pos.append(i)

        if(list_pos[-1]!=len(mask)):

          
          list_pos.append(len(mask))
          mask_pos.append(0)

    
        string_parts=[]
        for i in range(len(list_pos)-1):
            string_parts.append(text_tokens[list_pos[i]:list_pos[i+1]])


        if(params['bert_tokens']):
            word_tokens=[101]
            word_mask=[0]
        else:
            word_tokens=[]
            word_mask=[]

        for i in range(0,len(string_parts)):
            tokens=ek_extra_preprocess(post_id, " ".join(string_parts[i]),params,tokenizer)

            masks=[mask_pos[i]]*len(tokens)
            word_tokens+=tokens
            word_mask+=masks

        if(params['bert_tokens']):
            ### always post truncation
            word_tokens=word_tokens[0:(int(params['max_length'])-2)]
            word_mask=word_mask[0:(int(params['max_length'])-2)]
            word_tokens.append(102)
            word_mask.append(0)


        word_mask_all.append(word_mask)
        word_tokens_all.append(word_tokens)


        # if post_id=='13851720_gab':
        #   print('len(word_mask): ', len(word_mask))
        
#     for k in range(0,len(mask_all)):
#          if(mask_all[k][0]==-1):
#             word_mask_all[k] = [-1]*len(word_mask_all[k])
    if(len(mask_all)==0):
        word_mask_all=[]
    else:  
        word_mask_all=word_mask_all[0:len(mask_all)]

    return word_tokens_all[0],word_mask_all 



def get_training_data(data):
    post_ids_list=[]
    text_list=[]
    attention_list=[]
    label_list=[]
    
    final_binny_output = []
    print('total_data',len(data))
    for index,row in tqdm(data.iterrows(),total=len(data)):
        annotation=row['final_label']
        
        text=row['text']
        post_id=row['post_id']
        annotation_list=[row['label1'],row['label2'],row['label3']]
        # tokens_all = list(row['text'])
        # attention_masks =  [list(row['explain1']),list(row['explain2']),list(row['explain1'])]
        
        if(annotation!= 'undecided'):
            tokens_all,attention_masks=returnMask(row, params_data, tokenizer)
            # print(attention_masks)
            tokens_all = list(row['text'])
            attention_masks = row['rationales']
            final_binny_output.append([post_id, annotation, tokens_all, attention_masks, annotation_list])

    return final_binny_output

training_data=get_training_data(data_all_labelled)

import more_itertools as mit

def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


            
# Convert dataset into ERASER format: https://github.com/jayded/eraserbenchmark/blob/master/rationale_benchmark/utils.py
def get_evidence(post_id, anno_text, explanations):

    output = []

    indexes = sorted([i for i, each in enumerate(explanations) if each==1])
    span_list =  find_ranges(indexes) #list(find_ranges(indexes))


    for each in span_list:
        if type(each)== int:
            start = each
            end = each+1
        elif len(each) == 2:
            start = each[0]
            end = each[1]+1
        else:
            print('error')

  
        output.append({"docid":post_id, 
              "end_sentence": -1, 
              "end_token": end, 
              "start_sentence": -1, 
              "start_token": start, 
              "text": ' '.join([str(x) for x in anno_text[start:end]])})


    return output

# To use the metrices defined in ERASER, we will have to convert the dataset
def convert_to_eraser_format(dataset, method, save_split, save_path, id_division):  
    final_output = []
    
    if save_split:
        train_fp = open(save_path+'train.jsonl', 'w')
        val_fp = open(save_path+'val.jsonl', 'w')
        test_fp = open(save_path+'test.jsonl', 'w')
            
    for tcount, eachrow in enumerate(dataset):
        
        temp = {}
        post_id = eachrow[0]
        post_class = eachrow[1]
        anno_text_list = eachrow[2]

        majority_label = eachrow[1]
        
        if majority_label=='normal':
            continue
# final_binny_output.append([post_id, annotation, tokens_all, attention_masks, annotation_list])

        all_labels = eachrow[4]
        explanations = []
        for each_explain in eachrow[3]:
            explanations.append(list(each_explain))


        # For this work, we have considered the union of explanations. Other options could be explored as well.
        if method == 'union': #if any of annotator says a token is relevant then relevant
            final_explanation = [any(each) for each in zip(*explanations)]
            final_explanation = [int(each) for each in final_explanation]
        
        if post_id=='13162665_gab':
          print(len(final_explanation))
          print(anno_text_list)   
        temp['annotation_id'] = post_id
        temp['classification'] = post_class
        temp['evidences'] = [get_evidence(post_id, list(anno_text_list), final_explanation)]
        temp['query'] = "What is the class?"
        temp['query_type'] = None
        final_output.append(temp)
        
        if save_split:
            if not os.path.exists(save_path+'docs'):
                os.makedirs(save_path+'docs')
            
            with open(save_path+'docs/'+post_id, 'w') as fp:
                fp.write(' '.join([str(x) for x in list(anno_text_list)]))
            
            if post_id in id_division['train']:
                train_fp.write(json.dumps(temp)+'\n')
            
            elif post_id in id_division['val']:
                val_fp.write(json.dumps(temp)+'\n')
            
            elif post_id in id_division['test']:
                test_fp.write(json.dumps(temp)+'\n')
            else:
                print(post_id)
    
    if save_split:
        train_fp.close()
        val_fp.close()
        test_fp.close()
        
    return final_output

# !rm -r Model_Eval

# The post_id_divisions file stores the train, val, test split ids. We select only the test ids.
# with open('post_id_division_split2_seed_12345.json') as fp: 


import json
with open('post_id_divisions.json') as fp:
    id_division = json.load(fp)




import os



if not os.path.exists('Model_Eval'):
    os.makedirs('Model_Eval')

method = 'union'
save_split = True
save_path = 'Model_Eval/'  #The dataset in Eraser Format will be stored here.
convert_to_eraser_format(training_data, method, save_split, save_path, id_division)



if not os.path.exists('explanation_result'):
    os.makedirs('explanation_result')


if not os.path.exists('explanation_dicts'):
    os.makedirs('explanation_dicts')

import sys

# Get the arguments
score_file_name = sys.argv[2]


import json
output_file_name = score_file_name.split('.')[0]+'_metrics.json'


# Open the JSON file
with open('explanation_dicts/'+ "_".join(output_file_name.split('_')[0:-1])+'.json', 'r') as json_file:
    # Parse the JSON data to obtain a list of JSON objects
    data = json.load(json_file)

# Open a new file for writing the JSON Lines data
with open('explanation_dicts/'+ "_".join(output_file_name.split('_')[0:-1])+'.jsonl', 'w') as jsonl_file:
    # Iterate over the list of JSON objects and write each one as a string
    # separated by a newline character
    for json_obj in data:
        jsonl_file.write(json.dumps(json_obj) + '\n')


explanation_scores_file = "_".join(output_file_name.split('_')[0:-1])+'.jsonl'
model_explain_output_file = output_file_name



import subprocess

subprocess.run(['PYTHONPATH=./:$PYTHONPATH', 'python', 'metrics.py', '--split', 'test', '--strict', '--data_dir', 'Model_Eval/', '--results', 'explanation_dicts/'+explanation_scores_file, '--score_file', 'explanation_result/'+model_explain_output_file])


import json
print('======= '+model_explain_output_file+'==========')
with open('explanation_result/'+output_file_name) as fp:
    output_data = json.load(fp)

print('\nPlausibility')
print('IOU F1 :', output_data['iou_scores'][0]['macro']['f1'])
print('Token F1 :', output_data['token_prf']['instance_macro']['f1'])
print('AUPRC :', output_data['token_soft_metrics']['auprc'])

print('\nFaithfulness')
print('Comprehensiveness :', output_data['classification_scores']['comprehensiveness'])
print('Sufficiency', output_data['classification_scores']['sufficiency'])










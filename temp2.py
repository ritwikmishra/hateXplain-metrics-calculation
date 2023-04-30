from ferret import Benchmark
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ferret.explainers.lime import LIMEExplainer
from ferret.explainers.shap import SHAPExplainer

add_cls_sep_token_ids = False

def tokenize_word_ritwik(text):
    input_ids, attention_mask, word_break, word_break_len, tag_one_hot, tag_attention = [],[],[],[], [] , []

    # assert len(text) == len(tags)
    # onehot_encoding = pd.get_dummies(dataset_parameter['tag_set'])
    for text_i in text:
        input_ids_i, attention_mask_i, word_break_i, word_break_len_i, tag1hot_i, tag_att_i = ['<s>'] if add_cls_sep_token_ids else [], [], [], 0, [], [0] if add_cls_sep_token_ids else []

        # tokenized_text_i = []
        # doc = nlp(text_i)
        # for sents in doc.sentences:
        #   for word in sents.words:
        #       tokenized_text_i.append(word.text)
        tokenized_text_i = text_i.split() # since humne pehle filtering kardi hai, yahan pe dobara stanza se pass karne ki zarurat ni hai

        for i, word in enumerate(tokenized_text_i):
            input_ids_per_word = tokenizer.tokenize(word)
            print('--',word, input_ids_per_word)
            input('wait')
     
            # if len(input_ids_per_word) > 1:
            #     if model_parameter['token_ids_way']==1:
            #         input_ids_per_word = [input_ids_per_word[0]]
            #         word_break_i.append(1)
            #     elif model_parameter['token_ids_way']==2:
            #         input_ids_per_word = [input_ids_per_word[-1]]
            #         word_break_i.append(1)
            #     elif model_parameter['token_ids_way']==3:
            #         word_break_i.append(len(input_ids_per_word))
            #     else:
            #         raise "Error ajeeb"
            # else:
            #     word_break_i.append(1)
            # input_ids_i += input_ids_per_word

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


device = 'cuda:0'

name = "Hate-speech-CNERG/hindi-abusive-MuRIL"
model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
tokenizer = AutoTokenizer.from_pretrained(name)

bench = Benchmark(model, tokenizer)
explanations = bench.explain('laura loomer screamed at me', target=1)
df = bench.get_dataframe(explanations)

print(df)
print(df.columns)
print(len(df))

print(tokenize_word_ritwik(['laura loomer screamed at me']))
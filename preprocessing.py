# load data from directory(google drive)

import numpy as np
import pandas as pd
import csv

from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

pos_tagger = Okt()
stopwords=['뭐','으면','을','의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

def load_data_from_directory(file_name):
    print("load", file_name)
    f = open(file_name, 'r')
    rdr = csv.reader(f)
    data = []
    
    for i, line in enumerate(rdr):
        if line:
            data.append(line[0])
        else:
            data.append(np.nan)

    f.close()

    return data

# sampling for label 10 data
def sampling(train_data, n):
    return pd.concat([train_data[train_data['label'] != 10], train_data[train_data['label'] == 10].sample(n=n)])

# tokenize data frame with Okt module
def tokenize(df_data):
    data = []
    for i, sentence in enumerate(df_data):
        tmp = []
        tmp = pos_tagger.morphs(sentence)

        tokenized = []
        for token in tmp:
            if not token in stopwords:
                tokenized.append(token)

            data.append(tokenized)

            if i % 10000 == 0:
                print(i)

    return data

# sampling training data 
def train_down_sampling(train_data, down_ratio=0.2):
  x_, train_x, y_, train_y = train_test_split(
                              train_data['review'],
                              train_data['label'],
                              test_size=down_ratio,
                              shuffle=True,
                              stratify=train_data['label'],
                              random_state=34
                            )
  
  return train_x, train_y

# load data and remain only alphabet and space in review
def load_data(score_sampling, down_ratio):
    # load train data
    train_review = np.array(load_data_from_directory('train_data'))
    train_label = np.array(load_data_from_directory('train_label')).astype(np.int32)
    train_data = pd.DataFrame({'review' : train_review, 'label' : train_label}, columns={'review', 'label'})

    # load valid data
    valid_review = np.array(load_data_from_directory('valid_data'))
    valid_label = np.array(load_data_from_directory('valid_label')).astype(np.int32)
    valid_data = pd.DataFrame({'review' : valid_review, 'label' : valid_label}, columns={'review', 'label'})

    # load test data
    test_review = np.array(load_data_from_directory('test_data'))
    test_data = pd.DataFrame({'review' : test_review,}, columns={'review'})

    # down sampling for score 10
    train_data = sampling(train_data, score_sampling)

    # drop duplicate data. only remain one
    train_data = train_data.drop_duplicates(['review'], keep='first')

    # remain only Hangul alphabet
    train_data['review'] = train_data['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z ]","")
    valid_data['review'] = valid_data['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z ]","")
    test_data['review'] = test_data['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z ]","")

    # remove space
    train_data['review'] = train_data['review'].str.replace('^ +', "")
    train_data['review'].replace('', np.nan, inplace=True)

    valid_data['review'] = valid_data['review'].str.replace('^ +', "")
    valid_data['review'].replace('', np.nan, inplace=True)

    # drop nan data
    train_data = train_data.dropna()
    print("length of valid data before removing nan:", len(valid_data))
    valid_data = valid_data.dropna()
    print("length of valid data after removing nan:", len(valid_data))

    train_x, train_y = train_down_sampling(train_data, down_ratio)
    train_data = pd.DataFrame({'review' : train_x, 'label' : train_y}, columns={'review', 'label'})

    return train_data, valid_data, test_data

# preprocessing data
def preprocessing_data(train_data, valid_data, threshold, max_len):

    train_x = tokenize(train_data['review'])
    train_y = np.array(train_data['label'])
    train_y = train_y - 1
    
    valid_x = tokenize(valid_data['review'])
    valid_y = np.array(valid_data['label'])
    valid_y = valid_y - 1

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_x)
    
    total_cnt = len(tokenizer.word_index) # 단어의 수
    rare_cnt = 0
    total_freq = 0
    rare_freq = 0

    for key, value in tokenizer.word_counts.items():
        total_freq += value

        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    vocab_size = total_cnt - rare_cnt + 1
    print("vocab_size :",vocab_size)

    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(train_x)
    train_x_sequences = tokenizer.texts_to_sequences(train_x)
    valid_x_sequences = tokenizer.texts_to_sequences(valid_x)

    drop_train = [index for index, sentence in enumerate(train_x_sequences) if len(sentence) < 1]
    train_x_sequences = np.delete(train_x_sequences, drop_train, axis = 0)
    train_y = np.delete(train_y, drop_train, axis = 0)

    drop_valid = [index for index, sentence in enumerate(valid_x_sequences) if len(sentence) < 1]
    valid_x_sequences = np.delete(valid_x_sequences, drop_valid, axis = 0)
    valid_y = np.delete(valid_y, drop_valid, axis = 0)

    train_x_pads = pad_sequences(train_x_sequences, maxlen=max_len, padding='post')
    valid_x_pads = pad_sequences(valid_x_sequences, maxlen=max_len, padding='post')

    return train_x_pads, train_y, valid_x_pads, valid_y, tokenizer, vocab_size # test x

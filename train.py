# Movie rating prediction (Korean)

import tensorflow as tf
import numpy as np
import csv

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow import keras
from preprocessing import tokenize, load_data, preprocessing_data, 

score_sampling = 1000000 # remove score 10 for downsampling
embedding_size = 300
batch_size = 256
threshold = 3 # minimum frequency in train data
max_len = 30 # length of sequence vector
down_ratio = 0.2
path = '/content/gdrive/MyDrive/Algorima_Technical_Assignment/output.csv'

class Model(keras.models.Model):
    def __init__(self,
                vocab_size,
                embedding_size,
                name='model'
                ):
        super(Model, self).__init__(name=name)
        self.embedding = Embedding(vocab_size, embedding_size)
        self.bilstm = Bidirectional(LSTM(64, dropout=0.5))
        self.dense = Dense(10, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.001))

    def call(self, inputs):
        embedded_inputs = self.embedding(inputs)
        h_lstm = self.bilstm(embedded_inputs)
        score = self.dense(h_lstm)
        
        return score

# predict for test data and save result csv file
def predict_and_save_result(test_data, tokenizer, model, path):
    test_x = tokenize(test_data['review'])
    result = np.zeros(len(test_data))
    test_x_sequences = tokenizer.texts_to_sequences(test_x)
    drop_test = [index for index, sentence in enumerate(test_x_sequences) if len(sentence) < 1]
    result[drop_test] = 10
    test_x_sequences = np.delete(test_x_sequences, drop_test, axis=0)
    test_x_pads = pad_sequences(test_x_sequences, maxlen=max_len, padding='post')

    test_prediction = np.argmax(model.predict(test_x_pads), axis=1)
    i = 0
    for p in test_prediction:
        while(result[i] == 10):
            i += 1
        result[i] = p + 1
        i += 1

    result = result.astype(np.int32)
    write_result = [["ID", "Prediction"]]

    for i, r in enumerate(result):
        write_result.append([i,r])

    with open(path, 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(write_result)

train_data, valid_data, test_data = load_data(score_sampling, down_ratio)
train_x_pads, train_y, valid_x_pads, valid_y, tokenizer, vocab_size = preprocessing_data(
                                                                                        train_data,
                                                                                        valid_data,
                                                                                        threshold,
                                                                                        max_len
                                                                                    )

model = Model(vocab_size, embedding_size)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['acc'],
)

history = model.fit(
            train_x_pads,
            train_y,
            batch_size=batch_size,
            epochs = 1,
            validation_data=(valid_x_pads, valid_y)
)

model.save_weights("/content/gdrive/MyDrive/Algorima_Technical_Assignment/model/save_model_1000000_embedding_300_dropout")
predict_and_save_result(test_data, tokenizer, model, path)

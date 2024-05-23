import sys
sys.path.append('E:/ai_chatbot')

import pandas as pd
import tensorflow as tf
from keras import preprocessing
from keras import Model
from keras.layers import Input, Embedding, Dense, Dropout, LSTM

from util.Preprocess import Preprocess
from config.GlobalParams import GlobalParams

GlobalParams()

# Load Data
data = pd.read_csv('E:/ai_chatbot/models/intent/train_data.csv')
text = data['text'].tolist()
label = data['label'].tolist()

# Load Preprocessor
p = Preprocess(word2index_dic='E:/ai_chatbot/train_tools/dict/chatbot_dict.bin',
               userdic='E:/ai_chatbot/util/user_dic.tsv')

# Data Preprocess
sequences = []
for sentence in text:
    pos = p.pos(sentence)
    keywords = p.get_keywords(pos, without_tag=True)
    seq = p.get_wordidx_sequence(keywords)
    sequences.append(seq)

# Set padding length & pad sequences
from config.GlobalParams import MAX_SEQ_LEN
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

# Data to Tensor
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, label))
ds = ds.shuffle(len(text))

# Set Train & Validation & Test size
train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

train_ds = ds.take(train_size).batch(100)
val_ds = ds.skip(train_size).take(val_size).batch(100)
test_ds = ds.skip(train_size + val_size).batch(100)

# Hyperparameter
dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 3
VOCAB_SIZE = len(p.word_index) + 1

# RNN(LSTM) model definition
input_layer = Input(shape=(MAX_SEQ_LEN,))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE)(input_layer)
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

lstm = LSTM(128, return_sequences=True)(dropout_emb)  # 첫 번째 LSTM 레이어
lstm2 = LSTM(128, return_sequences=False)(lstm)  # 두 번째 LSTM 레이어

hidden = Dense(128, activation=tf.nn.relu)(lstm2)  # lstm2를 연결
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
predictions = Dense(4, activation="softmax")(dropout_hidden)

# RNN(LSTM) 모델 생성
model = Model(inputs=input_layer, outputs=predictions)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)

# Evaluation model
loss, accuracy = model.evaluate(test_ds, verbose=1)
print("Accuracy: %f" % (accuracy * 100))
print("Loss: %f" % (loss * 100))

# Save model
model.save('E:/ai_chatbot/models/intent/intent_model.h5')
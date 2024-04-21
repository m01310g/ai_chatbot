'''
#  BERT 모델2
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout
from keras.models import Model

# Load Data
data = pd.read_csv('E:/ai_chatbot/models/intent/train_data.csv')
text = data['text'].tolist()
label = data['label'].tolist()

# Load BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", return_attention_mask=True)

# Tokenize text using BERT tokenizer
input_ids = []
attention_masks = []

for sent in text:
    encoded_dict = tokenizer.encode_plus(
                        sent,                     
                        add_special_tokens = True, 
                        max_length = 128,          
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'tf',     
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# input_ids = np.array(input_ids)
# attention_masks = np.array(attention_masks)
# input_ids = input_ids.squeeze(axis=1)
# attention_masks = attention_masks.squeeze(axis=1)
input_ids = [np.array(ids) for ids in input_ids]
attention_masks = [np.array(mask) for mask in attention_masks]
input_ids = np.array(input_ids)
attention_masks = np.array(attention_masks)
labels = np.array(label)

# Split data into train, validation, and test sets
train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels = train_test_split(input_ids, attention_masks, labels, test_size=0.1, random_state=42)
train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(train_inputs, train_masks, train_labels, test_size=0.1, random_state=42)

# Define BERT model
bert_model = TFBertModel.from_pretrained("bert-base-multilingual-cased")

# Apply BERT model
# output = bert_model(input_ids, attention_mask=attention_masks)[0]
output = bert_model({'input_ids':input_ids, 'attention_mask':attention_masks})[0]

# Add additional layers
dropout = Dropout(0.1)(output[:, 0, :])  
dense = Dense(64, activation='relu')(dropout)
output = Dense(3, activation='softmax')(dense)

# Compile model
model = Model(inputs=[input_ids, attention_masks], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit([train_inputs, train_masks], train_labels, validation_data=([val_inputs, val_masks], val_labels), epochs=3, batch_size=32)

# Evaluate model
loss, accuracy = model.evaluate([test_inputs, test_masks], test_labels)
print("Accuracy: %f" % (accuracy * 100))
print("Loss: %f" % (loss))

# Save model
model.save('E:/ai_chatbot/models/intent/intent_model_bert.h5')
'''

'''
# BERT 모델
import sys
sys.path.append('E:/ai_chatbot')

import pandas as pd
import tensorflow as tf
import numpy as np
import sklearn

from keras import Model
from keras.layers import Input, Dense, Dropout

from transformers import TFBertModel, BertTokenizer

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

# BERT 토크나이저와 모델 로드
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = TFBertModel.from_pretrained("bert-base-multilingual-cased")

# 텍스트를 BERT의 입력 형식에 맞게 변환
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

MAX_LEN = 128  # BERT에 주입할 최대 길이

# 인코딩
padded_seqs, attention_masks, segment_ids = bert_encode(text, bert_tokenizer, MAX_LEN)

# 데이터 준비
from sklearn.model_selection import train_test_split

# 훈련 데이터와 테스트 데이터 분할
train_texts, test_texts, train_labels, test_labels = train_test_split(text, label, test_size=0.2, random_state=42)

# 훈련 데이터와 검증 데이터 further 분할
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

# 인코딩
train_input_word_ids, train_input_mask, train_segment_ids = bert_encode(train_texts, bert_tokenizer, MAX_LEN)
val_input_word_ids, val_input_mask, val_segment_ids = bert_encode(val_texts, bert_tokenizer, MAX_LEN)
test_input_word_ids, test_input_mask, test_segment_ids = bert_encode(test_texts, bert_tokenizer, MAX_LEN)

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

# 훈련 및 검증 데이터셋 준비
train_data = [train_input_word_ids, train_input_mask, train_segment_ids]
validation_data = ([val_input_word_ids, val_input_mask, val_segment_ids], val_labels) # 검증 데이터셋 설정

# BERT 모델 생성
input_word_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")
input_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_mask")
segment_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="segment_ids")

# BERT 모델의 출력을 받는 레이어
bert_layer = TFBertModel.from_pretrained("bert-base-multilingual-cased")
bert_outputs = bert_layer([input_word_ids, input_mask, segment_ids])

# BERT의 pooled_output을 사용하여 분류 수행
pooled_output = bert_outputs[1]
dense_layer = Dense(64, activation='relu')(pooled_output)
dropout = Dropout(0.1)(dense_layer)
predictions = Dense(len(np.unique(label)), activation='softmax')(dropout)

# 모델 구성
model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=predictions)

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 훈련 및 검증 데이터셋 준비 (validation_data 수정)
validation_data = ([val_input_word_ids, val_input_mask, val_segment_ids], val_labels)

# 모델 훈련
model.fit(train_data, train_labels, validation_data=validation_data, epochs=3, batch_size=32)

# 모델 평가
loss, accuracy = model.evaluate([test_input_word_ids, test_input_mask, test_segment_ids], test_labels, verbose=1)
print("Accuracy: %f" % (accuracy * 100))
print("Loss: %f" % (loss))


# Save model
model.save('E:/ai_chatbot/models/intent/intent_model.h5')

'''
'''
# LSTM 모델
import sys
sys.path.append('E:/ai_chatbot')

import pandas as pd
import tensorflow as tf
import numpy as np
import fasttext

# from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from keras import preprocessing
from keras import Model
from keras.layers import Input, Embedding, Dense, Dropout, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant

# from keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate

from util.Preprocess import Preprocess
from config.GlobalParams import GlobalParams

GlobalParams()

# Load Data
data = pd.read_csv('E:/ai_chatbot/models/intent/train_data.csv')
text = data['text'].tolist()
label = data['label'].tolist()

# FastText 임베딩 파일 경로
embedding_file = 'E:/ai_chatbot/util/wiki.ko.bin'

# FastText 모델 로드
# fasttext_model = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
fasttext_model = load_facebook_vectors(embedding_file)

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

from keras.layers import LSTM

# 임베딩 차원
embedding_dim = fasttext_model.vector_size

# 단어 인덱스를 사용하여 임베딩 매트릭스 생성
embedding_matrix = np.zeros((VOCAB_SIZE, embedding_dim))
for word, i in p.word_index.items():
    if word in fasttext_model:
        embedding_vector = fasttext_model[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# 사전 훈련된 임베딩을 사용하는 Embedding 레이어 생성
# embedding_layer = Embedding(
#     input_dim=VOCAB_SIZE,
#     output_dim=embedding_dim,
#     embeddings_initializer=Constant(embedding_matrix),
#     input_length=MAX_SEQ_LEN,
#     trainable=False
# )

embedding_layer = Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=embedding_dim,
    embeddings_initializer=Constant(embedding_matrix),
    trainable=False
)

# LSTM 모델 정의
# input_layer = Input(shape=(MAX_SEQ_LEN,))
# embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE)(input_layer)
# dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

# lstm_layer = LSTM(128)(dropout_emb)  # LSTM 레이어 추가

# hidden = Dense(128, activation=tf.nn.relu)(lstm_layer)
# dropout_hidden = Dropout(rate=dropout_prob)(hidden)
# predictions = Dense(3, activation="softmax")(dropout_hidden)

# 양방향 LSTM 모델
input_layer = Input(shape=(MAX_SEQ_LEN,))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE)(input_layer)
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

# 양방향 LSTM 레이어 추가
bi_lstm_layer = Bidirectional(LSTM(128))(dropout_emb)

hidden = Dense(128, activation=tf.nn.relu)(bi_lstm_layer)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
predictions = Dense(3, activation="softmax")(dropout_hidden)

# LSTM 모델 생성
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
model.save('E:/ai_chatbot/models/intent/intent_model.')

# 모델 구조 json 저장
model_json = model.to_json()
with open('E:/ai_chatbot/models/intent/intent_structure.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('E:/ai_chatbot/models/intent/intent_weights.weights.h5')
'''

'''
# CNN 모델
import sys
sys.path.append('E:/ai_chatbot')

import pandas as pd
import tensorflow as tf
from keras import preprocessing
from keras import Model
from keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate

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

# CNN model definition
input_layer = Input(shape=(MAX_SEQ_LEN,))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE)(input_layer)
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

conv1 = Conv1D(
    filters=128,
    kernel_size=3,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(
    filters=128,
    kernel_size=4,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(
    filters=128,
    kernel_size=5,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

concat = concatenate([pool1, pool2, pool3])

hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
logits = Dense(4, name='logits')(dropout_hidden)
predictions = Dense(4, activation="softmax")(logits)

# CNN model 생성
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
'''

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
predictions = Dense(3, activation="softmax")(dropout_hidden)

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
# 단어 사전 파일 생성
# 챗봇에 사용하는 사전 파일
import sys
sys.path.append('E:/ai_chatbot')

from util.Preprocess import Preprocess
import tensorflow as tf
import pickle
import pandas as pd
import nltk

nltk.download('punkt')

path = 'E:/ai_chatbot/변형데이터/'

# 말뭉치 데이터 읽어오기
purpose_data = pd.read_csv(path + '용도별목적대화데이터.csv')
topic_data = pd.read_csv(path + '주제별일상대화데이터.csv')
common_sense_data = pd.read_csv(path + '일반상식.csv')
add_data = pd.read_csv(path + 'AllData.csv')
region_data = pd.read_csv(path + '지역명데이터.csv')

# 결측치 제거
purpose_data.dropna(inplace=True)
topic_data.dropna(inplace=True)
common_sense_data.dropna(inplace=True)
add_data.dropna(inplace=True)
region_data.dropna(inplace=True)

# 필요한 데이터 리스트로 변환
text1 = list(purpose_data['text'])
text2 = list(topic_data['text'])
text3 = list(common_sense_data['query']) + list(common_sense_data['answer'])
text4 = list(add_data['req']) + list(add_data['res'])
text5 = list(region_data['SIDO_NM']) + list(region_data['SGG_NM']) + list(region_data['DONG_NM'])

corpus_data = text1 + text2 + text3 + text4 + text5

# 말뭉치 데이터에서 키워드만 추출 -> 사전 리스트 생성
p = Preprocess()
dict = []
for c in corpus_data:
    pos = p.pos(c)
    for k in pos:
        dict.append(k[0])

# 사전에 사용될 word2index 생성
# 사전의 첫번째 인덱스에는 OOV 사용
# tokenizer = Tokenizer(oov_token='OOV', num_words=100000)

# 사전에 사용될 word2index 생성 전 중복 제거
dict_set = set(dict)  # 중복 제거
tokenized_corpus = [nltk.tokenize.word_tokenize(text) for text in dict_set]

# 단어 인덱스 사전 생성
word_index = {word:i+1 for i, word in enumerate(dict_set)}  # OOV에 대한 처리를 위해서 1부터 인덱싱
word_index['OOV'] = 0
print(f"중복 제거 전 단어 수: {len(dict)}, 중복 제거 후 단어 수: {len(dict_set)}")

# 중복 제거된 리스트 사용
# tokenizer.fit_on_texts(list(dict_set))

# word_index = tokenizer.word_index
print(len(word_index))

# 사전 파일 생성
f = open("E:/ai_chatbot/train_tools/dict/chatbot_dict.bin", "wb")
try:
    pickle.dump(word_index, f)
except Exception as e:
    print(e)
finally:
    f.close()
import sys
sys.path.append('E:/ai_chatbot')

from util.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel
from keras.models import Model

p = Preprocess(word2index_dic='E:/ai_chatbot/train_tools/dict/chatbot_dict.bin',
               userdic='E:/ai_chatbot/util/user_dic.tsv')

# intent = IntentModel(model_name='E:/ai_chatbot/models/intent/intent_structure.json', weights_name='E:/ai_chatbot/models/intent/intent_weights.weights.h5', preprocess=p)
intent = IntentModel(model_name='E:/ai_chatbot/models/intent/intent_model.h5', preprocess=p)

query = "강원도에 스키 타러 갈거야"
predict = intent.predict_class(query)
predict_label = intent.label[predict]
print("=" * 30)
print(query)
print("의도 예측 클래스: ", predict)
print("의도 예측 레이블: ", predict_label)

query = ("부산 1박 2일로 해수욕장 갈거야")
predict = intent.predict_class(query)
predict_label = intent.label[predict]
print("=" * 30)
print(query)
print("의도 예측 클래스: ", predict)
print("의도 예측 레이블: ", predict_label)
import sys
sys.path.append('E:/ai_chatbot')

from util.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel
from keras.models import Model

p = Preprocess(word2index_dic='E:/ai_chatbot/train_tools/dict/chatbot_dict.bin',
               userdic='E:/ai_chatbot/util/user_dic.tsv')

# intent = IntentModel(model_name='E:/ai_chatbot/models/intent/intent_structure.json', weights_name='E:/ai_chatbot/models/intent/intent_weights.weights.h5', preprocess=p)
intent = IntentModel(model_name='E:/ai_chatbot/models/intent/intent_model.h5', preprocess=p)

query = "부산의 관광지를 보여줘"
predict = intent.predict_class(query)
predict_label = intent.label[predict]
print("=" * 30)
print(query)
print("의도 예측 클래스: ", predict)
print("의도 예측 레이블: ", predict_label)

query = ("부산 여행 일정 좀 알려줘")
predict = intent.predict_class(query)
predict_label = intent.label[predict]
print("=" * 30)
print(query)
print("의도 예측 클래스: ", predict)
print("의도 예측 레이블: ", predict_label)
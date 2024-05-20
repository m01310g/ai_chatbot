from flask import Flask, request, jsonify

from models.intent.IntentModel import IntentModel
from response.response import Response
from util.Preprocess import Preprocess

app = Flask(__name__)


# 이전에 작성한 함수들을 import 해줍니다.

@app.route('/generate_response', methods=['POST'])
def generate_response():
    data = request.json
    user_query = data['user_query']

    # 의도 파악 및 응답 생성
    p = Preprocess(word2index_dic='E:/ai_chatbot/train_tools/dict/chatbot_dict.bin',
                   userdic='E:/ai_chatbot/util/user_dic.tsv')

    intent = IntentModel(model_name='E:/ai_chatbot/models/intent/intent_model.h5', preprocess=p)

    predict = intent.predict_class(user_query)
    user_intent = intent.label[predict]

    response_generator = Response(user_intent, user_query)
    response = response_generator.generate_response(user_intent, user_query)

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

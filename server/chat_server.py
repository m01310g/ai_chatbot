from flask import Flask, request, jsonify
import sys
sys.path.append("E:/ai_chatbot")
from response.response import Response

app = Flask(__name__)
response = Response()

@app.route('/chatbot', methods=['POST'])
def chatbot():
    if request.method == 'POST':
        data = request.get_json()
        user_query = data['user_query']

        # 챗봇 기능
        result = response.main(user_query)

        return jsonify({'response': result})


if __name__ == '__main__':
    app.run(debug=True)
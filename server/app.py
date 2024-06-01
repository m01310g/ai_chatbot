from flask import Flask, request, jsonify, abort
import socket
import json
import sys
sys.path.append('E:/ai_chatbot/')

# 챗봇 엔진 서버 정보
host = "127.0.0.1"      # 챗봇 엔진 서버 IP
port = 5050

# Flask 애플리케이션
app = Flask(__name__)

# 챗봇 엔진 서버와 통신
def get_response_from_engine(query):
    try:
        # 챗봇 엔진 서버 연결
        mySocket = socket.socket()
        mySocket.connect((host, port))

        # 챗봇 엔진 질의 요청
        json_data = {
            'query': query,
        }

        message = json.dumps(json_data)
        mySocket.send(message.encode('utf-8'))

        # 챗봇 엔진 답변 출력
        data = mySocket.recv(2048).decode('utf-8')
        ret_data = json.loads(data)

        # 챗봇 엔진 서버 연결 소켓 닫기
        mySocket.close()
        return ret_data
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return {"error": "Invalid JSON format"}
    except Exception as ex:
        print(f"Exception occurred: {ex}")
        return {"error": "An error occurred while processing the request"}

# 초기 메시지 반환
@app.route('/init', methods=['GET'])
def index():
    try:
        message = '안녕하세요, 어디GO입니다!\n 더욱 자세한 응답을 위해서 여행지, 여행기간, 하고 싶은 활동 등을 입력해주세요\n (예: 부산으로 1박 2일 정도 해수욕장에 놀러 갈거야)'
        json_data = {
            'message':message
        }
        message = json.dumps(json_data, ensure_ascii=False)
        message = json.loads(message)
        return jsonify(message)

    except Exception as ex:
        print(f"Exception occured: {ex}")
        abort(500)


# 챗봇 엔진 query 전송 API
@app.route('/query', methods=['POST'])
def query():
    body = request.get_json()
    try:
        # 일반 질의응답 API
        ret = get_response_from_engine(query=body['query'])
        return jsonify(ret)

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return jsonify({"error": "Invalid JSON format"}), 400

    except KeyError as e:
        print(f"Key Error: {e}")
        return jsonify({"error": f"Missing key: {e}"}), 400

    except Exception as ex:
        print(f"Exception occurred: {ex}")
        # 오류 발생시 500 Error
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
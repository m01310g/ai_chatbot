# import requests
# import json
#
# url = 'http://localhost:5000/generate_response'
# headers = {'Content-Type': 'application/json'}
# data = {'user_query': '서울에 있는 맛집 추천해줘'}
#
# response = requests.post(url, headers=headers, data=json.dumps(data))
# print(response.json())

import socket
import json

# 챗봇 엔진 서버 접속 정보
host = "127.0.0.1"
port = 5050

# 클라이언트 프로그램 시작
while True:
    print("질문: ")
    query = input()
    if query.lower() == "exit":
        break
    print("=" * 40)

    # 챗봇 엔진 서버 연결
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as mySocket:
        mySocket.connect((host, port))
        print("서버 연결 완료")

        # 챗봇 엔진 질의 요청
        json_data = {
            'Query': query,
            'BotType': "Test",
        }

        message = json.dumps(json_data)
        mySocket.send(message.encode())
        print("질의 전송 완료")

        # 챗봇 엔진 답변 출력
        data = mySocket.recv(2048).decode()
        print("응답 수신 완료")
        ret_data = json.loads(data)
        # try:
        #     ret_data = json.loads(data)
        # except json.decoder.JSONDecodeError as e:
        #     print(f"Error decoding JSON: {e}")
        #     # Handle error gracefully (e.g., log error, provide default response)
        #     continue
        print("답변: ")
        print(ret_data['Response'])
        print()
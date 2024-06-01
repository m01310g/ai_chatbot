'''
소켓 서버
'''

import threading
import json
import socket
import sys
sys.path.append('E:/ai_chatbot/')

from util.BotServer import BotServer
from response.generate_response import Response

# 데이터를 청크 단위로 전송
def send_Data(conn, data):
    chunk_size = 1024
    encoded_data = data.encode('utf-8')

    for i in range(0, len(encoded_data), chunk_size):
        chunk = encoded_data[i:i+chunk_size]
        conn.sendall(chunk)

def to_client(conn, addr):
    try:
        read = conn.recv(2048)  # 수신 데이터가 있을 때까지 블로킹
        print("Connection from: %s" % str(addr))

        # 클라이언트 연결이 끊어지거나 오류가 있는 경우
        if not read:
            print("클라이언트 연결 오류")
            conn.close()
            return

        # 원시 데이터 출력
        print("Raw data receieved: ", read)

        # json 데이터로 변환
        recv_json_data = json.loads(read.decode('utf-8'))
        print("데이터 수신: ", recv_json_data)
        query = recv_json_data['query']
        print(query)

        # 의도 분류
        # intent_pred = intent.predict_class(query)
        # intent_label = intent.labels[intent_pred]
        # print(f"의도 분류 결과: {intent_label}")

        response_instance = Response()
        user_intent, location, sub_category, duration = response_instance.main(query)

        # 답변 검색
        # result = Response.generate_response(intent_label, query)
        # print(f"검색된 답변: {result}")

        # 결과를 JSON 형식으로 변환
        response_data = {
            'query': query,
            'response': {
                'user_intent':user_intent,
                'location':location,
                'sub_category':sub_category,
                'duration':duration
            }
        }
        response_json = json.dumps(response_data, ensure_ascii=False)

        # 응답 전송
        send_Data(conn, response_json)
        print("응답 전송 완료")

    except Exception as ex:
        print("Error: ", ex)
    finally:
        conn.close()  # 연결 닫기

if __name__ == "__main__":
    # chatbot 서버 동작
    port = 5050
    listen = 1000
    bot = BotServer(port, listen)
    bot.create_sock()
    print("챗봇 시작")

    while True:
        conn, addr = bot.client_ready()
        client = threading.Thread(target=to_client, args=(conn, addr))
        client.start()

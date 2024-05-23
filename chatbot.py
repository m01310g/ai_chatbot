"""
챗봇 엔진 서버
"""

import threading
import json
import socket

from util.BotServer import BotServer
from response.response import Response

def to_client(conn, addr):
    try:
        read = conn.recv(2048)  # 수신 데이터가 있을 때까지 블로킹
        print("Connection from: %s" % str(addr))

        # 클라이언트 연결이 끊어지거나 오류가 있는 경우
        if read is None or not read:
            print("클라이언트 연결 오류")
            conn.close()
            return

        # json 데이터로 변환
        recv_json_data = json.loads(read.decode())
        print("데이터 수신: ", recv_json_data)
        query = recv_json_data['Query']

        # 의도 분류
        # intent_pred = intent.predict_class(query)
        # intent_label = intent.labels[intent_pred]
        # print(f"의도 분류 결과: {intent_label}")

        response_instance = Response()
        result = response_instance.main(query)

        # 답변 검색
        # result = Response.generate_response(intent_label, query)
        # print(f"검색된 답변: {result}")

        send_json_data_str = {
            "Query": query,
            "Response": result
        }

        message = json.dumps(send_json_data_str)  # json 객체 문자열로 반환
        print("전송할 데이터:", message)  # 추가된 디버깅 코드
        conn.send(message.encode())  # 응답 전송
        print("응답 전송 완료")

    except Exception as ex:
        print("Error: ", ex)
    finally:
        conn.close()  # 연결 닫기

if __name__ == "__main__":
    # chatbot 서버 동작
    port = 5000
    listen = 1000
    bot = BotServer(port, listen)
    bot.create_sock()
    print("챗봇 시작")

    while True:
        conn, addr = bot.client_ready()
        client = threading.Thread(target=to_client, args=(conn, addr))
        client.start()
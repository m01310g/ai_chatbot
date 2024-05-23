import socket

class BotServer:
    """
    srv_port: 소켓 서버의 포트번호
    listen_num: 동시 접속 클라이언트 수
    create_sock: TCP/IP 소켓 생성&서버 포트로 설정한 수만큼 클라이언트 연결 수락
    client_ready: 클라이언트 연결 대기/수락
    get_sock: 생성된 서버 소켓 반환
    """
    def __init__(self, srv_port, listen_num):
        self.port = srv_port
        self.listen = listen_num
        self.mySock = None

    # socket 생성
    def create_sock(self):
        self.mySock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 소켓 닫아도 사용 가능하도록 설정
        self.mySock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.mySock.bind(("0.0.0.0", int(self.port)))
        self.mySock.listen(int(self.listen))

        return self.mySock

    # client 대기
    def client_ready(self):
        return self.mySock.accept()

    # socket 반환
    def get_sock(self):
        return self.mySock
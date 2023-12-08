import socket
from pipeline import Pipeline
import rich
import pyqt


def run_server(host='127.0.0.1', port=65432,pipeline=None):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if data
                if not data:
                    break
                conn.sendall(data)

if __name__ == "__main__":
    jsg_ppl = Pipeline()
    run_server()

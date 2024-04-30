
import socket
import random

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    address = ('localhost', 1025)     # family is deduced to be 'AF_INET'
    s.bind(address)
    s.listen()
    conn, addr = s.accept()
    print('Got client')

    while True:
        print('Waiting')
        msg = conn.recv(4096)
        if len(msg)>0:
            print('Got message')
        # set handPos msg
            print(msg)


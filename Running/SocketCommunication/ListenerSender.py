from queue import Queue
from threading import Thread
from multiprocessing.connection import Listener
import socket 
import pickle

# A thread that produces data
def cam_listenener(q):
    address = ('localhost', 6000)     # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=b'secret password')

    while True:
        conn = listener.accept()
        msg = conn.recv()
        print(msg)

        #TODO
        #convert from landmarks into pVec and Rvec
        
        q.put(msg)
        conn.close()

        
      
# A thread that consumes data
def send2robot(q):

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s = socket.socket()          
  
        #192.168.125.1
        address = ('localhost', 1025)
        try:
            s.connect(address) 
        except:
            while True:
                if (s.connect(address) != True ):
                    break
                print("trying to connect")
        while True:
            cmd = q.get()
            serialized_cmd = pickle.dumps(cmd)
            s.send(bytes(str(serialized_cmd), 'utf-8'))
            
          
        
    
# Create the shared queue and launch both threads
q = Queue()
t1 = Thread(target = cam_listenener, args =(q, ))
t2 = Thread(target = send2robot, args =(q, ))
t1.start()
t2.start()
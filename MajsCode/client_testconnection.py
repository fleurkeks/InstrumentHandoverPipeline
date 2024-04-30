import socket                
  
s = socket.socket()          
  
# Define the port on which you want to connect 
port = 1025
  
# connect to the server on local computer 
#s.connect(('127.0.0.1', port))

#address = ('192.168.125.1', port) #CHANGE THIS
address = ('127.0.0.1', port)

try:
  s.connect(address) 
except:
    while True:
        if (s.connect(address) != True ):
            break
        print("trying to connect")
cnt = 0

while True:
    s.send(bytes("1", 'utf-8'))
    cnt += 1
    #receive up to 4096 bytes
    data = s.recv(4096)
    if len(data)>0:
        res=str(data)
        print(res[2:len(res)-1])
    
    
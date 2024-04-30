from multiprocessing.connection import Client


address = ('localhost', 6000)
conn = Client(address, authkey=b'secret password')
msg=[[500,-100,500],[0,-0.707106781,0.707106781,0]],[[0,0,0],[1,0,0,0]]
conn.send(msg)

#can also send arbitrary objects:
#conn.send(['a', 2.5, None, int, sum])
#conn.close()
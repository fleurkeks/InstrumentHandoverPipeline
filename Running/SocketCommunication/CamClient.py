from multiprocessing.connection import Client
from queue import Queue

#innan vi skickar positionerna till robotern transformera vi de relativt kameran
#rob2hand=matrix multiplcation rob2cam@cam2hand

#logga tidigare positioner och smootha innan man skicker live
address = ('localhost', 6000)
conn = Client(address, authkey=b'secret password')
Hand_q = Queue()



#hämtar data fran cameran och hantera 
hand_array=[[22,0,0],[5,0,0,0]]


conn.send(msg)

#add a connection to the camera
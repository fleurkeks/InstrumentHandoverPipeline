from multiprocessing.connection import Client
from queue import Queue

address = ('localhost', 6000)
conn = Client(address, authkey=b'secret password')

#logga tidigare positioner och smootha innan man skicker live
Hand_q = Queue()

#TODO!!!!!!!!!!!!

#add a connection to the camera

#retrieve data from the camera, go from landmarks to tvec and quaternion
#smoothe with the latest 4 values received 

#innan vi skickar positionerna till robotern transformera vi de relativt kameran
#rob2hand=matrix multiplcation rob2cam@cam2hand

#conn.send(msg)


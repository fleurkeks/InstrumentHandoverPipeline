#takes our logfile in rvec, tvec and calculates the difference between marker and hands by converting to a matrix and doing matrix calc
#returns file in quaternions
#i.e this converts from marker and hand log to a log of directional vectors that describe the movement of the marker to the hand 
#(obs! in terms of hand2marker, because that is the format our robot uses)

import numpy as np
from transforms3d.euler import (mat2euler, euler2quat, euler2mat)
from transforms3d.quaternions import (mat2quat )
from math import radians
import cv2

def create_matrix(rvec, tvec):
    if isinstance(rvec, list):
        rvec = np.asarray(rvec)
        rvec = rvec.reshape(3, 1)
    if isinstance(tvec, list):
        tvec = np.asarray(tvec)
        tvec = tvec.reshape(3, 1)


    R, _ =cv2.Rodrigues(rvec)
   
    matrix = np.eye(4)

    matrix[0:3, 0:3] = R
    matrix[0, 3] = tvec[0,0]
    matrix[1, 3] = tvec[1,0]
    matrix[2, 3] = tvec[2,0]
    matrix[3, 3] = 1.0
    
    return matrix

def getvectors(matrix):
    tvec = matrix[0:3, -1]
    
    R = matrix[0:3, 0:3]
    rvec, _ = cv2.Rodrigues(R)
    return tvec, rvec

def getdata(filename):
    data = []
    with open(filename) as fn:
        lines = fn.readlines()
        for line in lines:
            
            datapoint = line.split()

            #retrieves data from any successfull reading
            if (len(datapoint)==15):
                frame = float(datapoint[0])
            
                rVecMarker = list(map(float, datapoint[2:5]))
                dVecMarker = list(map(float, datapoint[5:8]))

                rVecHand=list(map(float, datapoint[9:12]))
                dVecHand=list(map(float, datapoint[12:15]))
                
                #convert to matrix representation
                markerMatrix=create_matrix(rVecMarker,dVecMarker)
                handMatrix=create_matrix(rVecHand,dVecHand)

                #calculate the diff
                T=np.linalg.inv(handMatrix)@markerMatrix
                
                #convert to quaternion representation and append
                dvecT=getvectors(T)[0]
                rvecT=getvectors(T)[1]
        
                #converts rodrigues angles to a matrix, then the matrix to euler angles, then euler angles to quaternions
                R, _ =cv2.Rodrigues(rvecT)
                Euler=mat2euler(R)
                quatT=euler2quat(Euler[0],Euler[1],Euler[2])

                data.append((int(frame), (dvecT, quatT)))
    
    return data



def main():
    
    traj = getdata("Arching-log-smoothed.txt")
    with open('archtraj.txt', 'w') as f:
        for frame_nbr, item in traj:
            line = f"{frame_nbr} {' '.join(map(str, item[0]))} {' '.join(map(str, item[1]))}\n"
            f.write(line)
            
    



if __name__ == "__main__":
    main()
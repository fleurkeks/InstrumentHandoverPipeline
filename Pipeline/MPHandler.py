#using the palm landmarks we calculated the rotation relative to the directional vector (0,0,1). we also extract a pVec given 3 points
#includes functions for switching between quat, TMatrix and tvecs

import numpy as np
from pyquaternion import Quaternion
from transforms3d.euler import (quat2mat)
import cv2

#returns vector from p1 to p2 as an array
def PointToVec(p1, p2):
    vec=[(p2[0] - p1[0]), 
        (p2[1]- p1[1]), 
        (p2[2] - p1[2])]
    vec=np.array(vec)
    return vec

#returns the central point between three points as an array
#the landmarks outlining the palms are nbr 0, 5 and 17 the centroid of those landmarks is the hands pVec
def centroid(p1, p2, p3):
    centroid = [(p1[0] + p2[0] + p3[0]) / 3, 
            (p1[1] + p2[1] + p3[1]) / 3, 
            (p1[2] + p2[2] + p3[2]) / 3]
    
    centroid=np.array(centroid)

    return centroid

#returns a normalised vector indicating the direction the hand is currently facing
def dirVec(point):
    if isinstance(point, list):
        point = np.asarray(point)
    mag = np.linalg.norm(point)

    normalized_vector = point / mag
    return normalized_vector

def is_null_vector(vector):
    if isinstance(vector, list):
        vector = np.asarray(vector)
    return all(component == 0 for component in vector)

def are_antiparallel(vector1, vector2):
    return all(v1 == -v2 for v1, v2 in zip(vector1, vector2))

def angle_between_vectors(v1, v2):
  
    dot_product = np.dot(v1, v2)

    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)

    cos_angle = dot_product / (magnitude1 * magnitude2)

    angle_rad = abs(np.arccos(cos_angle))

    return angle_rad

#returns four quaternions describing the hands rotation
def land2quat(lm0, lm5, lm17):
    
    v1=PointToVec(lm0,lm5)
    v2=PointToVec(lm0,lm17)

    #normalen till handplan, lokal koordinater
    n=np.cross(v1,v2)

    #find rotation relative to the z axis pointing, pointing away from cam, find rotation in cameras coordinate system
    zAxis=np.array((0,0,1))

    #check 
    if(are_antiparallel(zAxis,n)):
        axis_of_rotation=(1,0,0)
        angle_of_rotation=np.pi
    
    #calculate the normal to the original dirVec of the hand and the current dirVec of the hand, that gives the axis of rotation
    else:
        axis_of_rotation=np.cross(zAxis,n)
         #check if the axis of rotation is 0, that would mean no rotation has happened --> return identity quaternion
        if(is_null_vector(axis_of_rotation)):
            return Quaternion(1, 0, 0, 0)
        #find the angle of rotation by finding the angle between original orientation and current orientation
        angle_of_rotation=angle_between_vectors(zAxis,n)
    

    #using this we can easly calculate the rotation in quaternions
    q = Quaternion(axis = axis_of_rotation, angle = angle_of_rotation)
    return q



#functions used in logger
def land2tvec(lm1, lm2, lm3):
    #sets tvec to the centroid of all landmarks
    tvec=centroid(lm1,lm2,lm3)

    #calculates rotation as quaternions
    quat=land2quat(lm1,lm2,lm3)

    #switches from quaternions to a rotation matrix
    rMat=quat2mat(quat)

    #calculates the rVec from the rotation matrix
    rvec, _ = cv2.Rodrigues(rMat)
    rvec=(rvec[0], rvec[1], rvec[2])

    return tvec, rvec




def main():
    # Example data
    lm0 = [0, 0, 0]
    lm5 = [-1, 0, 1]
    lm17 = [1, 0, 1]

    print(land2tvec(lm0,lm5,lm17))

    
if __name__ == "__main__":
    main()
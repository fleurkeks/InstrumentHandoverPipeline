#using the palm landmarks we calculated the rotation relative to the directional vector (0,0,1). we also extract a pVec given 3 points
#includes functions for switching between quat, TMatrix and tvecs

import numpy as np
from pyquaternion import Quaternion
from transforms3d.euler import (mat2euler, euler2quat, euler2mat, quat2mat)

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

    angle_rad = np.arccos(cos_angle)

    return angle_rad

#returns four quaternions describing the hands rotation
def hand_quaternion(lm0, lm5, lm17):
    
    #find the durrecnt dVec of the hand
    dVec=dirVec(centroid(lm0,lm5,lm17))
    print(dVec)

    #in the original position the normal vector of the hand will be (0,-1,0)
    #this is our initial position where the hand points straight upward
    ogRot=np.array((0,0,1))

    if(are_antiparallel(ogRot,dVec)):
        axis_of_rotation=(0,-1,0)
    
    #calculate the normal to the original dirVec of the hand and the current dirVec of the hand, that gives the axis of rotation
    else:
        axis_of_rotation=np.cross(ogRot,dVec)
         #check if the axis of rotation is 0, that would mean no rotation has happened --> return identity quaternion
        if(is_null_vector(axis_of_rotation)):
            return Quaternion(1, 0, 0, 0)
    print(axis_of_rotation)

   
    #find the angle of rotation by finding the angle between original orientation and current orientation
    angle_of_rotation=angle_between_vectors(ogRot,dVec)

    #using this we can easly calculate the rotation in quaternions
    q = Quaternion(axis = axis_of_rotation, angle = angle_of_rotation)
    return q

#goes from quaternion represenation to a translation matrix
def quatToMatrix(q):
    return quat2mat(q)


#takes Translation matrix and expresses it in form of tvec and rvec
def Matrix2vec(matrix):
    tvec = matrix[0:3, -1]
    R = matrix[0:3, 0:3]
    rvec = mat2euler(R)
    return tvec, rvec

def main():
    # Example data
    lm0 = [0, 0, 0]
    lm5 = [-1, 1, 0]
    lm17 = [1, 1, 0]

    # Test centroid function
    centroid_point = centroid(lm0, lm5, lm17)
    print("Centroid:", centroid_point)

    # Test rotation_quaternion function
    rotation = hand_quaternion(lm0, lm5, lm17)
    print("Rotation quaternion:", rotation)

    
if __name__ == "__main__":
    main()
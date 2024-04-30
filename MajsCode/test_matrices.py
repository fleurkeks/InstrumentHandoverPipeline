import sys
sys.path.insert(1, '/Users/maj/repos/tracker/yumi')
import numpy as np
import cv2
import yaml
import argparse

import cv2.aruco as aruco
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import math
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import (mat2euler, euler2quat, euler2mat)
from averageQuaternions import averageQuaternions

def create_matrix(rvec, tvec):
    if isinstance(rvec, list):
        rvec = np.asarray(rvec)
        rvec = rvec.reshape(3, 1)
    if isinstance(tvec, list):
        tvec = np.asarray(tvec)
        tvec = tvec.reshape(3, 1)

    rotation_matrix = euler2mat(rvec[0], rvec[1], rvec[2])
    matrix = np.eye(4)
    #for r in range(3):
    #    for c in range(3):
    #        matrix[r, c] = rotation_matrix[r, c]
    matrix[0:3, 0:3] = rotation_matrix
    matrix[0, 3] = tvec[0]
    matrix[1, 3] = tvec[1]
    matrix[2, 3] = tvec[2]
    matrix[3, 3] = 1.0
    return matrix

def getvectors(matrix):
    tvec = matrix[0:3, -1]
    print(tvec)
    R = matrix[0:3, 0:3]
    rvec = mat2euler(R)
    return tvec, rvec


def quatdist(q1, q2):
    pq1 = Quaternion(q1)
    pq2 = Quaternion(q2)
    conj_pq1 = pq1.inverse
    q12 = conj_pq1 * pq2
    L = math.hypot(q12[1], q12[2], q12[3])
    angle = 2 * math.atan2(L, q12[0])
    #assert angle == theta
    return angle

def avangle(list_rvec):
    Q = []
    for rvec in list_rvec:
        rot = R.from_rotvec(rvec)
        q = rot.as_quat() #scalar last
        q1 = [q[3], q[0], q[1], q[2]] #scalar first!
        Q.append(q1)
    Q = np.array(Q)
    avQ = averageQuaternions(Q)
    return avQ

#datalist is a list or array of tuples with (tvec, rvec)
def smoothing(number_of_points, datalist):
    smoothed_list = []
    padded_datalist = [datalist[0]] * (number_of_points -1) + datalist
    for i in range(len(padded_datalist) - number_of_points): #len(datalist)
        sublist = padded_datalist[i:i+number_of_points]
        tvecs = [t[0] for t in sublist]
        rvecs = [t[1] for t in sublist]
        
        mean = np.sum(tvecs, axis = 0)/number_of_points
        avQ = avangle(rvecs)
        smoothed_list.append((mean, avQ))
    return smoothed_list


def quatfromvector(vec):
    return euler2quat(vec)

def gentestdata(N = 1000):
    data = []
    delta = np.pi/N
    for i in range(N):
        noise = 0.5 * np.random.randn(3)
        point = np.array([40 * np.sin(i *delta), 60* np.sin(2 * i * delta), 30* np.cos(i * delta)]) + noise
        angle = np.array([0, 0, 0])
        data.append((point, angle))
    return data

def directionvector2quat(dirvec):
    if isinstance(dirvec, list):
        dirvec = np.asarray(dirvec)
   
    z = np.array([0, 0, 1])
    v = np.cross(z, dirvec)
    if np.linalg.norm(v) < 0.0000001:
        return Quaternion(1, 0, 0, 0)
    theta = np.arccos(np.dot(z, v) /np.linalg.norm(v))
    q = Quaternion(axis = v, angle = theta)
    return q

# T_hand is the homogenous matrix representing the transformation from the camera to the hand
# T_aruco is the homogenous matrix for the tranformation from the camera to the aruco marker
# The function returns the homogenous matrix T_traj for the transformation from the hand to the aruco marker
def diffbetweenmatrices(T_hand, T_aruco):
    #T_hand * T_traj = T_aruco => T_traj = inv(T_hand) * T_aruco
    T_traj = np.linalg.inv(T_hand) @ T_aruco # the @ sign is inline code for matrix multiplication
    return T_traj

# p_hand is the tuple (tvec, rvec) for the hand

def diffbetweenpoints(p_hand, p_aruco):
    hand_tvec, hand_rvec = p_hand 
    aruco_tvec, aruco_rvec = p_aruco
    T_hand = create_matrix(hand_rvec, hand_tvec) 
    T_aruco = create_matrix(aruco_rvec, aruco_tvec)
    T_traj = diffbetweenmatrices(T_hand, T_aruco)
    return getvectors(T_traj)
    

if __name__ == "__main__":
    test1()
    exit()
    n90 = np.pi/2
    A = create_matrix([n90, 0.0, 0.0], [1.0, 0.0, 0.0])
    print('A')
    print(A)
    tvec, rvec = getvectors(A)
    print('tvec', tvec)
    print('rvec', rvec)
    T = create_matrix([-n90, 0.0, 0.0], [0.0, 1.0, 0.0])
    B = np.matmul(A, T)
    print('B')
    print(B)
    
    #Testing the smoothing
    testdata = gentestdata()
    smoothed_data = smoothing(5, testdata)
    ax = plt.figure().add_subplot(projection='3d')
    X, Y , Z = [], [], []
    print(smoothed_data[0:10])
    X = [t[0][0] for t in smoothed_data]
    Y = [t[0][1] for t in smoothed_data]
    Z = [t[0][2] for t in smoothed_data]
    
    ax.plot(X, Y, Z)
    plt.show()



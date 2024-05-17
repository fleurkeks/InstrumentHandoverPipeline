import sys         
 
# appending the directory of mod.py 
# in the sys.path list
sys.path.append('C:/Users/Microcrew/Documents/Examensarbete/LeaRepo/InstrumentHandoverPipeline/Training')

import numpy as np
import cv2
import cv2.aruco as aruco
from sklearn.preprocessing import normalize
import mediapipe as mp #pip install mediapipe opencv-python
import sksurgerycore.transforms.matrix as matrix
import sksurgerycalibration.algorithms.pivot as pivot
from transforms3d.affines import compose
from transforms3d.euler import (euler2mat, mat2euler, euler2quat, quat2euler,
                     euler2axangle, axangle2euler, EulerFuncs)
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial.transform import Rotation as R
from transforms3d.affines import compose
import math
import warnings
from cad_model import model, model_corners
import camera_data as camera_data
import MPHandler

def drawcenter(img, imgpts): #Maj code
    corner = tuple(map(int, imgpts[0].ravel()))
    img = cv2.line(img, corner, tuple(map(int, imgpts[1].ravel())), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(map(int, imgpts[2].ravel())), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(map(int, imgpts[3].ravel())), (0,0,255), 5)
    return img

def process_frame_aruco(frame, logfile, width, height):

    rvec = np.array([0., 0., 0.])
    tvec = np.array([0., 0., 0.])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    marked = frame.copy()#aruco.drawDetectedMarkers(color_image.copy(), corners)#, ids)#Fails without the '.copy()', the 'drawDetectedMarkers' draw contours of the tags in the image
    
    marked = aruco.drawDetectedMarkers(marked, corners, ids, (0,255,0))
    

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    points3d = []
    crns = []

    if ids is None:
        return 0
    n = 0
    for index, id in enumerate(ids):
        
        if id[0] in model_corners:
            n += 1
            for corner in model_corners[id[0]]:
                points3d.append(corner)


            for corner_list in corners[index]:
                for corner in corner_list:
                    crns.append(corner)

                #logfile.write('{} {} {} {} {} {} '.format(frameno, id[0], corner_list[0], corner_list[1], corner_list[2], corner_list[3]))
        
        
    if n > 0:
        pts3d = np.array(points3d)
        corners = np.array(crns).reshape(4 * n , 2)
        assert(max(pts3d.shape) == 4 * n)
        assert(max(corners.shape) == 4* n)

        ret = cv2.solvePnP(pts3d, corners, camera_matrix, dist_coeffs, rvec, tvec)
        proj, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
        #positions.append((rvec, tvec))
        #logfile.write('{} {} {} {} {} {} {} {}\n'.format(frameno, 10, rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2]))
        marked = drawcenter(marked, proj)

        cv2.imshow("marked", marked)
        
        #ret = cv2.solvePnP(pts3d, corners, camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess = True)
        #proj, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
        #positions.append((rvec, tvec))
        logfile.write('{} {} {} {} {} {} {} '.format("aruco", rvec[0], rvec[1], rvec[2], tvec[0]/1000, tvec[1]/1000, tvec[2]/1000))
        #img = drawcenter(marked, proj)
        
        #cv2.imshow('frame',img)
        #out.write(img)
        # frame = cv.flip(frame, 0)
        # write the flipped frame
        #out.write(frame)
        
        key = cv2.waitKey(1)

        # Release everything if job is finished
        #cap.release()
        #cv2.destroyAllWindows()

# Filtering
kernel= np.ones((3,3),np.uint8)

#ArUco marker stuff
markerlength = 9
camera_matrix, dist_coeffs = camera_data.sony_data()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)#Aruco marker size 4x4 (+ border)
parameters =  aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
axis = np.float32([[0, 0, 0],[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)

#Logfile name
LOG = './handover1.txt'

cap= cv2.VideoCapture("./calibration_pictures_1cam/CameraLeft6.avi") 

with open(LOG, 'w+') as logfile:
    while True:
        res, frame = cap.read()
        height = frame.shape[0]
        width = frame.shape[1]

        frameno = cap.get(cv2.CAP_PROP_POS_FRAMES)
        logfile.write('{} '.format(frameno))

        process_frame_aruco(frame, logfile, width, height)

        logfile.write('\n') 



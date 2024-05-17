import cv2
import numpy as np
import logread
import camera_data
import time

camera_matrixR, camera_matrixL, dist_coeffR, dist_coeffL = camera_data.sony_data_2cam()

def draw(img, corners, imgpts):
    corner = tuple(map(int, corners.ravel()))
    img = cv2.line(img, corner, tuple(map(int, imgpts[0].ravel())), (0,0,255), 5)
    img = cv2.line(img, corner, tuple(map(int, imgpts[1].ravel())), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(map(int, imgpts[2].ravel())), (255,0,0), 5) #BGR
    return img


def project(rvector, tvector, img, show = True):
    axis = np.float32([[0, 0, 0], [30,0,0], [0,30,0], [0,0,30]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, rvector, tvector, camera_matrixR, dist_coeffR)
    print('Projectet base', imgpts)
    img = draw(img, imgpts[0], imgpts[1:])
    return img

#num_frame = 90

speed = 10
log = logread.processLogFile("RotationTopDown-log.txt")

cap = cv2.VideoCapture("handover_videos/RotationTopDownLeft.avi")

#cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame)

#frame_coords = log[num_frame-1]

while True:

    time.sleep(1/speed)
    
    res, img = cap.read()

    num_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    frame_coords = log[int(num_frame)]

    if(frame_coords["hand_rvec0"] != None):
        rvec_hand = np.array([frame_coords["hand_rvec0"], frame_coords["hand_rvec1"], frame_coords["hand_rvec2"]])
        tvec_hand = np.array([frame_coords["hand_tvec0"], frame_coords["hand_tvec1"], frame_coords["hand_tvec2"]])
        img = project(rvec_hand, tvec_hand, img)

    if(frame_coords["aruco_rvec0"] != None):
        rvec_aruco = np.array([frame_coords["aruco_rvec0"], frame_coords["aruco_rvec1"], frame_coords["aruco_rvec2"]])
        tvec_aruco = np.array([frame_coords["aruco_tvec0"], frame_coords["aruco_tvec1"], frame_coords["aruco_tvec2"]])
        img = project(rvec_aruco, tvec_aruco, img)
        

    #rvec_aruco = np.array([0.0, 0.0, 0.0])
    #tvec_aruco = np.array([0.0, 0.0, 1000.0])


    half = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
    cv2.imshow('img',half)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
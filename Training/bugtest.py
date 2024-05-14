import cv2
import numpy as np
import logread
import camera_data

camera_matrix, dist_coeffs = camera_data.sony_data()

def draw(img, corners, imgpts):
    corner = tuple(map(int, corners.ravel()))
    img = cv2.line(img, corner, tuple(map(int, imgpts[0].ravel())), (0,0,255), 5)
    img = cv2.line(img, corner, tuple(map(int, imgpts[1].ravel())), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(map(int, imgpts[2].ravel())), (255,0,0), 5) #BGR
    return img


def project(rvector, tvector, img, show = True):
    axis = np.float32([[0, 0, 0], [30,0,0], [0,30,0], [0,0,30]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, rvector, tvector, camera_matrix, dist_coeffs)
    print('Projectet base', imgpts)
    img = draw(img, imgpts[0], imgpts[1:])
    return img

num_frame = 22

log = logread.processLogFile("handover1.txt")

cap = cv2.VideoCapture("training_videos/CameraLeft1.avi")

frame_coords = log[num_frame-1]

cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
res, img = cap.read()

#rvec_hand = np.array([frame_coords["hand_rvec0"], frame_coords["hand_rvec1"], frame_coords["hand_rvec2"]])
#tvec_hand = np.array([frame_coords["hand_tvec0"], frame_coords["hand_tvec1"], frame_coords["hand_tvec2"]])

rvec_hand = np.array([frame_coords["hand_rvec0"], frame_coords["hand_rvec1"], frame_coords["hand_rvec2"]])
tvec_hand = np.array([frame_coords["hand_tvec2"], frame_coords["hand_tvec1"], frame_coords["hand_tvec0"]])


#rvec_aruco = np.array([frame_coords["aruco_rvec0"], frame_coords["aruco_rvec1"], frame_coords["aruco_rvec2"]])
#tvec_aruco = np.array([frame_coords["aruco_tvec0"], frame_coords["aruco_tvec1"], frame_coords["aruco_tvec2"]])

rvec_aruco = np.array([0.0, 0.0, 0.0])
tvec_aruco = np.array([0.0, 0.0, 100.0])


#print(rvec_aruco, tvec_aruco)
img = project(rvec_aruco, tvec_aruco, img)


half = cv2.resize(img, (0, 0), fx = 0.25, fy = 0.25)
cv2.imshow('img',half)
cv2.waitKey(0)
import sys   
import os    
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PIPELINE_PATH = os.getenv("PROJECT_PATH") + "/Pipeline"
LOGGING_PATH = PIPELINE_PATH + "/logging"

 
# appending the directory of mod.py 
# in the sys.path list
sys.path.append(PIPELINE_PATH)

# Package importation
import numpy as np
import cv2
import cv2.aruco as aruco
#import sksurgerycore.transforms.matrix as matrix
from transforms3d.affines import compose
from transforms3d.euler import (euler2mat, mat2euler, euler2quat, quat2euler,
                     euler2axangle, axangle2euler, EulerFuncs)
from transforms3d.affines import compose
from cad_model import model, model_corners
import camera_data as camera_data
import MPHandler
import mediapipe as mp


# Filtering
kernel= np.ones((3,3),np.uint8)

#ArUco marker stuff
markerlength = 9
camera_matrixR, camera_matrixL, dist_coeffR, dist_coeffL = camera_data.sony_data_2cam()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)#Aruco marker size 4x4 (+ border)
parameters =  aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
axis = np.float32([[0, 0, 0],[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)

#Video names
video_name = "Simple"
FILE_LEFT = PIPELINE_PATH + '/logging/handover_videos/'+video_name+'Left.avi'
FILE_RIGHT = PIPELINE_PATH + '/logging/handover_videos/'+video_name+'Right.avi'

#integer means webcam of that name (live feed), change to FILE_LEFT and FILE_RIGHT to use the files specified above. (1,2 for live cameras)
LEFT_CAMERA = FILE_LEFT 
RIGHT_CAMERA = FILE_RIGHT

#Number of calibration pictures
NUM_CAL_PICS = 34

#Which landmarks on the hand do we want to use
LANDMARK_A = 0
LANDMARK_B = 5
LANDMARK_C = 17

BASELINE = 75 #Distance between cameras in mm

LOG = PIPELINE_PATH + '/logging/'+video_name+'-log.txt'

def calibrate_cameras(folderLocation):

    print("Calibrating Cameras...")
    # Arrays to store object points and image points from all images
    objpoints= []   # 3d points in real world space
    imgpointsR= []   # 2d points in image plane
    imgpointsL= []

    CHECKERBOARD = (6,9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #size of each square 
    square_size = 24.0

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

    for i in range(0,NUM_CAL_PICS):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
        t= str(i)
        ChessImaR= cv2.imread(folderLocation + '/chessboard-R'+t+'.png',0)    # Right side
        ChessImaL= cv2.imread(folderLocation + '/chessboard-L'+t+'.png',0)    # Left side
        retR, cornersR = cv2.findChessboardCorners(ChessImaR,CHECKERBOARD,None)  # Define the number of chees corners we are looking for
        retL, cornersL = cv2.findChessboardCorners(ChessImaL,CHECKERBOARD,None)  # Left side
        if (True == retR) & (True == retL):
            objpoints.append(objp)
            cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)
            cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
            imgpointsR.append(cornersR)
            imgpointsL.append(cornersL)

    # Determine the new values for different parameters
    #   Right Side
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,imgpointsR,ChessImaR.shape[::-1],None,None)

    #   Left Side
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,imgpointsL,ChessImaL.shape[::-1],None,None)

    camera_matrixR= mtxR.tolist()
    camera_matrixL= mtxL.tolist()
    dist_coeffR = distR.tolist()
    dist_coeffL = distL.tolist()
    data = {"camera_matrixR": camera_matrixR, "dist_coeffR": dist_coeffR, "camera_matrixL": camera_matrixL, "dist_coeffL": dist_coeffL}
    fname = PIPELINE_PATH + "/logging/data_sony_2cam.yaml"
    
    import yaml
    with open(fname, "w") as f:
        yaml.dump(data, f)
    print('Calibration Complete!')

def process_frame_gmph(imageR, imageL, handsR, handsL, mp_drawing, mp_hands, camera_matrixR, camera_matrixL, baseline, logfile):
    #Handle right image first
    imageR = cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB)
    # Set flag
    imageR.flags.writeable = False
    # Detections
    resultsR = handsR.process(imageR)
    # Set flag to true
    imageR.flags.writeable = True
    # RGB 2 BGR
    imageR = cv2.cvtColor(imageR, cv2.COLOR_RGB2BGR)
    # Detections
    #print(results)
    

    # Rendering results
    if resultsR.multi_hand_landmarks:
        for num, hand in enumerate(resultsR.multi_hand_landmarks):
            mp_drawing.draw_landmarks(imageR, hand, mp_hands.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(22, 22, 250), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(20, 250, 255), thickness=2, circle_radius=2),
                                        )
            
    #Then handle left image
    imageL = cv2.cvtColor(imageL, cv2.COLOR_BGR2RGB)
    # Set flag
    imageL.flags.writeable = False
    # Detections
    resultsL = handsL.process(imageL)
    # Set flag to true
    imageL.flags.writeable = True
    # RGB 2 BGR
    imageL = cv2.cvtColor(imageL, cv2.COLOR_RGB2BGR)
    # Detections
    #print(results)
    

    # Rendering results
    if resultsL.multi_hand_landmarks:
        for num, hand in enumerate(resultsL.multi_hand_landmarks):
            mp_drawing.draw_landmarks(imageL, hand, mp_hands.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(22, 22, 250), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(20, 250, 255), thickness=2, circle_radius=2),
                                        )
            
    
    image_height, image_width, _ = imageR.shape        
    disp_a, disp_b, disp_c = get_disparity_of_landmarks(resultsR, resultsL, image_height, image_width)

    #calculate depth to landmarks if disparity extraction was successful
    if disp_a != None and disp_b != None and disp_c != None:
        bXf = (baseline * camera_matrixL[0][0])
        depth_a, depth_b, depth_c = bXf/disp_a, bXf/disp_b, bXf/disp_c

        #print(depth_a, depth_b, depth_c)

        #Now we want to use the depth, as well as x,y in pixels to get the real world lattitude and longitude
        a_3d, b_3d, c_3d = get_3d_coordinates_of_landmarks(resultsL, depth_a, depth_b, depth_c, baseline, camera_matrixL, image_height, image_width)

        tvec, rvec = MPHandler.land2tvec(a_3d, b_3d, c_3d)

        logfile.write('{} {} {} {} {} {} {} '.format("hand", rvec[0][0], rvec[1][0], rvec[2][0], tvec[0]+0.075, tvec[1], tvec[2]))



    return imageR, imageL

def get_disparity_of_landmarks(resultsR, resultsL, image_height, image_width):
    
    if resultsR.multi_hand_landmarks and resultsL.multi_hand_landmarks:
        hand_landmarks = resultsR.multi_hand_landmarks[0]
        
        landmark_a = hand_landmarks.landmark[LANDMARK_A]
        landmark_b = hand_landmarks.landmark[LANDMARK_B]
        landmark_c = hand_landmarks.landmark[LANDMARK_C]

        #Since Mediapipe gives x coordinates on a scale of 0 to 1, we need to multiply the coordinate value with the width and height of the image to get the pixel coordinate.
        x_a_pixelR = int(landmark_a.x * image_width)

        x_b_pixelR = int(landmark_b.x * image_width)

        x_c_pixelR = int(landmark_c.x * image_width)

        hand_landmarks = resultsL.multi_hand_landmarks[0]

        landmark_a = hand_landmarks.landmark[LANDMARK_A]
        landmark_b = hand_landmarks.landmark[LANDMARK_B]
        landmark_c = hand_landmarks.landmark[LANDMARK_C]

        x_a_pixelL = int(landmark_a.x * image_width)

        x_b_pixelL = int(landmark_b.x * image_width)

        x_c_pixelL = int(landmark_c.x * image_width)

        return x_a_pixelL - x_a_pixelR, x_b_pixelL - x_b_pixelR, x_c_pixelL - x_c_pixelR
    return None, None, None

def get_3d_coordinates_of_landmarks(resultsL, depth_a, depth_b, depth_c, baseline, camera_matrixL, image_height, image_width):
    

    if resultsL.multi_hand_landmarks:
    
        hand_landmarks = resultsL.multi_hand_landmarks[0]
        
        landmark_a = hand_landmarks.landmark[LANDMARK_A]
        landmark_b = hand_landmarks.landmark[LANDMARK_B]
        landmark_c = hand_landmarks.landmark[LANDMARK_C]

        x_a_pixel, y_a_pixel = int(landmark_a.x * image_width), int(landmark_a.y * image_height)

        x_b_pixel, y_b_pixel = int(landmark_b.x * image_width), int(landmark_b.y * image_height)

        x_c_pixel, y_c_pixel = int(landmark_c.x * image_width), int(landmark_c.y * image_height)

    #Get landmarks distance in pixels from image center (fc = from center)
    x_a_fc, x_b_fc, x_c_fc = x_a_pixel - (image_width/2), x_b_pixel - (image_width/2), x_c_pixel - (image_width/2)
    y_a_fc, y_b_fc, y_c_fc = y_a_pixel - (image_height/2), y_b_pixel - (image_height/2), y_c_pixel - (image_height/2)

    fx = camera_matrixL[0][0]
    fy = camera_matrixL[1][1]

    a_3d = [0,0,depth_a]
    b_3d = [0,0,depth_b]
    c_3d = [0,0,depth_c]
    
    a_3d[0] = depth_a * x_a_fc / fx
    a_3d[1] = depth_a * y_a_fc / fy

    b_3d[0] = depth_b * x_b_fc / fx
    b_3d[1] = depth_b * y_b_fc / fy

    c_3d[0] = depth_c * x_c_fc / fx
    c_3d[1] = depth_c * y_c_fc / fy

    return a_3d, b_3d, c_3d

def process_frame_aruco(frame, logfile):

    rvec = np.array([0., 0., 0.])
    tvec = np.array([0., 0., 0.])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    marked = frame.copy()

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
        
        ret = cv2.solvePnP(pts3d, corners, camera_matrixR, dist_coeffR, rvec, tvec)
        proj, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrixR, dist_coeffR)
        marked = drawcenter(marked, proj)
        marked_half = cv2.resize(marked, (0, 0), fx = 0.25, fy = 0.25)
        cv2.imshow("marked", marked_half)

        logfile.write('{} {} {} {} {} {} {} '.format("aruco", rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2]))

        key = cv2.waitKey(1)

def drawcenter(img, imgpts): #Maj code
    corner = tuple(map(int, imgpts[0].ravel()))
    img = cv2.line(img, corner, tuple(map(int, imgpts[1].ravel())), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(map(int, imgpts[2].ravel())), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(map(int, imgpts[3].ravel())), (0,0,255), 5)
    return img


def main():
    #calibrate_cameras(PIPELINE_PATH + "/logging/calibration_pictures")    

    camera_matrixR, camera_matrixL, dist_coeffR, dist_coeffL = camera_data.sony_data_2cam()

    camL= cv2.VideoCapture(LEFT_CAMERA) 
    camR= cv2.VideoCapture(RIGHT_CAMERA)

    retR, retL = True, True

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    with open(LOG, 'w+') as logfile:
        with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as handsR: 
            with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as handsL: 
                while retR and retL:
                    
                    retR, frameR= camR.read()
                    retL, frameL= camL.read()
                    frameno = camL.get(cv2.CAP_PROP_POS_FRAMES)
                    logfile.write('{} '.format(frameno))

                    process_frame_aruco(frameL, logfile)
                    gmphR, gmphL = process_frame_gmph(frameR, frameL, handsR, handsL, mp_drawing, mp_hands, camera_matrixR, camera_matrixL, BASELINE, logfile)
                    
                    logfile.write('\n')    

                    gmphR_half = cv2.resize(gmphR, (0, 0), fx = 0.25, fy = 0.25)
                    gmphL_half = cv2.resize(gmphL, (0, 0), fx = 0.25, fy = 0.25)

                    cv2.imshow("GMPH R",gmphR_half)
                    cv2.imshow("GMPH L",gmphL_half)

                    if cv2.waitKey(1) & 0xFF == ord(' '):
                        break

    

    
if __name__ == "__main__":
    main()




                





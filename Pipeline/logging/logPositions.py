#here we log the positions of recorded trainings-data
#we output a file containing, for each frame, the position and rotation or the aruco markers and the hand as tvecs and rvecs


# Package importation
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
from Training.logging.cad_model import model, model_corners
import Training.logging.camera_data as camera_data
import MPHandler


# Filtering
kernel= np.ones((3,3),np.uint8)

#ArUco marker stuff
markerlength = 10
camera_matrix, dist_coeffs = camera_data.sony_data()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)#Aruco marker size 4x4 (+ border)
parameters =  aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
axis = np.float32([[0, 0, 0],[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)

#Logfile name
LOG = './handover1.txt'
#Video names
FILE_LEFT = './training_videos/CameraLeft5Handover1.avi'
FILE_RIGHT = './training_videos/CameraRight5Handover1.avi'

#integer means webcam of that name (live feed), change to FILE_LEFT and FILE_RIGHT to use the files specified above. (1,2 for live cameras)
LEFT_CAMERA = FILE_LEFT 
RIGHT_CAMERA = FILE_RIGHT

NUM_CAL_PICS = 50

#Which landmarks on the hand do we want to use
LANDMARK_A = 0
LANDMARK_B = 5
LANDMARK_C = 17

def drawcenter(img, imgpts): #Maj code
    corner = tuple(map(int, imgpts[0].ravel()))
    img = cv2.line(img, corner, tuple(map(int, imgpts[1].ravel())), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(map(int, imgpts[2].ravel())), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(map(int, imgpts[3].ravel())), (0,0,255), 5)
    return img

def coords_mouse_disp(event,x,y,flags,param): #MIT code, not used in the final pipeline
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9
        Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        Distance= np.around(Distance*0.01,decimals=2)
        print('Distance: '+ str(Distance)+' m')

def distance_of_coodinate(coord_tuple, image_height, image_width):
    #Dont really know exactly how this works, but it spits out the distance as an average of the given pixel and the 8 pixels surrounding it.
    average=0
    num_values=0
    for u in range (-1,2):
        for v in range (-1,2):
            if coord_tuple[1]+u >= image_height or coord_tuple[0]+v >= image_width:
                continue
            average += disp[coord_tuple[1]+u,coord_tuple[0]+v]
            num_values += 1
    if num_values == 0:
        return 0
    average=average/num_values
    distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
    distance= np.around(distance*0.01,decimals=2)
    return distance

def get_coordinates_for_plane(hand_landmarks, image_height, image_width, points3D):
    #Save Landmarks
    landmark_a = hand_landmarks.landmark[LANDMARK_A]
    landmark_b = hand_landmarks.landmark[LANDMARK_B]
    landmark_c = hand_landmarks.landmark[LANDMARK_C]


    #CHECK HEREEEEE
    #Since Mediapipe gives x,y coordinates on a scale of 0 to 1, we need to multiply the coordinate value with the width and height of the image to get the pixel coordinate.
    x_a_pixel, y_a_pixel = int((1 - landmark_a.x) * image_width), int(landmark_a.y * image_height)## The coordinate would for some reason be mirrored when translating to the 3d point view, so I just use the 1 - x to flip the coordinate

    x_b_pixel, y_b_pixel = int((1 - landmark_b.x) * image_width), int(landmark_b.y * image_height)

    x_c_pixel, y_c_pixel = int((1 - landmark_c.x) * image_width), int(landmark_c.y * image_height)

    #print(points3D.shape, y_a_pixel, x_a_pixel)

     #Sometimes GMPH will predict a hand landmark is outside the frame, so just want to handle those edge cases
    if x_a_pixel > 1279:
        y_a_pixel = 1279
    if x_b_pixel > 1279:
        y_b_pixel = 1279
    if x_c_pixel > 1279: 
        y_c_pixel = 1279
    if y_a_pixel > 719:
        y_a_pixel = 719
    if y_b_pixel > 719:
        y_b_pixel = 719
    if y_c_pixel > 719: 
        y_c_pixel = 719
    

    x_a, y_a, z_a = points3D[y_a_pixel, x_a_pixel, 0], points3D[y_a_pixel, x_a_pixel, 1], points3D[y_a_pixel, x_a_pixel, 2]  
    x_b, y_b, z_b = points3D[y_b_pixel, x_b_pixel, 0], points3D[y_b_pixel, x_b_pixel, 1], points3D[y_b_pixel, x_b_pixel, 2]
    x_c, y_c, z_c = points3D[y_c_pixel, x_c_pixel, 0], points3D[y_c_pixel, x_c_pixel, 1], points3D[y_c_pixel, x_c_pixel, 2]
    
    return_dict = {
        LANDMARK_A : (x_a, y_a, z_a),
        LANDMARK_B : (x_b, y_b, z_b),
        LANDMARK_C : (x_c, y_c, z_c)
    } 

    return return_dict

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
        
        ret = cv2.solvePnP(pts3d, corners, camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess = True)
        proj, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
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

def process_frame_gmph(image, logfile, points3D):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Flip on horizontal (it is possible this could be removed since I flip it back later, but I havent tested if its vital for MP so I've left it for now)
    image = cv2.flip(image, 1)
    # Set flag
    image.flags.writeable = False
    # Detections
    results = hands.process(image)
    # Set flag to true
    image.flags.writeable = True
    # RGB 2 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Detections
    #print(results)
    # Rendering results
    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(22, 22, 250), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(20, 250, 255), thickness=2, circle_radius=2),
                                        )
    #Reverse flip
    image = cv2.flip(image, 1)


    """
    return_dict = {
        LANDMARK_A : (x_a, y_a, z_a),
        LANDMARK_B : (x_b, y_b, z_b),
        LANDMARK_C : (x_c, y_c, z_c)
    }
    """
    image_height, image_width, _ = image.shape
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            cfp = get_coordinates_for_plane(hand_landmarks, image_height, image_width, points3D) 
            lma = [cfp[LANDMARK_A][0], cfp[LANDMARK_A][1], cfp[LANDMARK_A][2]]
            lmb = [cfp[LANDMARK_B][0], cfp[LANDMARK_B][1], cfp[LANDMARK_B][2]]
            lmc = [cfp[LANDMARK_C][0], cfp[LANDMARK_C][1], cfp[LANDMARK_C][2]]
            quaternion = MPHandler.hand_quaternion(lma, lmb, lmc)
            qMatrix = MPHandler.quatToMatrix(quaternion)
            tvec, rvec = MPHandler.Matrix2vec(qMatrix)
            logfile.write('{} {} {} {} {} {} {} '.format("hand", rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2]))
            #logfile.write('{} {} {} {} {} {} {} {} {} {} {} {} '.format( LANDMARK_A, cfp[0][0], cfp[0][1], cfp[0][2], LANDMARK_B, cfp[1][0], cfp[1][1], cfp[1][3], LANDMARK_C, cfp[2][0], cfp[2][1], cfp[2][3]))
    return image

        

#*************************************************
#***** Parameters for Distortion Calibration *****
#*************************************************

# Termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for i in range(0,NUM_CAL_PICS):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
    t= str(i)
    ChessImaR= cv2.imread('./calibration_pictures/chessboard-R'+t+'.png',0)    # Right side
    ChessImaL= cv2.imread('./calibration_pictures/chessboard-L'+t+'.png',0)    # Left side
    retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                               (9,6),None)  # Define the number of chees corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                               (9,6),None)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)
        cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

# Determine the new values for different parameters
#   Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        ChessImaR.shape[::-1],None,None)
hR,wR= ChessImaR.shape[:2]
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                   (wR,hR),1,(wR,hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        ChessImaL.shape[::-1],None,None)
hL,wL= ChessImaL.shape[:2]
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

print('Cameras Ready to use')

#********************************************
#***** Calibrate the Cameras for Stereo *****
#********************************************

# StereoCalibrate function
#flags = 0
#flags |= cv2.CALIB_FIX_INTRINSIC
#flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
#flags |= cv2.CALIB_USE_INTRINSIC_GUESS
#flags |= cv2.CALIB_FIX_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_ASPECT_RATIO
#flags |= cv2.CALIB_ZERO_TANGENT_DIST
#flags |= cv2.CALIB_RATIONAL_MODEL
#flags |= cv2.CALIB_SAME_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_K3
#flags |= cv2.CALIB_FIX_K4
#flags |= cv2.CALIB_FIX_K5
retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          ChessImaR.shape[::-1],
                                                          criteria = criteria_stereo,
                                                          flags = cv2.CALIB_FIX_INTRINSIC)

# StereoRectify function
rectify_scale= 0 # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 ChessImaR.shape[::-1], R, T,
                                                 rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)
#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************

# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)

# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

#*************************************
#***** Starting the StereoVision *****
#*************************************

# Call the two cameras
camR= cv2.VideoCapture(LEFT_CAMERA) 
camL= cv2.VideoCapture(RIGHT_CAMERA)

## INITIATE MEDIAPIPE

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#Variables for aruco tracking
frame_width = int(camL.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
frame_height = int(camL.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
frametot = int(camL.get(cv2.CAP_PROP_FRAME_COUNT)+ 0.5)

with open(LOG, 'w+') as logfile:
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
        while True:
            # Start Reading Camera images
            retR, frameR= camR.read()
            retL, frameL= camL.read()
            # Rectify the images on rotation and alignement
            Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)  # Rectify the image using the kalibration parameters founds during the initialisation
            Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)

        ##    # Draw Red lines
        ##    for line in range(0, int(Right_nice.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
        ##        Left_nice[line*20,:]= (0,0,255)
        ##        Right_nice[line*20,:]= (0,0,255)
        ##
        ##    for line in range(0, int(frameR.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
        ##        frameL[line*20,:]= (0,255,0)
        ##        frameR[line*20,:]= (0,255,0)    
                
            # Show the Undistorted images
            #cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
            #cv2.imshow('Normal', np.hstack([frameL, frameR]))

            # Convert from color(BGR) to gray
            grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
            grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)

            # Compute the 2 images for the Depth_image
            disp= stereo.compute(grayL,grayR)#.astype(np.float32)/ 16
            points3D = cv2.reprojectImageTo3D(disp,Q) #this converts the disparity to a matrix of 3d-points
            dispL= disp
            dispR= stereoR.compute(grayR,grayL)
            dispL= np.int16(dispL)
            dispR= np.int16(dispR)

            # Using the WLS filter
            filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
            filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
            filteredImg = np.uint8(filteredImg)
            #cv2.imshow('Disparity Map', filteredImg)
            disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect
        ##    # Resize the image for faster executions
        ##    dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

            # Filtering the Results with a closing filter
            closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

            # Colors map
            dispc= (closing-closing.min())*255
            dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
            disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
            filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN) 

            #Start each row of the log with the frame number
            frameno = camL.get(cv2.CAP_PROP_POS_FRAMES)
            logfile.write('{} '.format(frameno))

            #LOG ARUCO
            process_frame_aruco(frameL, logfile, frame_width, frame_height)

            #LOG MEDIAPIPE
            image = process_frame_gmph(frameL, logfile, points3D)  

            logfile.write('\n')      

            # Show the result for the Depth_image
            #cv2.imshow('Disparity', disp)
            #cv2.imshow('Closing',closing)
            #cv2.imshow('Color Depth',disp_Color)
            cv2.imshow('Filtered Color Depth',filt_Color)
            
            cv2.imshow("camright", frameR)
            cv2.imshow("camleft with handtracking", image)

            # Mouse click
            cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,filt_Color)
            
            # End the Programme
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break
        
# Save excel
##wb.save("data4.xlsx")

# Release the Cameras
camR.release()
camL.release()
cv2.destroyAllWindows()

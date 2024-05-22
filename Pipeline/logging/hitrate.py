import sys 
import os    
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PIPELINE_PATH = os.getenv("PROJECT_PATH") + "/Pipeline"
sys.path.append(PIPELINE_PATH + '/Plotting')

import logread

allLogs = []
allLogs.append(logread.processLogFile(PIPELINE_PATH + "/logging/YOURLOG-log.txt")) #Append all your logfiles that you wish to get the hitrate of

both_tot = 0
hands_tot = 0
aruco_tot = 0
frames_tot = 0
for log in allLogs:
    both = 0
    hands = 0
    aruco = 0
    for frame in log:
        if(frame["hand_rvec0"] != None and frame["aruco_rvec0"] != None):
            both += 1
            continue
        elif(frame["hand_rvec0"] != None):
            hands += 1
        elif(frame["aruco_rvec0"] != None):
            aruco += 1
    print("Both: ",both/len(log),"Hands:", hands/len(log), "Aruco: ", aruco/len(log))
    both_tot += both
    hands_tot += hands
    aruco_tot += aruco
    frames_tot += len(log)

print("Both tot: ", both_tot/frames_tot,"Only Hands Tot: ", hands_tot/frames_tot, "Only Aruco Tot: ", aruco_tot/frames_tot)
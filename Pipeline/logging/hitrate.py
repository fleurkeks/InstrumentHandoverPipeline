import sys 
sys.path.append('C:/Users/Microcrew/Documents/Examensarbete/FreshStart/InstrumentHandoverPipeline/Pipeline/Plotting')

import logread

simple = logread.processLogFile("./Results/SimpleLog/Simple-log.txt")
topDown = logread.processLogFile("./Results/RotationTopDown/RotationTopDown-log.txt")
dwsd = logread.processLogFile("./Results/DiagonalWithStraightDepth/DiagonalWithStraightDepth-log.txt")
dwad = logread.processLogFile("./Results/DiagonalWithArchingDepthLog/DiagonalWithArchingDepth-log.txt")
arching = logread.processLogFile("./Results/ArchingLog/Arching-log.txt")

allLogs = [simple,topDown,dwsd,dwad,arching]

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
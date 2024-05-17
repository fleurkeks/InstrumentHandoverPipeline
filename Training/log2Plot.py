#takes our logfile in rvec, tvec and plots the positions over time.

import sys         
 
# appending the directory of mod.py 
# in the sys.path list
sys.path.append('C:/Users/Microcrew/Documents/Examensarbete/LeaRepo/InstrumentHandoverPipeline/Training/logging')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import logread as logread

filename = "Diagonal With Straight Depth"

data = logread.processLogFile("./" + filename.replace(' ', '') + "-log.txt")

aruco_Xpositions = []
aruco_Ypositions = []
aruco_Zpositions = []
hand_Xpositions = []
hand_Ypositions = []
hand_Zpositions = []

for frame in data:
    if frame["aruco_tvec0"] != None:
        aruco_Xpositions.append(frame["aruco_tvec0"])
    if frame["aruco_tvec1"] != None:
        aruco_Ypositions.append(frame["aruco_tvec1"])
    if frame["aruco_tvec2"] != None:
        aruco_Zpositions.append(frame["aruco_tvec2"])
    if frame["hand_tvec0"] != None:
        hand_Xpositions.append(frame["hand_tvec0"])
    if frame["hand_tvec1"] != None:
        hand_Ypositions.append(frame["hand_tvec1"])
    if frame["hand_tvec2"] != None:
        hand_Zpositions.append(frame["hand_tvec2"])

aruco_positions = np.array([aruco_Xpositions, aruco_Zpositions, aruco_Ypositions]) #organizing it x, z, y produces more easy to read plots
hand_positions = np.array([hand_Xpositions, hand_Zpositions, hand_Ypositions])

#aruco_positions = np.array([[data[514]["aruco_tvec0"], data[515]["aruco_tvec0"], data[516]["aruco_tvec0"]], [data[514]["aruco_tvec1"], data[515]["aruco_tvec1"], data[516]["aruco_tvec1"]], [data[514]["aruco_tvec2"], data[515]["aruco_tvec2"], data[516]["aruco_tvec2"]]]) 
#hand_positions = np.array([[data[514]["hand_tvec0"], data[515]["hand_tvec0"], data[516]["hand_tvec0"]], [data[514]["hand_tvec1"], data[515]["hand_tvec1"], data[516]["hand_tvec1"]], [data[514]["hand_tvec2"], data[515]["hand_tvec2"], data[516]["hand_tvec2"]]])  # Object 2 positions at different time points

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot aruco trajectory
ax.plot(aruco_positions[0], aruco_positions[1], aruco_positions[2], marker='o', label='aruco')

# Plot hand trajectory
ax.plot(hand_positions[0], hand_positions[1], hand_positions[2], marker='o', label='hand')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')
ax.set_title('3D Plot of ' + filename)
ax.invert_zaxis()
# Add legend
ax.legend()

plt.show()
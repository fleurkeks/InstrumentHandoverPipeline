#takes our logfile in rvec, tvec and plots the positions over time.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import logread

data = logread.processLogFile("Simple-log-smoothed.txt")
title="3D Plot of Coordinates over Time"
filename="simple-log-smoothed.png"

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

aruco_positions = np.array([aruco_Xpositions, aruco_Ypositions, aruco_Zpositions])
hand_positions = np.array([hand_Xpositions, hand_Ypositions, hand_Zpositions])


# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot aruco trajectory
ax.plot(aruco_positions[0], aruco_positions[1], aruco_positions[2], marker='o', label='aruco')

# Plot hand trajectory
ax.plot(hand_positions[0], hand_positions[1], hand_positions[2], marker='o', label='hand')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

##################change title##################
ax.set_title(title)


ax.legend()

#####################chan
plt.savefig(filename)

plt.show()
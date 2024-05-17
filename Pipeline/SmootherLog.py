import averageQuaternion
import numpy as np
from transforms3d.euler import mat2euler, euler2quat, euler2mat, quat2mat
import cv2

# Read a log from file
# Trajectory format: (frame "aruco" M.x M.y M.z M.r1 M.r2 M.r3 "hand" H.x H.y H.z H.r1 H.r2 H.r3  )
# filters outliers
# Takes a rolling average of three

def average_points(point1, point2, point3):
    avg_x = (point1[0] + point2[0] + point3[0]) / 3
    avg_y = (point1[1] + point2[1] + point3[1]) / 3
    avg_z = (point1[2] + point2[2] + point3[2]) / 3
    return avg_x, avg_y, avg_z

def get_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + 
                   (point1[1] - point2[1])**2 + 
                   (point1[2] - point2[2])**2)

def getdata(filename, distance_threshold):
    smoothed = []
    data_points = []
    with open(filename) as fn:
        lines = fn.readlines()
        for line in lines:
            line = line.strip().split()
            frame = float(line[0])

            if len(line) == 15:
                # Extract marker data and convert into quaternion format
                rVecMarker = np.array(list(map(float, line[2:5])))
                dVecMarker = list(map(float, line[5:8]))

                RMarker, _ = cv2.Rodrigues(rVecMarker)
                EulerMarker = mat2euler(RMarker)
                quatMarker = euler2quat(EulerMarker[0], EulerMarker[1], EulerMarker[2])

                # Do the same with the hand
                rVecHand = np.array(list(map(float, line[9:12])))
                dVecHand = list(map(float, line[12:]))

                RHand, _ = cv2.Rodrigues(rVecHand)
                EulerHand = mat2euler(RHand)
                quatHand = euler2quat(EulerHand[0], EulerHand[1], EulerHand[2])

               
                if len(data_points) == 0 or len(data_points) == 1 or get_distance(dVecMarker, data_points[-1][1]) <= distance_threshold:
                    data_points.append((int(frame), dVecMarker, quatMarker, dVecHand, quatHand))

    # Apply rolling average with window size 3
    for i in range(2, len(data_points)):
        frame = data_points[i][0]
        MarkerMeanPos = average_points(data_points[i-2][1], data_points[i-1][1], data_points[i][1])
        HandMeanPos = average_points(data_points[i-2][3], data_points[i-1][3], data_points[i][3])

        # Creates a matrix that contains the three quats we wanna average
        quatMatMarker = np.row_stack((data_points[i-2][2], data_points[i-1][2], data_points[i][2]))
        meanQuatMarker = averageQuaternion.averageQuaternions(quatMatMarker)
        meanMatMarker = quat2mat(meanQuatMarker)
        rVecMarker, _ = cv2.Rodrigues(meanMatMarker)

        quatMatHand = np.row_stack((data_points[i-2][4], data_points[i-1][4], data_points[i][4]))
        meanQuatHand = averageQuaternion.averageQuaternions(quatMatHand)
        meanMatHand = quat2mat(meanQuatHand)
        rVecHand, _ = cv2.Rodrigues(meanMatHand)

        smoothed.append((int(frame), MarkerMeanPos, rVecMarker.flatten(), HandMeanPos, rVecHand.flatten()))

    return smoothed

def main():
    distance_threshold = 130.0  # You can change this threshold distance as needed
    traj = getdata("Simple-log.txt", distance_threshold)
    with open('Simple-log-smoothed.txt', 'w') as f:
        for item in traj:
            frame_nbr = item[0]
            MarkerMeanPos = item[1]
            meanrVecMarker = item[2]
            HandMeanPos = item[3]
            meanrVecHand = item[4]

            line = f"{frame_nbr} " \
                   f"aruco " \
                   f"{' '.join(map(str, meanrVecMarker))} " \
                   f"{' '.join(map(str, MarkerMeanPos))} " \
                   f"hand " \
                   f"{' '.join(map(str, meanrVecHand))} " \
                   f"{' '.join(map(str, HandMeanPos))}\n"

            f.write(line)

if __name__ == "__main__":
    main()

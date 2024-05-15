#takes a file that is in quaternion format and smoothes it
#returns quaternion file



import averageQuaternion
import numpy as np
from transforms3d.euler import (mat2euler, euler2quat, euler2mat)


#read a trajectory from file
#trajectory format: (T.x T.y T.z T.q1 T.q2 T.q3 T.q4 )
#the first three items give the displacement along x,y,z and the last 4 items give the orientation in quaternion
#takes a rolling average of three#

def average_points(point1, point2, point3):
    
    avg_x = (point1[0] + point2[0] + point3[0]) / 3
    avg_y = (point1[1] + point2[1] + point3[1]) / 3
    avg_z = (point1[2] + point2[2] + point3[2]) / 3
    return avg_x, avg_y, avg_z


def getdata(filename):
    
    data = []
    data_points=[]
    with open(filename) as fn:
        lines = fn.readlines()
        for line in lines:
            
            line = line.strip().split()
            frame=line[0]
            displacement = list(map(float, line[1:4]))
            orientation = list(map(float, line[4:]))
            data_points.append((int(frame), displacement, orientation))
            print(data_points)
            
    # Apply rolling average with window size 3
    for i in range(2, len(data_points)):
        frame=data_points[i][0]
        meanPos=average_points(data_points[i-2][1], data_points[i-1][1], data_points[i][1])

        # Creates a matrix that contains the three quats we wanna average
        quatMat = np.row_stack((data_points[i-2][2],data_points[i-1][2],data_points[i][2]))
        #quatMat = np.row_stack((np.array(data_points[i-2][2]), np.array(data_points[i-1][2]), np.array(data_points[i][2])))
        meanQuat=averageQuaternion.averageQuaternions(quatMat)
        data.append((int(frame), (meanPos, meanQuat)))

    return data




def main():
    
    traj = getdata("trajectory.txt")
    with open('smoothed.txt', 'w') as f:
        for frame_nbr, item in traj:
            line = f"{frame_nbr} {' '.join(map(str, item[0]))} {' '.join(map(str, item[1]))}\n"
            f.write(line)
        


if __name__ == "__main__":
    main()
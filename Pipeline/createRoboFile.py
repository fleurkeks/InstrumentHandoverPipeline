import numpy as np
from transforms3d.euler import (mat2euler, euler2quat, euler2mat)

PREFIX = '''
MODULE MainModule
PERS tooldata Servo:=[TRUE,[[0,0,114.2],[1,0,0,0]],[0.215,[8.7,12.3,49.2],[1,0,0,0],0.00021,0.00024,0.00009]];
PERS tooldata tObject:=[TRUE,[[0,0,130],[0.707106781,-0.707106781,0,0]],[0.215,[8.7,12.3,49.2],[1,0,0,0],0.00021,0.00024,0.00009]];
PERS wobjdata handpos:= [FALSE,TRUE,"",[[-400,0,250],[1,0,0,0]],[[0,0,0],[1,0,0,0]]];
    PROC main()
        ConfL\\Off;
        SingArea\\Wrist;
        !Startpos
        MoveAbsJ [[0.441359,-16.418,10.8814,2.79121,60.2034,-14.3875],[168.362,9E+09,9E+09,9E+09,9E+09,9E+09]] \\NoEOffs, v1000, fine, wi_tGripper;
        path;
    ENDPROC
    PROC path()
'''


SUFFIX = '''
ENDPROC
ENDMODULE
'''

#read a trajectory from file
#trajectory format: (marker.x marker.y marker.z marker.q1 marker.q2 marker.q3 marker.q4 hand.x hand.y hand.z hand.q1 hand.q2 hand.q3 hand.q4)
#reads file that is in quaternion format

#during non-live tracking we can assume the hand is fixed so we are only feeding the vectors (dvec and quaternions) that describe the movement between marker and hand 
#as given by our trajectory file
def getdata(filename):
    data = []
    with open(filename) as fn:
        lines = fn.readlines()
        for line in lines:
            datapoint = line.split()
            frame = int(datapoint[0])
            
            position = list(map(float, datapoint[1:]))
            dvec = position[0:3]
            quat = position[3:7]

            data.append((dvec, quat))

    return data


def gencode(traj):
    declarations = []
    motions = []
    for i, (tvec, q) in enumerate(traj):
        #data is converted into mm which the robot uses
        x, y, z = 1000 * tvec[0],  1000 * tvec[1],  1000 * tvec[2] 
        # Double check the order of the quaternion!!!
        # Change the elbow angle?
        decl = f'VAR robtarget t{i} := [[{x}, {y}, {z}], [0, -1, 0, 0], [0,0,-1,4], [-96,9E+09,9E+09,9E+09,9E+09,9E+09]];'
        declarations.append(decl)

        #move linearly along vector + velocity + z1 om inom en mm far den gar vidare, blending + punkten som ska move (just nu gripper, kommer ändras till pincet, robotern
        #wObject, transform the coordinate system relative to work Object. effectively move towards in our case
        motions.append(f"MoveL t{i}, v100\V:=100.000, z1, wi_tGripper, \WObj:=handpos;")
    s = '\n'.join(declarations) + '\n'.join(motions)
    return s

def main():
    traj = getdata("smoothed.txt")
    path_str = gencode(traj)
    program = [PREFIX, path_str, SUFFIX]
    with open('traj1smoothed.MOD', 'w') as f:
        code_str = '\n'.join(program)
        f.write(code_str)
    


if __name__ == "__main__":
    main()
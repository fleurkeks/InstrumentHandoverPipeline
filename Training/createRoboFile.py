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

def getdata(filename):
    data = []
    with open(filename) as fn:
        lines = fn.readlines()
        for line in lines:
            datapoint = line.split()
            frame = int(datapoint[0])
            
            position = list(map(float, datapoint[1:]))
            tMarker = position[0:3]
            quatMarker = position[3:7]

            #######################to do 
            #fix correct difference
            
            tHand=position[7:10]
            quatHand=position[10:14]
        

            tHand=np.array(tHand)
            quatHand=np.array(quatHand)

            #translation from hand to marker
            tvec=tMarker-tHand
            
            #rotation from hand to marker
            quat=quatMarker-quatHand
          
            #calculate the difference from hand to marker, append to our data
            data.append((tvec, quat))

    return data

###################################TODO##################
#smoothe data
#filter data


def gencode(traj):
    declarations = []
    motions = []
    for i, (tvec, q) in enumerate(traj):
        x, y, z = 1000 * tvec[0],  1000 * tvec[1],  1000 * tvec[2] #converting to mm
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
    traj = getdata("Training\hand_positions.txt")
    path_str = gencode(traj)
    program = [PREFIX, path_str, SUFFIX]
    with open('robotcode.MOD', 'w') as f:
        code_str = '\n'.join(program)
        f.write(code_str)
    



if __name__ == "__main__":
    main()
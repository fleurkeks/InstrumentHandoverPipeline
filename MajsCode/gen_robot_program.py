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
        MoveAbsJ [[0.441359,-16.418,10.8814,2.79121,60.2034,-14.3875],[168.362,9E+09,9E+09,9E+09,9E+09,9E+09]] \NoEOffs, v1000, fine, wi_tGripper;
        path;
    ENDPROC
    PROC path()
'''

SUFFIX = '''
ENDPROC
ENDMODULE
'''

#create a trajectry and read it from file
#räknar ut diff mellan hand markör
def getdata(filename):
    data = []
    with open(filename) as fn:
        lines = fn.readlines()
        for line in lines:
            datapoint = line.split()
            frame = int(datapoint[0])
            position = list(map(float, datapoint[1:]))
            tvec = position[0:3]
            quat = position[3:7]
            data.append((tvec, quat))

    return data

def gencode(traj):
    declarations = []
    motions = []
    for i, (tvec, q) in enumerate(traj):
        #VAR robtarget t8 
        #VAR robtarget t9 := [[-7.369, 75.120, 214.511], [0.212467, 0.658133, -0.719321, -0.065544], [1,-2,0,0], [163.420,9E+09,9E+09,9E+09,9E+09,9E+09]]; 
        print(tvec)
        print(len(tvec))
        x, y, z = 1000 * tvec[0],  1000 * tvec[1],  1000 * tvec[2] #converting to mm
        # Double check the order of the quaternion!!!
        # Change the elbow angle?
        decl = f'VAR robtarget t{i} := [[{x}, {y}, {z}], [0, -1, 0, 0], [0,0,-1,4], [-96,9E+09,9E+09,9E+09,9E+09,9E+09]];'
        declarations.append(decl)
        motions.append(f"MoveL t{i}, v100\V:=100.000, z1, wi_tGripper, \WObj:=handpos;")
    s = '\n'.join(declarations) + '\n'.join(motions)
    return s

def main():
    traj = getdata("MajsCode\hand_positions.txt")
    path_str = gencode(traj)
    program = [PREFIX, path_str, SUFFIX]
    with open('robotcode.MOD', 'w') as f:
        code_str = '\n'.join(program)
        f.write(code_str)
    



if __name__ == "__main__":
    main()
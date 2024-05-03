#given the logged positions of the training trajectory we create a file that is used to train the robots movements
#this includes transforming point data to the correct format and smoothing data

#from matrix of hand and marker to direction matrix. 

#average quaterions
#average direction
#smooth plots



import numpy as np

from transforms3d.euler import (mat2euler, euler2quat, euler2mat)

PREFIX = '''
MODULE MainModule
PERS tooldata Servo:=[TRUE,[[0,0,114.2],[1,0,0,0]],[0.215,[8.7,12.3,49.2],[1,0,0,0],0.00021,0.00024,0.00009]];
PERS tooldata tDie:=[TRUE,[[0,0,130],[0.707106781,-0.707106781,0,0]],[0.215,[8.7,12.3,49.2],[1,0,0,0],0.00021,0.00024,0.00009]];
PERS wobjdata campos:= [FALSE,TRUE,"",[[500,-100,500],[0,-0.707106781,0.707106781,0]],[[0,0,0],[1,0,0,0]]];
    PROC main()
        ConfL\\Off;
        SingArea\\Wrist;
        !Startpos
        MoveAbsJ [[-92.4278,-44.3477,23.0081,-55.6876,60.7508,-103.887],[105.882,9E+9,9E+9,9E+9,9E+9,9E+9]]\\NoEOffs, v1000, z50, t\\WObj:=campos;
        path;
    ENDPROC
    PROC path()
'''

SUFFIX = '''
ENDPROC
ENDMODULE
'''

#create a trajectry and read it from file
def gentestdata(N = 1000):
    data = []
    delta = np.pi/N
    for i in range(N):
        noise = 0.5 * np.random.randn(3)
        point = np.array([40 * np.sin(i *delta), 60* np.sin(2 * i * delta), 30* np.cos(i * delta)]) + noise
        angle = np.array([0, 0, 0])
        data.append((point, angle))
    return data

def gencode(traj):
    declarations = []
    motions = []
    for i, (tvec, rvec) in enumerate(traj):
        #VAR robtarget t8 
        #VAR robtarget t9 := [[-7.369, 75.120, 214.511], [0.212467, 0.658133, -0.719321, -0.065544], [1,-2,0,0], [163.420,9E+09,9E+09,9E+09,9E+09,9E+09]]; 
        x, y, z = 1000 * tvec #converting to mm
        # Double check the order of the quaternion!!!
        q = euler2quat(rvec[0], rvec[1], rvec[2])
        # Change the elbow angle?
        decl = f'VAR robtarget t{i} := [[{x}, {y}, {z}], [{q[0]}, {q[1]}, {q[2]}, {q[3]}], [1,-2,0,0], [163.420,9E+09,9E+09,9E+09,9E+09,9E+09]];'
        declarations.append(decl)
        motions.append(f"MoveL t{i}, v100\V:=100.000, z1, tObject, \WObj:=handpos;")
    s = '\n'.join(declarations) + '\n'.join(motions)
    return s

def main():
    traj = gentestdata()
    path_str = gencode(traj)
    program = [PREFIX, path_str, SUFFIX]
    with open('robotcode.MOD', 'w') as f:
        code_str = '\n'.join(program)
        f.write(code_str)
    



if __name__ == "__main__":
    main()
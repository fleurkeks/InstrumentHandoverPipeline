#contains filtering and smoothing



import numpy as np

from skspatial.objects import Points, Line
from skspatial.plotting import plot_3d
from collections import defaultdict

def readlog(log, CENTID):
    positions = []
    cent_pos = []
    frames = []
    # Define the codec and create VideoWriter object
    
    with open(log, 'r+') as logfile: 
        
        for line in logfile.readlines():
            line = line.replace('[', '')
            line = line.replace(']', '')
            
            data = line.split()
            frameno = int(float(data[0]))
            frames.append(frameno)
            id = int(data[1])
            if id == CENTID:
                rvec = np.array(list(map(float, data[2:5])))
                tvec = np.array(list(map(float, data[5:8])))
                cent_pos.append((frameno, rvec, tvec))
            elif id < 10:
                corners = []
                for i in range(4):
                    px, py = float(data[2 + 2 * i]), float(data[2 + 2 * i + 1])
                    corners.append((px, py))
                positions.append((frameno, corners))
    return positions, cent_pos, frames

def dist(p1, p2):
    su = 0.0
    
    for i in range(3):
        su += (p1[i] - p2[i]) ** 2
        if abs(p1[i]) > 1000 or abs(p2[i]) > 1000:
            return 0
    euk = su ** 0.5
    return euk/1000

def tot_dist(cent_pos):
    tot = 0.0
    last_frame = -10
    _, _, last = cent_pos[0]
    #last = [0, 0, 0]
    cnt = 0
    #print(len(cent_pos))
    mx = 0
    streak = defaultdict(int)
    streak_dist = {}
    started_at = 0
    tmp = 0
    for frameno, rvec, tvec in cent_pos[1:]:
        #if frameno < last_frame + 2:
        d = dist(last, tvec)
        tot += d
        last = tvec
        #tmp += d
            #d = max(d, mx)
            #print(f'Dist {d} {last} {tvec}')
            
        #else:
        #    last_frame = frameno
        #    started_at = frameno
        #    last = tvec
        #    tmp = 0
        
        #streak_dist[started_at] = tmp
        #streak[started_at] += 1
        #cnt += 1
        #if cnt % 5000:
            #print(tot)
        
    #v = list(streak.values())
    #v.sort()
    #mx_len = v[-1]
    print("Tot dist =", tot)
    calc_speed2(cent_pos)
    #calc_speed(streak, streak_dist)
    

    return tot

def filter_data(cent_pos):
    filtered = []
    for frameno, rvec, tvec in cent_pos:
        z = tvec[2]
        if 300 <= z <= 700:
            filtered.append((frameno, rvec, tvec))
    return filtered

def get_smoothed(curr):
    su = sum(curr)
    return su/len(curr)

def smooth(cent_pos):

    smoothness = 1  
    new_pos = {}
    curr = []
    '''
    for prev in range(start, start+10):
        if prev in positions:
            curr.append(positions[prev][0])
            print(prev)
    '''
    #print(len(curr))
    smoothed = []
    frames = []
    for frameno, rvec, tvec in cent_pos:
        if len(curr) == 0:
            curr = [tvec] * smoothness
        else:
            curr = curr[1:]
            curr.append(tvec)
        v = get_smoothed(curr)
        smoothed.append((frameno, rvec, v))
        #frames.append(frameno)
    
    return smoothed
 

def calc_speed2(cent_pos):

    last_frame, _, last = cent_pos[0]
    fps = 1/25
    mx_speed = 0
    tot_speed = 0
    cnt = 0

    for frameno, rvec, tvec in cent_pos[1:]:
        #if frameno < last_frame + 2:
        d = dist(last, tvec)
        dt = (frameno - last_frame) * fps
        v = d/dt
        tot_speed += v
        cnt += 1
        '''
        if v > mx_speed:
            print(f'Max speed {v} between {last_frame} and {frameno}')
            print(f'Pos {last} to {tvec}')
        '''
        mx_speed = max(mx_speed, v)
        last_frame = frameno
        last = tvec
    print(f'Max speed {mx_speed}')
    print(f'Av speed {tot_speed/cnt}')

def calc_speed(streaks, streak_dist):

    fps = 1/25
    mx_speed = 0

    for start_at, d in streak_dist.items():
        no = streaks[start_at]
        time = no * fps
        speed = d/time
        mx_speed = max(mx_speed, speed)
    print(f'Max speed {mx_speed}')

def main():
    FILE = './trimmed2.mp4'
    _, cent_pos, _ = readlog('Case1_min2.txt', 10)
    
    cent_pos = smooth(filter_data(cent_pos))
    print(len(cent_pos))
    
    d = tot_dist(cent_pos)
    print(f'Tot len {d}')


if __name__ == '__main__':

    main()
def processLogFile(fileName):
    returnable = []
    with open(fileName, "r") as f:
        for line in f:
            words = line.split()
            if len(words) == 0:
                continue
            index = 0
            frame = words[index]

            d = {
                "frame" : frame,
                "aruco_rvec0" : None,
                "aruco_rvec1" : None,
                "aruco_rvec2" : None,
                "aruco_tvec0" : None,
                "aruco_tvec1" : None,
                "aruco_tvec2" : None,
                "hand_rvec0" : None,
                "hand_rvec1" : None,
                "hand_rvec2" : None,
                "hand_tvec0" : None,
                "hand_tvec1" : None,
                "hand_tvec2" : None,
            }

            index += 1
            while index < len(words):
                if words[index] == "aruco":
                    d["aruco_rvec0"] = float(words[index+1]) 
                    d["aruco_rvec1"] = float(words[index+2]) 
                    d["aruco_rvec2"] = float(words[index+3]) 
                    d["aruco_tvec0"] = float(words[index+4])
                    d["aruco_tvec1"] = float(words[index+5])
                    d["aruco_tvec2"] = float(words[index+6])
                    
                    index += 7

                elif words[index] == "hand":
                    d["hand_rvec0"] = float(words[index+1])
                    d["hand_rvec1"] = float(words[index+2])
                    d["hand_rvec2"] = float(words[index+3]) 
                    d["hand_tvec0"] = float(words[index+4]) 
                    d["hand_tvec1"] = float(words[index+5])
                    d["hand_tvec2"] = float(words[index+6])

                    index += 7

            
            returnable.append(d)

        
     
    return returnable
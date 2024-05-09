#takes our logfile in rvec, tvec and calculates the difference between marker and hands by converting to a matrix and doing matrix calc
#returns file in quaternions
#i.e this converts from marker and hand log to a log of directional vectors that describe the movement of the marker to the hand 
#(obs! in terms of hand2marker, because that is the format our robot uses)
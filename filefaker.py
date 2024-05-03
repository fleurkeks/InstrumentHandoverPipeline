import numpy as np
from scipy.spatial.transform import Rotation as R

# Define parameters
num_frames = 200  # Number of frames
hand_position = [0.3, 0.3, 0.3]  # Hand position (stationary) within the room


# Generate fake data
with open("hand_positions.txt", "w") as file:
    marker_position=[0, 0, 0]
    for frame in range(num_frames):
        # Move marker towards hand along a straight line
        steps=0.15/num_frames
        marker_position[0] = 0.15 +steps*frame
        marker_position[1] =0.15 + steps*frame
        marker_position[2] = 0.15 + steps*frame
        # Generate random orientation for marker (quaternion)
        marker_orientation = [1, 0.0, 0.0, 0.0]
        #R.from_quat(np.random.rand(4)).as_quat()
        
        # Write data to file with frame number
        data = [frame + 1] + list(marker_position) + list(marker_orientation) + hand_position + [1, 0, 0, 0]
        file.write(" ".join(map(str, data)) + "\n")

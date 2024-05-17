import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to perform quaternion multiplication
def quaternion_multiply(q1, q0):
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

# Read file and store transformations
def read_transformations(file_name):
    transformations = []
    with open(file_name, 'r') as file:
        for line in file:
            data = line.strip().split()
            T = np.array([float(x) for x in data[1:]], dtype=np.float64)
            transformations.append(T)
    return transformations

# Plot vectors using local transformations
def plot_vectors(transformations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    start_point=transformations[0][0:3]
    prev_translation=start_point
    

    for idx, T in enumerate(transformations):
        # Extract translation and rotation from transformation
        translation = T[:3]
      
        # Perform local transformation
        current_translation=prev_translation+translation

         # Plot line segment
        if idx > 0:  # Skip the first transformation if it's the starting point
            ax.plot([prev_translation[0], current_translation[0]], 
                    [prev_translation[1], current_translation[1]], 
                    [prev_translation[2], current_translation[2]], color='b')

        # Update previous translation for next transformation
        prev_translation = current_translation.copy()


    # Plot starting point
    ax.scatter(start_point[0], start_point[1], start_point[2], color='r', label='Starting Point')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory vectors for each frame')
    ax.legend()
    plt.show()

# Example usage
file_name = 'smoothed4.txt'  # Change this to your file name
transformations = read_transformations(file_name)
plot_vectors(transformations)

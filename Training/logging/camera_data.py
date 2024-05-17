import yaml
import numpy as np

def sony_data():
    YAML_FILE = "data_sony_new.yaml"
    with open(YAML_FILE, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        camera_matrix = np.array(data_loaded['camera_matrix'])
        dist_coeffs = np.array(data_loaded['dist_coeff'])
    return camera_matrix, dist_coeffs

def sony_data_2cam():
    YAML_FILE = "data_sony_2cam.yaml"
    with open(YAML_FILE, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        camera_matrixR = np.array(data_loaded['camera_matrixR'])
        camera_matrixL = np.array(data_loaded['camera_matrixL'])
        dist_coeffR = np.array(data_loaded['dist_coeffR'])
        dist_coeffL = np.array(data_loaded['dist_coeffL'])
    return camera_matrixR, camera_matrixL, dist_coeffR, dist_coeffL

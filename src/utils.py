import os
import csv
import json
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def getInputData(data_path):
    curr_vels = []
    pos = []
    target_vels = []

    with open(data_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            target_vels.append([float(row[1]), float(row[2])])  # target velocity
            curr_vels.append([float(row[3]), float(row[4])])    # current velocity
            pos.append([float(row[5]), float(row[6]), float(row[7])])      # position (x, z) and angle

    curr_vels = curr_vels[:-1] # now curr_vels
    target_vels = target_vels[1:] # next target_vels
    pos_delta = []
    for i in range(len(pos) - 1):
        global_dx = pos[i + 1][0] - pos[i][0]
        global_dz = pos[i + 1][1] - pos[i][1]
        d_angle = pos[i + 1][2] - pos[i][2]
        if d_angle > 180:
            d_angle -= 360
        elif d_angle < -180:
            d_angle += 360
        
        # convert global deltas to local deltas
        # in unity, angle 0 means facing along positive z axis
        angle_rad = np.deg2rad(pos[i][2])
        local_dx = global_dx * np.cos(angle_rad) + global_dz * np.sin(angle_rad)
        local_dz = -global_dx * np.sin(angle_rad) + global_dz * np.cos(angle_rad)
        pos_delta.append([local_dx, local_dz, d_angle])

    pos_input = [[0.0, 0.0, 0.0]] + pos_delta[:-1]  # initial delta is zero
    # concat vels and pos_input to form input data
    input_data = []
    for i in range(len(curr_vels)):
        input_data.append(target_vels[i] + curr_vels[i] + pos_input[i])

    output_data = pos_delta

    input_data, input_scaler = normalize(np.array(input_data))
    output_data, output_scaler = normalize(np.array(output_data))

    joblib.dump(input_scaler, os.path.join(os.path.dirname(__file__), '..', 'input_scaler.save'))
    joblib.dump(output_scaler, os.path.join(os.path.dirname(__file__), '..', 'output_scaler.save'))

    train_size = int(0.8 * len(input_data))
    train_input = input_data[:train_size]
    train_output = output_data[:train_size]
    test_input = input_data[train_size:]
    test_output = output_data[train_size:]
    return train_input, train_output, test_input, test_output, input_scaler, output_scaler

def loadConfig():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def normalize(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data, scaler

def splitTrainVal(X, Y, val_ratio=0.1):
    total_size = X.shape[0]
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_x = X[:train_size]
    train_y = Y[:train_size]
    val_x = X[train_size:]
    val_y = Y[train_size:]

    return train_x, train_y, val_x, val_y
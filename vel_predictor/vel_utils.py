import os
import csv
import json
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def getInputData(data_path):
    target_vels = []
    car_pos = []
    current_vels = []
    target_pos = []

    with open(data_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            target_vels.append([float(row[1]), float(row[2])])  # target velocity
            current_vels.append([float(row[3]), float(row[4])])  # current velocity
            car_pos.append([float(row[5]), float(row[6]), float(row[7])])  # position x, z, angle
            target_pos.append([float(row[8]), float(row[9]), float(row[10])])  # target position x, z, angle

    target_vels = target_vels[1:]  # remove first entry to align with state deltas
    current_vels = current_vels[:-1]  # remove last entry to align with state deltas
    pos_delta = []
    for i in range(len(car_pos) - 1):
        global_dx = target_pos[i][0] - car_pos[i][0]
        global_dz = target_pos[i][1] - car_pos[i][1]
        d_angle = target_pos[i][2] - car_pos[i][2]
        if d_angle > 180:
            d_angle -= 360
        elif d_angle < -180:
            d_angle += 360
        
        # convert global deltas to local car frame
        angle_rad = np.radians(car_pos[i][2])
        local_dx = global_dx * np.cos(-angle_rad) - global_dz * np.sin(-angle_rad)
        local_dz = global_dx * np.sin(-angle_rad) + global_dz * np.cos(-angle_rad)
        pos_delta.append([local_dx, local_dz, d_angle])

    input_data = np.hstack((current_vels, pos_delta))  # concatenate current velocity and position deltas
    output_data = np.array(target_vels)

    input_data, input_scaler = normalize(input_data)
    output_data, output_scaler = normalize(output_data)

    joblib.dump(input_scaler, os.path.join(os.path.dirname(__file__), '..', 'input_scaler.save'))
    joblib.dump(output_scaler, os.path.join(os.path.dirname(__file__), '..', 'output_scaler.save'))

    train_size = int(0.8 * len(target_vels))
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

def splitTrainVal(input_data, output_data, val_ratio=0.1):
    total_size = len(input_data)
    val_size = int(total_size * val_ratio)
    train_input = input_data[:-val_size]
    train_output = output_data[:-val_size]
    val_input = input_data[-val_size:]
    val_output = output_data[-val_size:]
    return train_input, train_output, val_input, val_output
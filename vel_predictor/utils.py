import os
import csv
import json
import torch
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def getInputData(data_path):
    vels = []
    states = []

    with open(data_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            vels.append([float(row[1]), float(row[2])])  # target velocities
            # current state: vel_left, vel_right, pos_x, pos_z, angle
            states.append([float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])

    vels = vels[:-1] # remove last velocity since we can't compute its delta state
    state_delta = []
    for i in range(len(states) - 1):
        global_dx = states[i + 1][2] - states[i][2]
        global_dz = states[i + 1][3] - states[i][3]
        d_angle = states[i + 1][4] - states[i][4]
        if d_angle > 180:
            d_angle -= 360
        elif d_angle < -180:
            d_angle += 360
        
        # convert global deltas to local deltas
        # in unity, angle 0 means facing along positive z axis
        angle_rad = np.deg2rad(states[i][4])
        local_dx = global_dx * np.cos(angle_rad) + global_dz * np.sin(angle_rad)
        local_dz = -global_dx * np.sin(angle_rad) + global_dz * np.cos(angle_rad)
        state_delta.append([states[i][0], states[i][1], local_dx, local_dz, d_angle])

    vels, vel_scaler = normalize(np.array(vels))
    state_delta, state_scaler = normalize(np.array(state_delta))

    joblib.dump(vel_scaler, os.path.join(os.path.dirname(__file__), '..', 'vel_scaler.save'))
    joblib.dump(state_scaler, os.path.join(os.path.dirname(__file__), '..', 'state_scaler.save'))

    train_size = int(0.8 * len(vels))
    train_vels = vels[:train_size]
    train_states = state_delta[:train_size]

    test_vels = vels[train_size:]
    test_states = state_delta[train_size:]
    return train_vels, train_states, test_vels, test_states, vel_scaler, state_scaler

def loadConfig():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def normalize(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, scaler

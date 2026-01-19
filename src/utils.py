import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def getInputData(data_path):
    df = pd.read_csv(data_path)
    inputs = []
    outputs = []

    angle = np.deg2rad(df['angle'].values)
    pos_x = df['pos_x'].values
    pos_y = df['pos_y'].values
    cmd_l = df['target_vel_left'].values
    cmd_r = df['target_vel_right'].values

    lookahead = 20  # number of steps to look ahead for target velocity
    count = len(df) - lookahead
    for i in range(count):
        curr_vel_l = df['vel_left'].values[i]
        curr_vel_r = df['vel_right'].values[i]

        delta_x = pos_x[i + lookahead] - pos_x[i]
        delta_y = pos_y[i + lookahead] - pos_y[i]
        delta_angle = angle[i + lookahead] - angle[i]
        delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]

        local_x = delta_x * np.cos(-angle[i]) - delta_y * np.sin(-angle[i])
        local_y = delta_x * np.sin(-angle[i]) + delta_y * np.cos(-angle[i])

        input_vector = [cmd_l[i], cmd_r[i], curr_vel_l, curr_vel_r]
        output_vector = [local_x, local_y, delta_angle]

        inputs.append(input_vector)
        outputs.append(output_vector)

    return np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32)

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
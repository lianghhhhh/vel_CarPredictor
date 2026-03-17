import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def getInputData(data_path):
    df = pd.read_csv(data_path)
    inputs = []
    outputs = []

    for i in range(len(df)):
        curr_vel_l = df['vel_left'].values[i]
        curr_vel_r = df['vel_right'].values[i]
        curr_x = df['pos_x'].values[i]
        curr_y = df['pos_y'].values[i]
        curr_angle = np.deg2rad(df['angle'].values[i])
        curr_angle = curr_angle + np.pi  # flip angle by 180 degrees
        if curr_angle > np.pi:
            curr_angle -= 2 * np.pi  # normalize to [-pi, pi]

        input_vector = [-curr_vel_l, -curr_vel_r]
        output_vector = []

        for k in range(10):  # 10 target points
            target_x = df[f'target_x_{k}'].values[i]
            target_y = df[f'target_y_{k}'].values[i]
            target_angle = df[f'target_angle_{k}'].values[i]
            delta_x = target_x - curr_x
            delta_y = target_y - curr_y
            local_x = delta_x * np.cos(-curr_angle) - delta_y * np.sin(-curr_angle)
            local_y = delta_x * np.sin(-curr_angle) + delta_y * np.cos(-curr_angle)
            delta_angle = target_angle - curr_angle
            delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]
            input_vector.append(local_x)
            input_vector.append(local_y)
            input_vector.append(delta_angle)

        for k in range(5):  # 5 obstacles
            obstacle_x = df[f'obstacle_x_{k}'].values[i]
            obstacle_y = df[f'obstacle_y_{k}'].values[i]
            obstacle_radius = df[f'obstacle_radius_{k}'].values[i]
            delta_x = obstacle_x - curr_x
            delta_y = obstacle_y - curr_y
            local_x = delta_x * np.cos(-curr_angle) - delta_y * np.sin(-curr_angle)
            local_y = delta_x * np.sin(-curr_angle) + delta_y * np.cos(-curr_angle)
            input_vector.append(local_x)
            input_vector.append(local_y)
            input_vector.append(obstacle_radius)

        for k in range(3):  # 3 planned path points
            planned_x = df[f'planned_x_{k}'].values[i]
            planned_y = df[f'planned_y_{k}'].values[i]
            planned_angle = df[f'planned_angle_{k}'].values[i]
            delta_x = planned_x - curr_x
            delta_y = planned_y - curr_y
            local_x = delta_x * np.cos(-curr_angle) - delta_y * np.sin(-curr_angle)
            local_y = delta_x * np.sin(-curr_angle) + delta_y * np.cos(-curr_angle)
            delta_angle = planned_angle - curr_angle
            delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]
            output_vector.append(local_x)
            output_vector.append(local_y)
            output_vector.append(delta_angle)

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
# a neural network model using car's current velocity and delta state to predict target velocity
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathGenerator import PathGenerator
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from path_utils import getInputData, loadConfig, splitTrainVal

def selectMode():
    print("Select mode:")
    print("1. Train")
    print("2. Inference")
    mode = input("Enter mode (1 or 2): ")
    return mode

def weighted_mse_loss(pred, target, inputs):
    # Standard Mean Squared Error
    loss = torch.nn.functional.mse_loss(pred, target, reduction='none')
    
    # Extract the obstacle flags from your inputs (X)
    # Assuming the 10 flags are located at indices 62, 65, 68... 89 in your 92-dim input
    flags = inputs[:, 62:92:3] 
    
    # If a row has any obstacle (sum > 0), give it a weight of 5.0. Otherwise 1.0.
    has_obs = (flags.sum(dim=1) > 0).float()
    weights = torch.where(has_obs > 0, 5.0, 1.0)
    
    # Apply weights and return the mean
    weighted_loss = loss.mean(dim=1) * weights
    return weighted_loss.mean()

def trainModel(model, input_tensor, output_tensor, epochs=100, learning_rate=0.001, name="model"):
    if os.path.exists(f'../{config["name"]}.pth'):
        print(f"Model {config['name']} already exists. Loading existing model.")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
    else:
        print(f"Using new model {config['name']}.")

    X = input_tensor
    Y = output_tensor
    train_x, train_y, val_x, val_y = splitTrainVal(X, Y)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    train_dataset = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(val_x, val_y)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    smoothness_weight = 0.1  # Weight for smoothness loss
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        train_loss = 0.0
        for i, data in enumerate(train_dataloader):
            batch_input = data[0].to(device)
            batch_output = data[1].to(device)
            model.train()
            optimizer.zero_grad()
            batch_input_flat = batch_input.view(batch_input.size(0), -1)
            batch_pred = model(batch_input_flat)
            # loss = criterion(batch_pred, batch_output)
            # loss += smoothness_weight * torch.mean(torch.abs(batch_pred[:, 2:] - batch_pred[:, :-2]))  # Smoothness loss
            loss = weighted_mse_loss(batch_pred, batch_output, batch_input)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                val_input = data[0].to(device)
                val_output = data[1].to(device)
                val_input_flat = val_input.view(val_input.size(0), -1)
                val_pred = model(val_input_flat)
                loss = weighted_mse_loss(val_pred, val_output, val_input)
                # loss = criterion(val_pred, val_output)
                # loss += smoothness_weight * torch.mean(torch.abs(val_pred[:, 2:] - val_pred[:, :-2]))  # Smoothness loss
                val_loss += loss.item()

        train_loss_avg = train_loss / len(train_dataloader)
        val_loss_avg = val_loss / len(val_dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_avg}, Val Loss: {val_loss_avg}')
        writer.add_scalar('Loss/train', train_loss_avg, epoch+1)
        writer.add_scalar('Loss/val', val_loss_avg, epoch+1)

        os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models', name), exist_ok=True)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '..', 'models', name, f'{epoch+1}.pth'))

    writer.close()
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '..', f'{name}.pth'))
    print(f'Model saved as {name}.pth')


def runInference(model, input_tensor, output_tensor, name="model"):
    X = input_tensor
    Y = output_tensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    X = X.to(device)
    Y = Y.to(device)
    print(X.shape, Y.shape)
    pred_list = []

    with torch.no_grad():
        for i in range(X.shape[0]):
            input = X[i].view(1, -1)  # shape: (1, input_size)
            vel_pred = model(input)  # shape: (1, output_size)
            pred_list.append(vel_pred.cpu())

    vel_pred = torch.cat(pred_list, dim=0)  # shape: (num_samples, output_size)
    vel_pred = vel_pred.numpy()
    Y_np = Y.cpu().numpy()

    print(vel_pred[0])
    print(Y_np[0])
    # plot the differences
    time_steps = np.arange(vel_pred.shape[0])
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    for i in range(3):  # plot first 3 dimensions
        difference = vel_pred[:, i] - Y_np[:, i]
        count = 0
        for j in range(difference.shape[0]):
            if abs(difference[j]) > 2.0:
                count += 1
                # print(f'Time step {j}: Predicted={vel_pred[j,i]}, Actual={Y_np[j,i]}, Difference={difference[j]}')
        print(f'Total significant differences (>2.0): {count}')
        axs[i].plot(time_steps, difference, label='Difference', color='red')
        axs[i].set_title(f'Dimension {i+1}')
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Value')
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(f'../{name}.png')
    plt.close()

if __name__ == "__main__":
    mode = selectMode()
    config = loadConfig()
    X, y = getInputData(config['data'])
    # X_2, y_2 = getInputData(config['data_2'])
    # X = np.concatenate((X, X_2), axis=0)
    # y = np.concatenate((y, y_2), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    print("Sample")
    print("input:", X_train[0:5], "shape:", X_train.shape)
    print("output:", y_train[0:5], "shape:", y_train.shape)
    model = PathGenerator(
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout']
    )

    if mode == "1":
        print("Training model.")
        trainModel(model, X_train, y_train, config['model']['epochs'], config['model']['learning_rate'], config['name'])

    elif mode == "2":
        print("Inference")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
        runInference(model, X_test, y_test, config['name'])

    else:
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
        test_input = [100, 100]
        test_input = torch.tensor(test_input).float()
        test_output = [90, 90, 0, 0, 0]
        test_output = torch.tensor(test_output).float()
        # Ensure both tensors have the same batch dimension
        test_input = test_input.unsqueeze(0)  # shape: (1, 2)
        test_output = test_output.unsqueeze(0)  # shape: (1, 5)
        input = torch.cat((test_input, test_output), dim=1)  # concatenate along last dimension
        print("Initial input:")
        print(input)
        predict_x = []
        model.eval()
        with torch.no_grad():
            for step in range(50):  # predict 50 steps ahead
                next_x = model(input)
                predict_x.append(next_x.cpu().numpy())
                input = torch.cat((test_input, next_x), dim=2)

            print("Predicted x over 50 steps:")
            predict_x = np.array(predict_x).squeeze()
            print(predict_x)
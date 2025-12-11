# a neural network model using car's target velocity and current velocity to predict delta state
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from carPredictor import CarPredictor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from utils import getInputData, loadConfig, splitTrainVal

def selectMode():
    print("Select mode:")
    print("1. Train")
    print("2. Inference")
    mode = input("Enter mode (1 or 2): ")
    return mode

def trainModel(model, input_tensor, output_tensor, epochs=100, learning_rate=0.001, name="model"):
    if os.path.exists(f'../{config["name"]}.pth'):
        print(f"Model {config['name']} already exists. Loading existing model.")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
    else:
        print(f"Using new model {config['name']}.")
        
    # Prepare sliding window dataset: use 10 steps to predict the next step
    window_size = 10
    X = []
    Y = []
    for i in range(len(input_tensor) - window_size):
        X.append(input_tensor[i:i+window_size])
        Y.append(output_tensor[i+window_size])
    X = torch.stack(X)  # shape: (num_samples, window_size, vel_dim)
    Y = torch.stack(Y)  # shape: (num_samples, state_dim)

    train_x, train_y, val_x, val_y = splitTrainVal(X, Y)

    train_dataset = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(val_x, val_y)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        train_loss = 0.0
        for batch_input, batch_output in train_dataloader:
            batch_input = batch_input.to(device)  # shape: (batch, window_size, vel_dim)
            batch_output = batch_output.to(device)  # shape: (batch, state_dim)
            model.train()
            optimizer.zero_grad()
            batch_input_flat = batch_input.view(batch_input.size(0), -1)  # flatten to (batch, window_size * vel_dim)
            batch_pred = model(batch_input_flat)
            loss = criterion(batch_pred, batch_output)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_input, val_output in val_dataloader:
                val_input = val_input.to(device)
                val_output = val_output.to(device)
                val_input_flat = val_input.view(val_input.size(0), -1)  # flatten
                val_pred = model(val_input_flat)
                v_loss = criterion(val_pred, val_output)
                val_loss += v_loss.item()

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


def runInference(model, input_tensor, output_tensor, output_scaler, name="model"):
    window_size = 10
    X = []
    Y = []
    for i in range(len(input_tensor) - window_size):
        X.append(input_tensor[i:i+window_size])
        Y.append(output_tensor[i+window_size])
    X = torch.stack(X)  # shape: (num_samples, window_size, vel_dim)
    Y = torch.stack(Y)  # shape: (num_samples, state_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    X = X.to(device)
    Y = Y.to(device)
    print(X.shape, Y.shape)
    pred_list = []

    with torch.no_grad():
        for i in range(X.shape[0]):
            input = X[i].view(1, -1)  # shape: (1, window_size * vel_dim)
            x_dot_pred = model(input)  # shape: (1, 3)
            pred_list.append(x_dot_pred.unsqueeze(0))  # shape: (1, 1, 3)

    x_pred = torch.cat(pred_list, dim=0)
    x_pred = x_pred.cpu().numpy() # shape: (time_steps, 1, 3)
    x_pred = output_scaler.inverse_transform(x_pred.reshape(-1, 3)).reshape(x_pred.shape)

    Y = Y.cpu().numpy()
    Y = output_scaler.inverse_transform(Y.reshape(-1, 3)).reshape(Y.shape)

    print(x_pred[0])
    print(Y[0])
    # plot the differences
    time_steps = np.arange(x_pred.shape[0])
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    for i in range(3):
        difference = x_pred[:, 0, i]-Y[:, i]
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
    train_input, train_output, test_input, test_output, input_scaler, output_scaler = getInputData(config['data'])

    train_input = torch.tensor(train_input).float()
    train_output = torch.tensor(train_output).float()
    test_input = torch.tensor(test_input).float()
    test_output = torch.tensor(test_output).float()

    print("Sample")
    print("input:", train_input[0:5], "shape:", train_input.shape)
    print("output:", train_output[0:5], "shape:", train_output.shape)
    model = CarPredictor(
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout']
    )

    if mode == "1":
        print("Training model.")
        trainModel(model, train_input, train_output, config['model']['epochs'], config['model']['learning_rate'], config['name'])

    elif mode == "2":
        print("Inference")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
        runInference(model, test_input, test_output, output_scaler, config['name'])

    else:
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
        test_input = [100, 100, 100, 100]
        test_input = input_scaler.transform(np.array(test_input))
        test_input = torch.tensor(test_input).float()
        test_output = [0,0,0,1]
        test_output = torch.tensor(test_output).float()
        # Ensure both tensors have the same batch dimension
        test_input = test_input.unsqueeze(0)  # shape: (1, 4, 1)
        test_output = test_output.unsqueeze(0)  # shape: (1, 4, 1)
        input = torch.cat((test_input, test_output), dim=2)  # concatenate along last dimension
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
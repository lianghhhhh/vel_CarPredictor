# a neural network model using car's target velocity and current velocity to predict delta state
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from carPredictor import CarPredictor
from utils import getInputData, loadConfig
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

def selectMode():
    print("Select mode:")
    print("1. Train")
    print("2. Inference")
    mode = input("Enter mode (1 or 2): ")
    return mode

def trainModel(model, vel_tensor, state_tensor, epochs=100, learning_rate=0.001, name="model"):
    if os.path.exists(f'../{config["name"]}.pth'):
        print(f"Model {config['name']} already exists. Loading existing model.")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
    else:
        print(f"Using new model {config['name']}.")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    vel_tensor = vel_tensor.to(device)
    state_tensor = state_tensor.to(device)
    dataset = TensorDataset(vel_tensor, state_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        train_loss = 0.0
        for i, data in enumerate(dataloader):
            batch_vel = data[0]
            batch_state = data[1]
            model.train()
            optimizer.zero_grad()
            x_dot_pred = model(batch_vel)
            loss = criterion(x_dot_pred, batch_state)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        loss_avg = train_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_avg}')
        writer.add_scalar('Loss/train', loss_avg, epoch+1)

        os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models', name), exist_ok=True)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '..', 'models', name, f'{epoch+1}.pth'))

    writer.close()
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '..', f'{name}.pth'))
    print(f'Model saved as {name}.pth')


def runInference(model, vel_tensor, state_tensor, state_scaler, name="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    vel_tensor = vel_tensor.to(device)
    state_tensor = state_tensor.to(device)
    print(vel_tensor.shape, state_tensor.shape)
    pred_list = []

    with torch.no_grad():
        for i in range(vel_tensor.shape[0]):
            input_vel = vel_tensor[i].unsqueeze(0)  # shape: (1, 3)
            x_dot_pred = model(input_vel)  # shape: (1, 3)
            pred_list.append(x_dot_pred.unsqueeze(0))  # shape: (1, 1, 3)

    x_pred = torch.cat(pred_list, dim=0)
    x_pred = x_pred.cpu().numpy() # shape: (time_steps, 1, 3)
    x_pred = state_scaler.inverse_transform(x_pred.reshape(-1, 3)).reshape(x_pred.shape)

    state_tensor = state_tensor.cpu().numpy()
    state_tensor = state_scaler.inverse_transform(state_tensor.reshape(-1, 3)).reshape(state_tensor.shape)

    print(x_pred[0])
    print(state_tensor[0])

    # plot the differences
    time_steps = np.arange(x_pred.shape[0])
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    for i in range(3):
        difference = x_pred[:, 0, i]-state_tensor[:, i]
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
    train_vels, train_states, test_vels, test_states, vel_scaler, state_scaler = getInputData(config['data'])

    train_vels = torch.tensor(train_vels).float()
    train_states = torch.tensor(train_states).float()
    test_vels = torch.tensor(test_vels).float()
    test_states = torch.tensor(test_states).float()

    print("Sample")
    print("vel:", train_vels[0:5], "shape:", train_vels.shape)
    print("state:", train_states[0:5], "shape:", train_states.shape)

    model = CarPredictor(
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout']
    )

    if mode == "1":
        print("Training model.")
        trainModel(model, train_vels, train_states, config['model']['epochs'], config['model']['learning_rate'], config['name'])

    elif mode == "2":
        print("Inference")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
        runInference(model, test_vels, test_states, state_scaler, config['name'])

    else:
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
        test_vels = [100, 100, 100, 100]
        test_vels = vel_scaler.transform(np.array(test_vels))
        test_vels = torch.tensor(test_vels).float()
        test_states = [0,0,0,1]
        test_states = torch.tensor(test_states).float()
        # Ensure both tensors have the same batch dimension
        test_vels = test_vels.unsqueeze(0)  # shape: (1, 4, 1)
        test_states = test_states.unsqueeze(0)  # shape: (1, 4, 1)
        input = torch.cat((test_vels, test_states), dim=2)  # concatenate along last dimension
        print("Initial input:")
        print(input)
        predict_x = []
        model.eval()
        with torch.no_grad():
            for step in range(50):  # predict 50 steps ahead
                next_x = model(input)
                predict_x.append(next_x.cpu().numpy())
                input = torch.cat((test_vels, next_x), dim=2)

            print("Predicted x over 50 steps:")
            predict_x = np.array(predict_x).squeeze()
            print(predict_x)
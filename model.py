import torch
import torch.nn as nn  # Contains neural network modules
import torch.optim as optim  # Contains optimizers such as SGD and Adam
import torch.nn.functional as F  # Contains activation functions
import os


class LinearQNet(nn.Module):  # Inherit from nn.Module
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()  # Initialize the base class
        self.lin1 = nn.Linear(input_size, hidden_size)  # Linear layer
        self.lin2 = nn.Linear(hidden_size, output_size)  # Linear layer

    # Forward pass
    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.lin1(x))
        return F.relu(self.lin2(x))

    # Save model parameters to file
    def save(self, file_name='model.pth') -> None:
        path = "./models"

        if not os.path.exists(path):
            os.makedirs(path)

        file_name = os.path.join(path, file_name)

        # state_dict() returns a dictionary containing the model's parameters
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, learn_rate, gamma) -> None:
        self.lr = learn_rate  # learning rate
        self.gamma = gamma
        self.model = model  # neural network
        self.optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        self.criterion = nn.MSELoss()  # loss function

    def train_step(self, state, action, reward, next_state, status) -> None:
        # Convert to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Check length of state
        if len(state.shape) == 1:
            # Add a dimension to state
            state = torch.unsqueeze(state, 0)  # unsqueeze() adds a dimension
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            action = torch.unsqueeze(action, 0)
            status = (status, )

        # Get the Q values for the current state
        # Q(s, a)
        prediction = self.model(state)

        # Get the Q values for the next state
        # Q_new = r + y * max(Q(s', a'))
        target = prediction.clone()  # Clone the tensor

        for index in range(len(status)):
            Q_new = reward[index]

            if not status[index]:
                Q_new = reward[index] + self.gamma * \
                    torch.max(self.model(next_state[index]))

            target[index][torch.argmax(action).item()] = Q_new

        # Calculate loss
        self.optimizer.zero_grad()  # Zero the gradients
        loss = self.criterion(target, prediction)
        loss.backward()  # Backpropagation

        self.optimizer.step()  # Update the weights

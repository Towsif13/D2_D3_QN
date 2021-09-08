import torch
import torch.nn as nn
import torch.nn.functional as F


class DuellingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64, fc3_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuellingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 64)
        self.valuefc = nn.Linear(64, 64)
        self.valuefc2 = nn.Linear(64, 32)
        self.advfc = nn.Linear(64, 64)
        self.advfc2 = nn.Linear(64, 32)
        self.value = nn.Linear(32, 1)
        self.advantage = nn.Linear(32, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        value_x = F.relu(self.valuefc(x))
        value_x = F.relu(self.valuefc2(value_x))
        adv_x = F.relu(self.advfc(x))
        adv_x = F.relu(self.advfc2(adv_x))
        value = self.value(value_x)
        advantage = self.advantage(adv_x)
        Q = value+advantage-torch.max(advantage)

        return value, advantage, Q

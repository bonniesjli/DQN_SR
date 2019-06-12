import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        phi = F.relu(self.fc1(state))
        phi = F.normalize(phi, p=2, dim=1)
        x = F.relu(self.fc2(phi))
        x = self.fc3(x)
        return phi, x
    
class SRNetwork(nn.Module):
    """Successor Representation Model"""
    
    def __init__(self, input_size = 128, output_size = 128, fc1_units = 64):
        super(SRNetwork, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, output_size)
        
    def forward(self, phi):
        x = phi.detach()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
class PredNetwork(nn.Module):
    """Next state prediction model"""
    def __init__(self, state_size, action_size, fc1_units = 64, fc2_units = 128):
        super(PredNetwork, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.input_size = 128
        self.output_size = state_size
        
        self.fc_a = nn.Linear(action_size, fc1_units)
        
        self.fc1 = nn.Linear(self.input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, self.output_size)
        
    def forward(self, phi, action):
        action_ = action.float().to(device)
        p = F.relu(self.fc1(phi))
        a = self.fc_a(action_)
        # element wise multiplication
        x = F. relu(self.fc2(p * a))
        x = self.fc3(x)
        return x
        
        
        
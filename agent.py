import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, SRNetwork, PredNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(3e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99           # discount factor
TAU = 5e-3              # for soft update of target parameters (default: 1e-3)
LR = 2.5e-4               # RMS learning rate (Paper: 2.5e-4)
UPDATE_EVERY = 4        # how often to update the network

BETA = 0.025               # coefficient for intrinsic reward
W_TD = 1 
W_SR = 1000
W_PRED = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.q_optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=LR, eps=1e-5)
        
        # Succesor Representation Network
        self.snetwork = SRNetwork().to(device)
        self.snetwork_target = SRNetwork().to(device)
        self.s_optimizer = optim.RMSprop(self.snetwork.parameters(), lr=LR, eps=1e-5)
        
        self.pnetwork = PredNetwork(state_size, 1).to(device)
        self.p_optimizer = optim.RMSprop(self.pnetwork.parameters(), lr=LR, eps=1e-5)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        
            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
                    # print ("ONE LEARNING BATCH COMPLETE")

    def parallel_act(self, states, eps = 0.):
        actions = []
        intrinsic_rewards = []
        for state in states:
            action, r_int = self.act(state, eps)
            actions.append(action)
            intrinsic_rewards.append(r_int)
        return np.asarray(actions), intrinsic_rewards
    
    def act(self, state, eps):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        self.snetwork.eval()
        with torch.no_grad():
            phi, action_values = self.qnetwork_local(state)
            psi = self.snetwork_target(phi)
        r_int = 1/np.linalg.norm(psi.detach())
        self.qnetwork_local.train()
        self.snetwork.train()
        # print ("Intrinsic reward", r_int)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()), r_int
        else:
            return random.choice(np.arange(self.action_size)), r_int

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        state, action, reward, next_state, done, indices, weights = experiences
        # loss = torch.Tensor([0])
        # for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            
        # ------------------------------------------------------------------------------------------
        # Get max predicted Q values (for next states) from target model
        phi_next, next_action_values = self.qnetwork_target(next_state)
        Q_target_next = next_action_values.detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states 
        Q_target = reward + (gamma * Q_target_next * (1 - done)) 

        # Get expected Q values from local model
        phi, v = self.qnetwork_local(state)
        Q_expected = v.gather(1, action)

        # Compute TD loss
        L_td = (Q_expected - Q_target).pow(2)
        # print ("TD loss length:", len(L_td), L_td[0])

        # ------------------------------------------------------------------------------------------
        # compute state representation
        # phi, v = self.qnetwork_local(state)
        # phi_next, next_action_values = self.qnetwork_target(next_state)

        # compute succecer representation 
        # L_sr = phi[target_network] + GAMMA * psi_next[target_network] - psi[local_network]
        phi_, _ = self.qnetwork_target(state)
        psi = self.snetwork(phi)
        psi_next = self.snetwork_target(phi_next).detach()
        
        psi_target = phi_.detach() + GAMMA * psi_next
        # L_sr = F.mse_loss(psi, psi_target)
        L_sr = torch.sum((psi - psi_target).pow(2), dim = -1).unsqueeze(1)
        # print ("SR loss length:", len(L_sr), L_sr[0])
  
        
        # -----------------------------------------------------------------------------------------
        # compute predicted next state
        s_pred = self.pnetwork(phi, action)
        # compute prediction loss
        L_pred = torch.sum((s_pred - next_state).pow(2), dim = -1).unsqueeze(1)
        # print ("PR loss length: ", len(L_pred), L_pred[0])

        # -----------------------------------------------------------------------------------------
        # print ("\rTD error: {}\tSR error: {}\tPred error: {}".format(L_td, L_sr, L_pred))
        loss = W_TD * L_td + W_SR * L_sr + W_PRED * L_pred
        prios = loss + 1e-5
        
        loss = (loss * weights.unsqueeze(1)).mean()
        
        # Minimize the loss
        self.q_optimizer.zero_grad()
        self.s_optimizer.zero_grad()
        self.p_optimizer.zero_grad()
        
        loss.backward()
        
        self.q_optimizer.step()
        self.s_optimizer.step()
        self.p_optimizer.step()
        
        
        # ------------------- update replay buffer ------------------- #
        self.memory.update_priorities(indices, prios.data.cpu().numpy())  

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)   
        self.soft_update(self.snetwork, self.snetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        # self.memory = deque(maxlen=buffer_size)  
        self.buffer_size = buffer_size
        self.memory = []
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
        self.prob_alpha = 0.6
        self.position = 0
        self.priorities = np.zeros((buffer_size, ), dtype = np.float32)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        # self.memory.append(e)
        
        max_prio = self.priorities.max() if self.memory else 1.0
        
        if self.__len__() < self.buffer_size:
            self.memory.append(e)
        if self.__len__() == self.buffer_size:
            self.memory[self.position] = e
            
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self, beta = 0.4):
        """Sample a batch of experiences from memory accordingly to priorities."""
        # experiences = random.sample(self.memory, k=self.batch_size)
        
        if self.__len__() == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), self.batch_size, p = probs)
        experiences = [self.memory[idx] for idx in indices]
        
        total = self.__len__()
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype = np.float32)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        weights = torch.from_numpy(weights).float().to(device)
  
        return (states, actions, rewards, next_states, dones, indices, weights)
    
    def update_priorities(self, batch_indices, batch_priorities):
            for idx, prio in zip(batch_indices, batch_priorities):
                self.priorities[idx] = prio
                
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
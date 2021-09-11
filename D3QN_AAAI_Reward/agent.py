import numpy as np
import random
from collections import namedtuple, deque

from model import DuellingQNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BUFFER_SIZE_2 = 50000 # replay buffer size
BATCH_SIZE = 16         # minibatch size
BATCH_SIZE_2 = 32 
GAMMA = 0.99            # discount factor
TAU = 0.01              # for soft update of target parameters
LR = 0.001               # learning rate


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alpha_start=0.05
alpha_end= 1
alpha_rate=0.995

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
        self.qnetwork_local = DuellingQNetwork(
            state_size, action_size, seed).to(device)
        self.qnetwork_target = DuellingQNetwork(
            state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.q = np.zeros((state_size,action_size))
        # Replay memory
        self.memory1 = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.memory2 = ReplayBuffer(action_size, BUFFER_SIZE_2, BATCH_SIZE, seed)
        self.i = 0
        self.lenmem1 = 0
        self.lenmem2 = 0
        self.criterion = nn.MSELoss()
        self.counter = 1000

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory1.add(state, action, reward, next_state, done)
      
        self.lenmem1 += 1
 
        self.experience_replay()
    
    def experience_replay(self):
        
        alpha = alpha_start 

        if random.random() > alpha:

            if len(self.memory1) > BATCH_SIZE:
                experiences = self.memory1.sample()
                self.learn(experiences, GAMMA)

        else:
            
            if len(self.memory2) > BATCH_SIZE_2:
                experiences = self.memory2.sample()
                self.learn(experiences, GAMMA)
         
        alpha = min(alpha_end, alpha_rate*alpha)     

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            value, advantage, Q = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(Q.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        target_values, target_advantages, target_Qs = self.qnetwork_target(
            next_states)
        _, _, Qs = self.qnetwork_local(states)
        _, _, next_state_Qs = self.qnetwork_local(next_states)
        predicted_actions = torch.argmax(next_state_Qs, 1)
        predicted_actions = predicted_actions.reshape(-1, 1)
        y = rewards+gamma*target_Qs.gather(1, predicted_actions)*(1-dones)
       
        self.rewards_np = rewards.detach().cpu().numpy()

        avg_reward = np.sum(self.rewards_np) / (BATCH_SIZE)
        
        if(avg_reward > -50):           
            self.memory2.add(states.cpu(), actions.cpu(), rewards.cpu(), next_states.cpu(), dones.cpu())
            self.lenmem2 += 1
          
        if (self.lenmem1 >= BUFFER_SIZE):
            if (self.i % self.counter  == 0):
                pstate, paction, preward, pnext_state, pdone = self.memory1.popleft()
                self.memory2.add(pstate, paction, preward, pnext_state, pdone)
        loss = F.mse_loss(y, Qs.gather(1, actions))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.i+=1

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

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
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


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
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
        
    def popleft(self):
        """retutn the 1st experience."""
        return self.memory.popleft()
    

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

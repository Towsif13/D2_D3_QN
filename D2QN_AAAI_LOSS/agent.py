import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

BUFFER_SIZE = int(1e5)  # replay buffer size
BUFFER_SIZE_2 = 50000 # replay buffer size
BATCH_SIZE = 16         # minibatch size
BATCH_SIZE_2 = 32   # FOR NEW MEMORY
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

alpha_start=0.05
alpha_end= 1
alpha_rate=0.995

class Agent():

    def __init__(self, state_size, action_size, seed, l, c):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.q = np.zeros((state_size,action_size))
        # Replay memory
        self.memory1 = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.memory2 = ReplayBuffer(action_size, BUFFER_SIZE_2, BATCH_SIZE, seed)
        self.i = 0
        self.lenmem1 = 0
        self.lenmem2 = 0
    
#         ap = argparse.ArgumentParser()
#         ap.add_argument("-l","--loss", required=True,
# 	        help="path to input loss threshold", type=int)
#         ap.add_argument("-c","--counter", required=True,
# 	        help="path to input pop counter", type=int)
#         args = vars(ap.parse_args())
#         #args = ap.parse_args(argv)
#         self.loss_ = args["loss"]
#         self.counter = args["counter"]
        self.loss_ = l
        self.counter = c


    def step(self, state, action, reward, next_state, done):
        
        self.memory1.add(state, action, reward, next_state, done)
      
        self.lenmem1 += 1
 
        self.experience_replay()
    
    # CHOOSE TO REPLAY
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
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
         
        states, actions, rewards, next_states, dones = experiences
        
        Q_targets_next = self.qnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        q_target_np = Q_targets.detach().cpu().numpy()
        q_expected_np = Q_expected.detach().cpu().numpy()
        self.q = abs(q_target_np - q_expected_np)
        
        if(np.max(self.q) < self.loss_):            
            self.memory2.add(states.cpu(), actions.cpu(), rewards.cpu(), next_states.cpu(), dones.cpu())
            self.lenmem2 += 1
          
        if (self.lenmem1 >= BUFFER_SIZE):
            if (self.i % self.counter  == 0):
                pstate, paction, preward, pnext_state, pdone = self.memory1.popleft()
                self.memory2.add(pstate, paction, preward, pnext_state, pdone)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
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


# import numpy as np
# import random
# from collections import namedtuple, deque

# from model import QNetwork

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# BUFFER_SIZE = int(1e5)  # replay buffer size
# BUFFER_SIZE_2 = 50000 # replay buffer size
# BATCH_SIZE = 16         # minibatch size
# BATCH_SIZE_2 = 32   # FOR NEW MEMORY
# GAMMA = 0.99            # discount factor
# TAU = 1e-3              # for soft update of target parameters
# LR = 5e-4 

# alpha_start=0.05
# alpha_end= 1
# alpha_rate=0.995

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class Agent():
#     """Interacts with and learns from the environment."""

#     def __init__(self, state_size, action_size, seed):
#         """Initialize an Agent object.

#         Params
#         ======
#             state_size (int): dimension of each state
#             action_size (int): dimension of each action
#             seed (int): random seed
#         """
#         self.state_size = state_size
#         self.action_size = action_size
#         self.seed = random.seed(seed)

#         # Q-Network
#         self.qnetwork_local = QNetwork(
#             state_size, action_size, seed).to(device)
#         self.qnetwork_target = QNetwork(
#             state_size, action_size, seed).to(device)
#         self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

#         self.q = np.zeros((state_size,action_size))
#         # Replay memory
#         self.memory1 = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
#         self.memory2 = ReplayBuffer(action_size, BUFFER_SIZE_2, BATCH_SIZE, seed)
#         self.i = 0
#         self.lenmem1 = 0
#         self.lenmem2 = 0
#         self.criterion = nn.MSELoss()
#         self.counter = 1000

#     def step(self, state, action, reward, next_state, done):
#         # Save experience in replay memory
#         self.memory1.add(state, action, reward, next_state, done)
#         self.lenmem1 += 1
 
#         self.experience_replay()
    
#     def experience_replay(self):
        
#         alpha = alpha_start 

#         if random.random() > alpha:

#             if len(self.memory1) > BATCH_SIZE:
#                 experiences = self.memory1.sample()
#                 self.learn(experiences, GAMMA)

#         else:
            
#             if len(self.memory2) > BATCH_SIZE_2:
#                 experiences = self.memory2.sample()
#                 self.learn(experiences, GAMMA)
         
#         alpha = min(alpha_end, alpha_rate*alpha)        


#     def act(self, state, eps=0.):
#         """Returns actions for given state as per current policy.

#         Params
#         ======
#             state (array_like): current state
#             eps (float): epsilon, for epsilon-greedy action selection
#         """
#         state = torch.from_numpy(state).float().unsqueeze(0).to(device)
#         self.qnetwork_local.eval()
#         with torch.no_grad():
#             action_values = self.qnetwork_local(state)
#         self.qnetwork_local.train()

#         # Epsilon-greedy action selection
#         if random.random() > eps:
#             return np.argmax(action_values.cpu().data.numpy())
#         else:
#             return random.choice(np.arange(self.action_size))

#     def learn(self, experiences, gamma):
#         """Update value parameters using given batch of experience tuples.

#         Params
#         ======
#             experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
#             gamma (float): discount factor
#         """
#         states, actions, rewards, next_states, dones = experiences

#         predicted_actions = torch.argmax(self.qnetwork_local(next_states), 1)
#         predicted_actions = predicted_actions.reshape(-1, 1)

#         maxQ = self.qnetwork_target(next_states).gather(1, predicted_actions)
#         y = rewards+gamma*maxQ*(1-dones)
#         Q_exp = self.qnetwork_local(states).gather(1, actions)
#         y_np = y.detach().cpu().numpy()
#         Q_exp_np = Q_exp.detach().cpu().numpy()
        
#         self.q = abs(y_np - Q_exp_np)
        
#         if(np.max(self.q) < 5):            
#             self.memory2.add(states.cpu(), actions.cpu(), rewards.cpu(), next_states.cpu(), dones.cpu())
#             self.lenmem2 += 1
          
#         if (self.lenmem1 >= BUFFER_SIZE):
#             if (self.i % self.counter  == 0):
#                 pstate, paction, preward, pnext_state, pdone = self.memory1.popleft()
#                 self.memory2.add(pstate, paction, preward, pnext_state, pdone)
        
#         loss = F.mse_loss(y, Q_exp)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         self.i+=1

#         # ------------------- update target network ------------------- #
#         self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

#     def soft_update(self, local_model, target_model, tau):
#         """Soft update model parameters.
#         θ_target = τ*θ_local + (1 - τ)*θ_target

#         Params
#         ======
#             local_model (PyTorch model): weights will be copied from
#             target_model (PyTorch model): weights will be copied to
#             tau (float): interpolation parameter 
#         """
#         for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#             target_param.data.copy_(
#                 tau*local_param.data + (1.0-tau)*target_param.data)


# class ReplayBuffer:
#     """Fixed-size buffer to store experience tuples."""

#     def __init__(self, action_size, buffer_size, batch_size, seed):
#         """Initialize a ReplayBuffer object.

#         Params
#         ======
#             action_size (int): dimension of each action
#             buffer_size (int): maximum size of buffer
#             batch_size (int): size of each training batch
#             seed (int): random seed
#         """
#         self.action_size = action_size
#         self.memory = deque(maxlen=buffer_size)
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience", field_names=[
#                                      "state", "action", "reward", "next_state", "done"])
#         self.seed = random.seed(seed)

#     def add(self, state, action, reward, next_state, done):
#         """Add a new experience to memory."""
#         e = self.experience(state, action, reward, next_state, done)
#         self.memory.append(e)
        
#     def popleft(self):
#         """retutn the 1st experience."""
#         return self.memory.popleft()

#     def sample(self):
#         """Randomly sample a batch of experiences from memory."""
#         experiences = random.sample(self.memory, k=self.batch_size)

#         states = torch.from_numpy(
#             np.vstack([e.state for e in experiences if e is not None])).float().to(device)
#         actions = torch.from_numpy(
#             np.vstack([e.action for e in experiences if e is not None])).long().to(device)
#         rewards = torch.from_numpy(
#             np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
#         next_states = torch.from_numpy(np.vstack(
#             [e.next_state for e in experiences if e is not None])).float().to(device)
#         dones = torch.from_numpy(np.vstack(
#             [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

#         return (states, actions, rewards, next_states, dones)

#     def __len__(self):
#         """Return the current size of internal memory."""
#         return len(self.memory)

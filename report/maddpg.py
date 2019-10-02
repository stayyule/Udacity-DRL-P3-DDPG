import random
import copy
from collections import namedtuple, deque

import torch.optim as optim
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

BUFFER_SIZE = 2 ** 14  # replay buffer size
BATCH_SIZE = 1000         # minibatch size
GAMMA = 0.95            # discount factor
TAU_ACTOR = 1e-2              # for soft update of target parameters
TAU_CRITIC = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
LEARN_EVERY = 20        # learning timestep interval
LEARN_NUM = 20          # number of learning passes
LEARN_AFTER = 6000
SEED = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # TEST
# BUFFER_SIZE = 32
# BATCH_SIZE = 8         # minibatch size
# LEARN_EVERY = 1       # learning timestep interval
# LEARN_NUM = 1           # number of learning passes
# LEARN_AFTER = 0


def hidden_init(layer):
	fan_in = layer.weight.data.size()[0]
	lim = 1. / np.sqrt(fan_in)
	return (-lim, lim)


class Actor(nn.Module):
	"""Actor (Policy) Model."""
	
	def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
		"""Initialize parameters and build model.
		Params
		======
			state_size (int): Dimension of each state
			action_size (int): Dimension of each action
			seed (int): Random seed
			fc1_units (int): Number of nodes in first hidden layer
			fc2_units (int): Number of nodes in second hidden layer
		"""
		super(Actor, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, fc1_units)
		self.bn1 = nn.BatchNorm1d(fc1_units)
		self.fc2 = nn.Linear(fc1_units, fc2_units)
		self.fc3 = nn.Linear(fc2_units, action_size)
		self.reset_parameters()
	
	def reset_parameters(self):
		self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
		self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
		self.fc3.weight.data.uniform_(-3e-3, 3e-3)
	
	def forward(self, state):
		"""Build an actor (policy) network that maps states -> actions."""
		x = F.relu(self.bn1(self.fc1(state)))
		x = F.relu(self.fc2(x))
		return torch.tanh(self.fc3(x))


class Critic(nn.Module):
	"""Critic (Value) Model."""
	
	def __init__(self, state_size, action_size, seed, fcsa_units=300, fc_units=300):
		"""Initialize parameters and build model.
		Params
		======
			state_size (int): Dimension of each state
			action_size (int): Dimension of each action
			seed (int): Random seed
			fcs1_units (int): Number of nodes in the first hidden layer
			fc2_units (int): Number of nodes in the second hidden layer
		"""
		super(Critic, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fcs = nn.Linear(state_size, fcsa_units)
		self.fca = nn.Linear(action_size, fcsa_units)
		self.bn1 = nn.BatchNorm1d(fcsa_units)
		self.bn2 = nn.BatchNorm1d(fcsa_units)
		self.fc2 = nn.Linear(fcsa_units * 2, fc_units)
		self.fc3 = nn.Linear(fc_units, fc_units)
		self.fc4 = nn.Linear(fc_units, 1)
		self.reset_parameters()
	
	def reset_parameters(self):
		self.fcs.weight.data.uniform_(*hidden_init(self.fcs))
		self.fca.weight.data.uniform_(*hidden_init(self.fca))
		self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
		self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
		self.fc4.weight.data.uniform_(-3e-3, 3e-3)
	
	def forward(self, states, actions):
		"""Build a critic (value) network that maps (state, action) pairs -> Q-values."""
		s = self.bn1(self.fcs(states))
		a = self.bn2(self.fca(actions))
		x = F.relu(self.fc2(torch.cat((s, a), dim=1)))
		x = F.relu(self.fc3(x))
		return self.fc4(x)


class OUNoise:
	"""Ornstein-Uhlenbeck process."""
	
	def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
		"""Initialize parameters and noise process."""
		self.mu = mu * np.ones(size)
		self.theta = theta
		self.sigma = sigma
		self.seed = random.seed(seed)
		self.reset()
	
	def reset(self):
		"""Reset the internal state (= noise) to mean (mu)."""
		self.state = copy.copy(self.mu)
	
	def sample(self):
		"""Update internal state and return it as a noise sample."""
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
		self.state = x + dx
		return self.state


class SumTree:

    def __init__(self, capacity):
        """
        :param capacity: capacity of saving data which should be equal to 2 ** n
        """
        self.capacity = capacity
        # node value (priority)
        self.tree = np.zeros(2 * capacity - 1)
        # data saved in node
        self.data = np.zeros(capacity, dtype=object)
        # index in the tree to overwrite
        self.write = 0
        # non-empty data count
        self.n_entries = 0

    def _propagate(self, idx, change):
        """
        update current value to the root node
        :param idx: index is current node(leaf)
        :param change: changed value
        :return:
        """
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        find sample on leaf node
        :param idx: idx is current node(parent node if leaf exists)
        :param s: node value(priority)
        :return:
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """
        sum value of the whole tree
        :return:
        """
        return self.tree[0]

    def add(self, p, data):
        """
        store priority and sample
        :param p: priority(node value)
        :param data: data to save
        :return:
        """
        # all value will be saved in bottom layer of the tree
        idx = self.write + self.capacity - 1
        if self.n_entries > 0:
            p_ave = self.total() / self.n_entries
        else:
            p_ave = 0
        print('p_ave', p_ave)
        # overwrite the p below average only
        while p < p_ave:
            self.write += 1
            if self.write >= self.capacity:
                self.write = 0

            idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        # the data will be overwrite when data number is larger than capacity
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """
        update priority to whole tree
        :param idx: node to update
        :param p: priority(node value)
        :return:
        """
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """
        get priority and sample
        :param s: target priority equal or less than tree.total
        :return: index of sample, priority of the sample, data of the sample
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[dataIdx]


# prioritized experience replay memory
class PERMemory:
	e = 0.01
	a = 0.6
	beta = 0.4
	beta_increment_per_sampling = 0.001
	
	def __init__(self, capacity, random_seed):
		self.tree = SumTree(capacity)
		self.capacity = capacity
		self.seed = random.seed(random_seed)
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
	
	def _get_priority(self, error):
		return (np.abs(error) + self.e) ** self.a
	
	def add(self, state, action, reward, next_state, done, error):
		e = self.experience(state, action, reward, next_state, done)
		p = self._get_priority(error)
		self.tree.add(p, e)
	
	#         print(self.tree.tree)
	
	def sample(self, n):
		"""
		get samples of count n
		:param n: sample count
		:return: data, indexes, weights
		"""
		batch = []
		idxs = []
		# priority segment
		segment = self.tree.total() / n
		priorities = []
		
		self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
		
		for i in range(n):
			a = segment * i
			b = segment * (i + 1)
			
			s = random.uniform(a, b)
			(idx, p, data) = self.tree.get(s)
			priorities.append(p)
			batch.append(data)
			idxs.append(idx)
		
		sampling_probabilities = priorities / self.tree.total()
		#         print('is_weight_0', sampling_probabilities * self.tree.n_entries)
		# importance sampling weight
		is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
		if is_weights.max() != 0:
			is_weights /= is_weights.max()
		#         print('idxs', idxs)
		#         print('is_weights', is_weights)
		
		states = torch.from_numpy(np.vstack([e.state for e in batch if e is not None])).float().to(device)
		actions = torch.from_numpy(np.vstack([e.action for e in batch if e is not None])).float().to(device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in batch if e is not None])).float().to(device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in batch if e is not None])).float().to(device)
		dones = torch.from_numpy(np.vstack([e.done for e in batch if e is not None]).astype(np.uint8)).float().to(
			device)
		
		return states, actions, rewards, next_states, dones, idxs, is_weights
	
	def update(self, idx, error):
		"""
		update tree priority with TD-error
		:param idx: index of sample
		:param error: TR-error
		:return:
		"""
		p = self._get_priority(error)
		self.tree.update(idx, p)
	
	def __len__(self):
		return self.tree.n_entries


class Agent:
	"""Interacts with and learns from the environment."""
	
	def __init__(self, state_size, action_size, num_agents, random_seed):
		"""Initialize an Agent object.

		Params
		======
			state_size (int): dimension of each state
			action_size (int): dimension of each action
			random_seed (int): random seed
		"""
		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(random_seed)
		self.num_agents = num_agents
		
		# Actor Network (w/ Target Network)
		# turn model to list, one for each agent
		self.actor_local = [Actor(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
		self.actor_target = [Actor(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
		self.actor_optimizer = [optim.Adam(x.parameters(), lr=LR_ACTOR) for x in self.actor_local]
		
		# Critic Network (w/ Target Network)
		self.critic_local = [Critic(state_size * num_agents, action_size * num_agents, random_seed).to(device) for _ in
		                     range(num_agents)]
		self.critic_target = [Critic(state_size * num_agents, action_size * num_agents, random_seed).to(device) for _ in
		                      range(num_agents)]
		self.critic_optimizer = [optim.Adam(x.parameters(), lr=LR_CRITIC, weight_decay=0) for x in self.critic_local]
		
		# Noise process for 400 * 300 actor network
		self.noise_h = OUNoise(1, random_seed, mu=-0.25, theta=0.2, sigma=0.1)
		self.noise_v = OUNoise(1, random_seed, mu=-0., theta=0.15, sigma=0.15)
		
		self.eps = 1.0
		self.eps_end = 0.01
		self.eps_decay = 1e-5
		
		# Replay memory
		self.memory = PERMemory(BUFFER_SIZE, random_seed)
		
		# Debug
		self.debug = False
	
	def debugon(self):
		self.debug = True
	
	def debugoff(self):
		self.debug = False
	
	def deprint(self, *content):
		if self.debug:
			print(*content)
	
	def calc_td(self, state, action, reward, next_state, done, gamma):
		actions_all = torch.tensor(()).to(device)
		for j in range(self.num_agents):
			actions_all = torch.cat((actions_all, torch.FloatTensor([action[j]]).to(device)), dim=1)
		# print('actions_all', actions_all)
		
		states_all = torch.tensor(()).to(device)
		for j in range(self.num_agents):
			states_all = torch.cat((states_all, torch.FloatTensor([state[j]]).to(device)), dim=1)
		# print('states_all', states_all)
		
		next_states_all = torch.tensor(()).to(device)
		for j in range(self.num_agents):
			next_states_all = torch.cat((next_states_all, torch.FloatTensor([next_state[j]]).to(device)), dim=1)
		# print('next_states_all', next_states_all)
		
		td_errors = 0
		for i in range(self.num_agents):
			# Get predicted next-state actions and Q values from target models
			self.actor_target[i].eval()
			actions_next = self.actor_target[i](torch.FloatTensor([next_state[i]]).to(device))
			self.actor_target[i].train()
			
			actions_next_all = torch.tensor(())
			for j in range(self.num_agents):
				if i < j:
					actions_next_all = torch.cat((actions_next, torch.FloatTensor([action[j]]).to(device)), dim=1)
				if j < i:
					actions_next_all = torch.cat((torch.FloatTensor([action[j]]).to(device), actions_next), dim=1)
			
			self.critic_target[i].eval()
			Q_targets_next = self.critic_target[i](next_states_all, actions_next_all)
			self.critic_target[i].train()
			
			Q_targets = reward[i] + (gamma * Q_targets_next * (1 - done[i]))
			
			# Compute critic loss
			self.critic_local[i].eval()
			with torch.no_grad():
				Q_expected = self.critic_local[i](states_all, actions_all)
				td_error = torch.abs(Q_expected - Q_targets)
			self.critic_local[i].train()
			#             .detach().numpy()
			td_errors += td_error
		
		return td_errors.view(-1).cpu().data.numpy()[0]
	
	def step(self, state, action, reward, next_state, done, timestep):
		"""Save experience in replay memory, and use random sample from buffer to learn."""
		#         print('states', state)
		#         print('actions', action)
		#         print('rewards', reward)
		#         print('next_states', next_state)
		#         print('dones', done)
		# calculate td-error
		td_error = self.calc_td(state, action, reward, next_state, done, GAMMA)
		# self.deprint('reward', reward, 'td_error', td_error)
		self.memory.add(state, action, reward, next_state, done, td_error)
		self.eps = max(self.eps_end, self.eps - self.eps_decay)  # decrease epsilon
		
		# Learn at defined interval, if enough samples are available in memory
		if len(self.memory) > BATCH_SIZE and timestep % LEARN_EVERY == 0 and timestep > LEARN_AFTER:
			c_loss = []
			a_loss = []
			experiences = self.memory.sample(BATCH_SIZE)
			for _ in range(LEARN_NUM):
				#                 experiences = self.memory.sample(BATCH_SIZE)
				c, a = self.learn(experiences, GAMMA)
				c_loss.append(c)
				a_loss.append(a)
			
			return c_loss, a_loss
		else:
			return None, None
	
	def act(self, state, add_noise=True):
		"""Returns actions for given state as per current policy."""
		state = torch.from_numpy(state).float().to(device)
		
		actions = []
		for i in range(self.num_agents):
			self.actor_local[i].eval()
			with torch.no_grad():
				action = self.actor_local[i](state[i].view(-1, self.state_size)).cpu().data.numpy()
			self.actor_local[i].train()
			if add_noise:
				# self.deprint('action', action)
				noise = np.array([self.noise_h.sample(), self.noise_v.sample()]).reshape(-1, self.action_size)
				# self.deprint('noise', noise)
				action += self.eps * noise
				# self.deprint('action_with_noise', action)
			
			actions.append(action)
		actions = np.vstack(actions)
		
		return np.clip(np.around(actions, decimals=2), -1, 1)
	
	def reset(self):
		self.noise_h.reset()
		self.noise_v.reset()
	
	def learn(self, experiences, gamma):
		"""Update policy and value parameters using given batch of experience tuples.
		Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
		where:
			actor_target(state) -> action
			critic_target(state, action) -> Q-value

		Params
		======
			experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
			gamma (float): discount factor
		"""
		states, actions, rewards, next_states, dones, idxs, is_weights = experiences
		
		struct_states = states.reshape(-1, self.num_agents, self.state_size)
		struct_actions = actions.reshape(-1, self.num_agents, self.action_size)
		struct_next_states = next_states.reshape(-1, self.num_agents, self.state_size)
		# <a1, a2, ... an>
		#         print('===LEARN===')
		#         print('states', states)
		#         print('actions', actions)
		#         print('rewards', rewards)
		#         print('next_states', next_states)
		#         print('dones', dones)
		#         print('struct_states', struct_states)
		#         print('struct_actions', struct_actions)
		#         print('struct_next_states', struct_next_states)
		
		c_loss = np.zeros(self.num_agents)
		a_loss = np.zeros(self.num_agents)
		
		actions_all = torch.tensor(()).to(device)
		for j in range(self.num_agents):
			actions_all = torch.cat((actions_all, struct_actions[:, j, :].to(device)), dim=1)
		# print('actions_all', actions_all)
		
		states_all = torch.tensor(()).to(device)
		for j in range(self.num_agents):
			states_all = torch.cat((states_all, struct_states[:, j, :].to(device)), dim=1)
		# print('states_all', states_all)
		
		next_states_all = torch.tensor(()).to(device)
		for j in range(self.num_agents):
			next_states_all = torch.cat((next_states_all, struct_next_states[:, j, :].to(device)), dim=1)
		# print('next_states_all', next_states_all)
		
		td_error_all = torch.tensor(()).to(device)
		for i in range(self.num_agents):
			# ---------------------------- update critic ---------------------------- #
			# Get predicted next-state actions and Q values from target models
			actions_next = self.actor_target[i](struct_next_states[:, i, :])
			# print('actions_next', actions_next)
			actions_next_all = torch.tensor(())
			for j in range(self.num_agents):
				if i < j:
					actions_next_all = torch.cat((actions_next, struct_actions[:, j, :]), dim=1)
				if j < i:
					actions_next_all = torch.cat((struct_actions[:, j, :], actions_next), dim=1)
			# print('next_states[:,{},:]'.format(i), struct_next_states[:, i, :])
			# print('actions_next_all', actions_next_all)
			Q_targets_next = self.critic_target[i](next_states_all, actions_next_all)
			# print('Q_targets_next', Q_targets_next)
			Q_targets = rewards[:, i].view(-1, 1) + (gamma * Q_targets_next * (1 - dones[:, i].view(-1, 1)))
			
			# print('rewards[:,{}]'.format(i), rewards[:, i])
			# print('Q_targets', Q_targets)
			
			# Compute critic loss
			Q_expected = self.critic_local[i](states_all, actions_all)
			# print('Q_expected', Q_expected, 'Q_targets', Q_targets)
			critic_loss = F.mse_loss(Q_expected, Q_targets)
			#             critic_loss = (torch.FloatTensor(is_weights).to(device) * mse).mean()
			
			td_errors = torch.abs(Q_expected - Q_targets).to(device).detach()
			td_error_all = torch.cat((td_error_all, td_errors), dim=1)
			
			# Minimize the loss
			self.critic_optimizer[i].zero_grad()
			critic_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.critic_local[i].parameters(), 1)
			self.critic_optimizer[i].step()
			
			# ---------------------------- update actor ---------------------------- #
			actions_pred = self.actor_local[i](struct_states[:, i, :])
			actions_pred_all = torch.tensor(())
			for j in range(self.num_agents):
				if i < j:
					actions_pred_all = torch.cat((actions_pred, struct_actions[:, j, :]), dim=1)
				if j < i:
					actions_pred_all = torch.cat((struct_actions[:, j, :], actions_pred), dim=1)
			
			actor_loss = -self.critic_local[i](states_all, actions_pred_all).mean()
			
			# Minimize the loss
			self.actor_optimizer[i].zero_grad()
			actor_loss.backward()
			#             torch.nn.utils.clip_grad_norm_(self.actor_local[i].parameters(), 1)
			self.actor_optimizer[i].step()
			
			c_loss[i] = critic_loss.cpu().data.numpy()
			a_loss[i] = actor_loss.cpu().data.numpy()
		
		td_error_all = np.sum(td_error_all.cpu().data.numpy(), axis=1)
		#         print('rewards', rewards, 'td_error_all', td_error_all, 'is_weights', is_weights)
		
		for idx, error in zip(idxs, td_error_all):
			self.memory.update(idx, error)
		
		for i in range(self.num_agents):
			# ----------------------- update target networks ----------------------- #
			self.soft_update(self.critic_local[i], self.critic_target[i], TAU_CRITIC)
			self.soft_update(self.actor_local[i], self.actor_target[i], TAU_ACTOR)
		
		return c_loss, a_loss
	
	def soft_update(self, local_model, target_model, tau):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target

		Params
		======
			local_model: PyTorch model (weights will be copied from)
			target_model: PyTorch model (weights will be copied to)
			tau (float): interpolation parameter
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)



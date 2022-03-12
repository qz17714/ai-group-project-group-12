
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLOAT = torch.FloatTensor
DOUBLE = torch.DoubleTensor
LONG = torch.LongTensor


def to_device(*args):
    return [arg.to(device) for arg in args]


def get_flat_params(model: nn.Module):
    """
    get tensor flatted parameters from model
    :param model:
    :return: tensor
    """
    return torch.cat([param.view(-1) for param in model.parameters()])


def get_flat_grad_params(model: nn.Module):
    """
    get flatted grad of parameters from the model
    :param model:
    :return: tensor
    """
    return torch.cat(
        [param.grad.view(-1) if param.grad is not None else torch.zeros(param.view(-1).shape) for param in
         model.parameters()])


def set_flat_params(model, flat_params):
    """
    set tensor flatted parameters to model
    :param model:
    :param flat_params: tensor
    :return:
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def resolve_activate_function(name):
    if name.lower() == "relu":
        return nn.ReLU
    if name.lower() == "sigmoid":
        return nn.Sigmoid
    if name.lower() == "leakyrelu":
        return nn.LeakyReLU
    if name.lower() == "prelu":
        return nn.PReLU
    if name.lower() == "softmax":
        return nn.Softmax
    if name.lower() == "tanh":
        return nn.Tanh

def dqn_step(value_net, optimizer_value, value_net_target, states, actions, rewards, next_states, masks, gamma):
    masks = masks.unsqueeze(-1)
    rewards = rewards.unsqueeze(-1)
    actions = actions.unsqueeze(-1)
    """update value net"""
    q_values = value_net(states).gather(1, actions)
    with torch.no_grad():
        q_target_next_values = value_net_target(next_states)
        q_target_values = rewards + gamma * masks * \
            q_target_next_values.max(1)[0].view(q_values.size(0), 1)

    value_loss = nn.MSELoss()(q_target_values, q_values)

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    return {"critic_loss": value_loss}

def doubledqn_step(value_net, optimizer_value, value_net_target, states, actions, rewards, next_states, masks, gamma,
                   polyak, update_target=False):
    masks = masks.unsqueeze(-1)
    rewards = rewards.unsqueeze(-1)
    actions = actions.unsqueeze(-1)
    """update q value net"""
    q_values = value_net(states).gather(1, actions)
    with torch.no_grad():
        q_target_next_values = value_net(next_states)
        q_target_actions = q_target_next_values.max(
            1)[1].view(q_values.size(0), 1)
        q_next_values = value_net_target(next_states)
        q_target_values = rewards + gamma * masks * \
            q_next_values.gather(1, q_target_actions).view(q_values.size(0), 1)

    value_loss = nn.MSELoss()(q_target_values, q_values)

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    if update_target:
        """update target q value net"""
        value_net_target.load_state_dict(value_net.state_dict())
        # value_net_target_flat_params = get_flat_params(value_net_target)
        # value_net_flat_params = get_flat_params(value_net)
        # set_flat_params(value_net_target, polyak * value_net_target_flat_params + (1 - polyak) * value_net_flat_params)

    return {"critic_loss": value_loss}


class BaseQNet(nn.Module):
    def __init__(self, dim_state, dim_action, dim_hidden=64):
        super(BaseQNet, self).__init__()

        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_action = dim_action

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def get_action(self, states):
        raise NotImplementedError()



def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class QNet_dqn(BaseQNet):
    def __init__(self, dim_state, dim_action, dim_hidden=64, activation=nn.LeakyReLU):
        super().__init__(dim_state, dim_action, dim_hidden)

        self.qvalue = nn.Sequential(nn.Linear(self.dim_state, self.dim_hidden),
                                    activation(),
                                    nn.Linear(self.dim_hidden, self.dim_hidden),
                                    activation(),
                                    nn.Linear(self.dim_hidden, self.dim_action))
        self.apply(init_weight)

    def forward(self, states, **kwargs):
        q_values = self.qvalue(states)
        return q_values

    def get_action(self, states):
        """
        >>>a = torch.rand(3, 4)
        tensor([[0.3643, 0.7805, 0.6098, 0.6551],
        [0.3953, 0.8059, 0.4277, 0.0126],
        [0.2667, 0.0109, 0.0467, 0.5328]])
        >>>a.max(dim=1)[1]
        tensor([1, 1, 3])
        :param states:
        :return: max_action (tensor)
        """
        q_values = self.forward(states, )
        max_action = q_values.max(dim=1)[1]  # action index with largest q values
        return max_action


import random
from collections import namedtuple, deque

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob'))


class Memory(object):
    def __init__(self, size=None):
        self.memory = deque(maxlen=size)

    # save item
    def push(self, *args):
        self.memory.append(Transition(*args))

    def clear(self):
        self.memory.clear()

    def append(self, other):
        self.memory += other.memory

    # sample a mini_batch
    def sample(self, batch_size=None):
        # sample all transitions
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:  # sample with size: batch_size
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def __len__(self):
        return len(self.memory)

Transition_h = namedtuple(
    'Transition', ('state', 'action_d', 'action_c', 'reward', 'next_state', 'mask', 'log_prob'))

class Memory_h(object):
    def __init__(self, size=None):
        self.memory = deque(maxlen=size)

    # save item
    def push(self, *args):
        self.memory.append(Transition_h(*args))

    def clear(self):
        self.memory.clear()

    def append(self, other):
        self.memory += other.memory

    # sample a mini_batch
    def sample(self, batch_size=None):
        # sample all transitions
        if batch_size is None:
            return Transition_h(*zip(*self.memory))
        else:  # sample with size: batch_size
            random_batch = random.sample(self.memory, batch_size)
            return Transition_h(*zip(*random_batch))

    def __len__(self):
        return len(self.memory)


from gym.spaces import Discrete

__all__ = ['get_env_info', 'get_env_space']


def get_env_space(env_id):
    env = gym.make(env_id)
    # env = env.unwrapped
    num_states = env.observation_space.shape[0]
    if type(env.action_space) == Discrete:
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]

    return env, num_states, num_actions

import gym

def get_env_info(env_id, unwrap=False):
    env = gym.make(env_id)
    if unwrap:  
        env = env.unwrapped
    num_states = env.observation_space.shape[0]
    env_continuous = False
    if type(env.action_space) == Discrete:
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]
        env_continuous = True

    return env, env_continuous, num_states, num_actions

import os


def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLOAT = torch.FloatTensor
DOUBLE = torch.DoubleTensor
LONG = torch.LongTensor


def to_device(*args):
    return [arg.to(device) for arg in args]


def get_flat_params(model: nn.Module):
    """
    get tensor flatted parameters from model
    :param model:
    :return: tensor
    """
    return torch.cat([param.view(-1) for param in model.parameters()])


def get_flat_grad_params(model: nn.Module):
    """
    get flatted grad of parameters from the model
    :param model:
    :return: tensor
    """
    return torch.cat(
        [param.grad.view(-1) if param.grad is not None else torch.zeros(param.view(-1).shape) for param in
         model.parameters()])


def set_flat_params(model, flat_params):
    """
    set tensor flatted parameters to model
    :param model:
    :param flat_params: tensor
    :return:
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def resolve_activate_function(name):
    if name.lower() == "relu":
        return nn.ReLU
    if name.lower() == "sigmoid":
        return nn.Sigmoid
    if name.lower() == "leakyrelu":
        return nn.LeakyReLU
    if name.lower() == "prelu":
        return nn.PReLU
    if name.lower() == "softmax":
        return nn.Softmax
    if name.lower() == "tanh":
        return nn.Tanh




class ZFilter2:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)
        self.fix = False

    def __call__(self, x, update=True):
        return x

import numpy as np

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

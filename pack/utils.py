import numpy as np
import tensorflow as tf


# data spec for observ, action, reward, etc.
class DataSpec(object):

    def __init__(self, shape=1, dtype=np.float64, mins=-1, maxs=1):
        self.shape = shape
        self.dtype = dtype
        self.mins = np.array(mins, dtype)
        self.maxs = np.array(maxs, dtype)
        if self.mins.shape == ():
            self.mins = np.ones(shape, dtype) * mins
        if self.maxs.shape == ():
            self.maxs = np.ones(shape, dtype) * maxs

    def __call__(self):
        return 'shape: {}, dtype: {}, mins: {}, maxs: {}'.format(self.shape, self.dtype, self.mins, self.maxs)


def matrix_dot(matrix_i, matrix_j):
    return abs(np.trace(matrix_i.conj().T @ matrix_j))


def tanh_squash(x, x_spec):
    mean = (x_spec.maxs + x_spec.mins) / 2.0
    scale = (x_spec.maxs - x_spec.mins) / 2.0
    return mean + scale * tf.tanh(x)


def softsign_squash(x, x_spec):
    mean = (x_spec.maxs + x_spec.mins) / 2.0
    scale = (x_spec.maxs - x_spec.mins) / 2.0
    return mean + scale * x / (1 + tf.math.abs(x))


def inv_softplus(x):
    return np.log(np.exp(x) - 1)


def get_discounted_return(rewards, discount=0.99):
    return tf.scan(fn=lambda discounted_return, reward: reward + discount * discounted_return, elems=rewards,
                   reverse=True)


def get_advantage(rewards, discounted_returns, values, discount=0.99, temporal_diff_weight=0.95,
                  use_generalized_advantage=True):
    if use_generalized_advantage:
        next_values = values[1:]
        values = values[:-1]
        temporal_diffs = rewards + discount * next_values - values
        temporal_diffs[-1] = rewards[-1] - values[-1]
        weighted_discount = discount * temporal_diff_weight
        return tf.scan(
            fn=lambda generalized_advantage, temporal_diff: temporal_diff + weighted_discount * generalized_advantage,
            elems=temporal_diffs, reverse=True)
    else:
        return discounted_returns - values[:-1]


def get_discounted_return_and_advantage(rewards, values, discount=0.99, temporal_diff_weight=0.95,
                                        use_generalized_advantage=True, use_temporal_diff_return=True):
    discounted_returns = get_discounted_return(rewards, discount)
    advantages = get_advantage(rewards, discounted_returns, values, discount, temporal_diff_weight,
                               use_generalized_advantage)
    if use_temporal_diff_return:
        discounted_returns = advantages + values[:-1]
    return discounted_returns, advantages


def joint_first_step(pulse):
    return np.concatenate((pulse[0:1], pulse))


def save_network(agent, path_name):
    agent.actor_network.network.save_weights(path_name + '/actor_network')
    agent.actor_network.proj_network.mean_proj_network.save_weights(path_name + '/mean_proj_network')
    agent.actor_network.proj_network.stddev_proj_network.save_weights(path_name + '/stddev_proj_network')
    agent.value_network.network.save_weights(path_name + '/value_network')


def load_network(agent, path_name, use_stddev_proj_network=False):
    agent.actor_network.network.load_weights(path_name + '/actor_network')
    agent.actor_network.proj_network.mean_proj_network.load_weights(path_name + '/mean_proj_network')
    if use_stddev_proj_network:
        agent.actor_network.proj_network.stddev_proj_network.load_weights(path_name + '/stddev_proj_network')
    agent.value_network.network.load_weights(path_name + '/value_network')
    return agent


def load_model(agent, path_name, use_stddev_proj_network=False):
    agent.actor_network.network = tf.keras.models.load_model(path_name + '/actor_network')
    agent.actor_network.proj_network.mean_proj_network = tf.keras.models.load_model(path_name + '/mean_proj_network')
    if use_stddev_proj_network:
        agent.actor_network.proj_network.stddev_proj_network = tf.keras.models.load_model(path_name + '/stddev_proj_network')
    agent.value_network.network = tf.keras.models.load_model(path_name + '/value_network')
    return agent

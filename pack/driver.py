import numpy as np
import tensorflow as tf
from pack.utils import get_discounted_return_and_advantage


class EpisodeDriver(object):

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def run(self, env, agent, greedy=False):
        observation_lst = []
        action_lst = []
        action_mean_lst = []
        action_stddev_lst = []
        value_lst = []
        reward_lst = []

        observation = env.reset()

        while not env.end:
            value = agent.predict(observation)
            action_distribution = agent.policy(observation)
            action = action_distribution.mode() if greedy else action_distribution.sample()
            action_mean, action_stddev = action_distribution.mean(), action_distribution.stddev()

            observation_lst.append(observation[0])
            action_lst.append(action[0])
            action_mean_lst.append(action_mean[0])
            action_stddev_lst.append(action_stddev[0])
            value_lst.append(value[0])

            observation, reward = env.step(action)
            reward_lst.append(reward[0])

            if env.end:
                value = agent.predict(observation)
                value_lst.append(value[0])

        return (
            np.array(observation_lst, self.dtype),
            np.array(action_lst, self.dtype),
            np.array(action_mean_lst, self.dtype),
            np.array(action_stddev_lst, self.dtype),
            np.array(value_lst, self.dtype),
            np.array(reward_lst, self.dtype)
        )

    def collect(self, env, agent, buffer, num_collects):
        for _ in range(num_collects):
            observations, actions, action_means, action_stddevs, values, rewards = self.run(env, agent)
            discounted_returns, advantages = get_discounted_return_and_advantage(
                rewards,
                values,
                agent.discount,
                agent.temporal_diff_weight,
                agent.use_generalized_advantage,
                agent.use_temporal_diff_return
            )
            buffer.add({
                'observation': observations,
                'action': actions,
                'action_mean': action_means,
                'action_stddev': action_stddevs,
                'reward': rewards,
                'discounted_return': discounted_returns,
                'advantage': advantages
            })

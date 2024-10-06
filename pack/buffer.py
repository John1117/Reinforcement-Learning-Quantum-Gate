import numpy as np
import tensorflow as tf


class ReplayBuffer(object):

    def __init__(self, max_size=np.inf):
        self.buffer = {
            'observation': None,
            'action': None,
            'action_mean': None,
            'action_stddev': None,
            'reward': None,
            'discounted_return': None,
            'advantage': None
        }
        self.max_size = max_size
        self.size = 0

    def add(self, new_data):
        self.size = min(self.size + len(new_data['observation']), self.max_size)
        reserved_i = -self.size
        for name, data in new_data.items():
            data = tf.stop_gradient(data)
            if self.buffer[name] is None:
                self.buffer[name] = data[reserved_i:]
            else:
                self.buffer[name] = np.concatenate((self.buffer[name], data), axis=0)[reserved_i:]

    def get(self, batch_size=None, random=True):
        if batch_size is None:
            batch_size = self.size
        elif batch_size > self.size:
            random = True

        batch = {}
        if random:
            sampled_i = np.random.choice(range(self.size), size=batch_size, replace=True)
            for name, data in self.buffer.items():
                batch[name] = np.array([data[i] for i in sampled_i])
        else:
            for name, data in self.buffer.items():
                batch[name] = data[-batch_size:]
        return batch

    def clear(self):
        self.__init__()

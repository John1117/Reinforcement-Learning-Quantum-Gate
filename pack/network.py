import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from pack.utils import softsign_squash, inv_softplus


class NormalProjNetwork(keras.Model):

    def __init__(
            self,
            action_spec,
            # use truncated normal to avoid flat reward for actions outside the bound
            distribution_fn=tfp.distributions.TruncatedNormal,
            mean_activation_fn=None,
            # initialize mean kernel and bias to zeros to make zero-initialized pulse
            mean_kernel_init=keras.initializers.Zeros(),
            mean_bias_init=keras.initializers.Zeros(),
            mean_transform_fn=softsign_squash,
            stddev_activation_fn=None,
            stddev_kernel_init=keras.initializers.Zeros(),
            stddev_transform_fn=tf.math.softplus,
            init_action_stddev=2.0,
            train_stddev=True,
            observation_dependent_stddev=False,
            dtype=np.float64
    ):

        super().__init__()

        self.action_spec = action_spec
        self.distribution_fn = distribution_fn
        self.mean_transform_fn = mean_transform_fn
        self.stddev_transform_fn = stddev_transform_fn
        self.train_stddev = train_stddev
        self.action_stddev = init_action_stddev
        self.observation_dependent_stddev = observation_dependent_stddev if train_stddev else False

        self.mean_proj_network = keras.Sequential(name='mean_proj_network')
        self.mean_proj_network.add(
            keras.layers.Dense(
                action_spec.shape,
                activation=mean_activation_fn,
                kernel_initializer=mean_kernel_init,
                bias_initializer=mean_bias_init,
                dtype=dtype,
                name='mean_proj_layer'
            )
        )

        stddev_bias_init = keras.initializers.Constant(inv_softplus(init_action_stddev))
        if observation_dependent_stddev:
            self.stddev_proj_network = keras.Sequential(name='stddev_proj_network')
            self.stddev_proj_network.add(keras.layers.InputLayer((action_spec.shape,)))
            self.stddev_proj_network.add(
                keras.layers.Dense(
                    action_spec.shape,
                    activation=stddev_activation_fn,
                    kernel_initializer=stddev_kernel_init,
                    bias_initializer=stddev_bias_init,
                    dtype=dtype,
                    name='stddev_proj_layer'
                )
            )

        else:
            self.stddev_proj_network = keras.Sequential(name='stddev_proj_network')
            self.stddev_proj_network.add(keras.layers.InputLayer((action_spec.shape,)))
            self.stddev_proj_network.add(
                keras.layers.Dense(
                    action_spec.shape,
                    activation=None,
                    kernel_initializer=stddev_kernel_init,
                    bias_initializer=stddev_bias_init,
                    dtype=dtype,
                    name='stddev_proj_layer'
                )
            )
        self.stddev_proj_network.get_layer('stddev_proj_layer').trainable = True if train_stddev else False

    def __call__(self, actor_network_output):
        action_mean = self.mean_transform_fn(self.mean_proj_network(actor_network_output), self.action_spec)
        if self.train_stddev:
            if self.observation_dependent_stddev:
                stddev_proj_output = self.stddev_proj_network(actor_network_output)
            else:
                stddev_proj_output = self.stddev_proj_network(tf.zeros_like(action_mean))
            action_stddev = self.stddev_transform_fn(stddev_proj_output)
        else:
            action_stddev = self.action_stddev
        return self.distribution_fn(action_mean, action_stddev, self.action_spec.mins, self.action_spec.maxs)

    def get_action_stddev(self):
        return self.action_stddev

    def set_action_stddev(self, action_stddev):
        self.action_stddev = action_stddev


class ActorNetwork(keras.Model):

    def __init__(
            self,
            observation_spec,
            action_spec,
            network_config=(0,),
            activation_fn=keras.activations.relu,
            proj_network=None,
            dtype=np.float64
    ):
        super().__init__()

        self.proj_network = proj_network

        network = keras.Sequential(name='actor_network')
        network.add(keras.layers.InputLayer(observation_spec.shape, dtype=dtype, name='input_layer'))
        for i, layer_param in enumerate(network_config):
            network.add(keras.layers.Dense(layer_param, activation_fn, dtype=dtype, name='hidden_layer_{}'.format(i + 1)))
        network.add(keras.layers.Dense(action_spec.shape, dtype=dtype, name='output_layer'))
        self.network = network

    def __call__(self, observation):
        action_distribution = self.proj_network(self.network(observation))
        return action_distribution


class ValueNetwork(keras.Model):

    def __init__(
            self,
            observation_spec,
            network_config=(0,),
            activation_fn=keras.activations.relu,
            dtype=np.float64
    ):
        super().__init__()

        network = keras.Sequential(name='value_network')
        network.add(keras.layers.InputLayer(observation_spec.shape, dtype=dtype, name='input_layer'))
        for i, layer_param in enumerate(network_config):
            network.add(keras.layers.Dense(layer_param, activation_fn, dtype=dtype, name='hidden_layer_{}'.format(i + 1)))
        network.add(keras.layers.Dense(1, dtype=dtype, name='output_layer'))
        self.network = network

    def __call__(self, observation):
        return self.network(observation)

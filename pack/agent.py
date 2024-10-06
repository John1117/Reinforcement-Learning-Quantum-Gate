import numpy as np
import tensorflow as tf
from pack.network import NormalProjNetwork, ActorNetwork, ValueNetwork
from tensorflow import keras


class PPOAgent(object):

    def __init__(
            self,
            observation_spec=None,
            action_spec=None,
            optimizer=None,
            actor_network=None,
            value_network=None,
            ratio_clip_value=0.2,
            discount=1.,
            temporal_diff_weight=1.,
            use_generalized_advantage=True,
            use_temporal_diff_return=True,
            dtype=np.float64
    ):

        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.optimizer = optimizer
        self.actor_network = actor_network
        self.value_network = value_network
        self.ratio_clip_value = ratio_clip_value
        self.discount = discount
        self.temporal_diff_weight = temporal_diff_weight
        self.use_generalized_advantage = use_generalized_advantage
        self.use_temporal_diff_return = use_temporal_diff_return
        self.dtype = dtype
        self.loss_info = {'policy_gradient_loss': None, 'value_predict_loss': None}

    def policy(self, observation):
        return self.actor_network(observation)

    def predict(self, observation):
        return self.value_network(observation)

    def train(self, buffer, num_epochs, batch_size):
        for _ in range(num_epochs):
            batch = buffer.get(batch_size)
            observation = batch['observation']
            action = batch['action']
            action_mean = batch['action_mean']
            action_stddev = batch['action_stddev']
            advantage = batch['advantage']
            discounted_return = batch['discounted_return']
            with tf.GradientTape() as tape:
                loss = self.get_loss(observation, action, action_mean, action_stddev, advantage, discounted_return)
            variables = self.actor_network.trainable_variables + self.value_network.trainable_variables
            gradient = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradient, variables))

    def get_loss(self, observation, action, action_mean, action_stddev, advantage, discounted_return):
        policy_gradient_loss = self.get_policy_gradient_loss(observation, action, action_mean, action_stddev, advantage)
        value_predict_loss = self.get_value_predict_loss(observation, discounted_return)
        return policy_gradient_loss + value_predict_loss

    def get_policy_gradient_loss(self, observation, action, action_mean, action_stddev, advantage):
        old_action_log_prob = self.actor_network.proj_network.distribution_fn(
            action_mean,
            action_stddev,
            self.action_spec.mins,
            self.action_spec.maxs
        ).log_prob(action)
        new_action_log_prob = self.actor_network(observation).log_prob(action)

        ratio = tf.exp(new_action_log_prob - old_action_log_prob)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.ratio_clip_value, 1 + self.ratio_clip_value)
        loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clipped_ratio * advantage))
        self.loss_info['policy_gradient_loss'] = loss.numpy()
        return loss

    def get_value_predict_loss(self, observation, discounted_return):
        value = self.value_network(observation)
        loss = tf.reduce_mean(tf.math.squared_difference(value, discounted_return))
        self.loss_info['value_predict_loss'] = loss.numpy()
        return loss

    def get_action_stddev(self):
        return self.actor_network.proj_network.get_action_stddev()

    def set_action_stddev(self, action_stddev):
        self.actor_network.proj_network.set_action_stddev(action_stddev)

    def get_learning_rate(self):
        return self.optimizer.lr.numpy()

    def set_learning_rate(self, learning_rate):
        self.optimizer.lr.assign(learning_rate)


def build_ppo(env, init_learning_rate, actor_network_config, value_network_config, init_action_stddev, train_stddev):

    adam = keras.optimizers.Adam(
            learning_rate=init_learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=True
        )

    actor_network = ActorNetwork(
        env.observation_spec,
        env.action_spec,
        network_config=actor_network_config,
        proj_network=NormalProjNetwork(
            env.action_spec,
            init_action_stddev=init_action_stddev,
            train_stddev=train_stddev
        )
    )

    value_network = ValueNetwork(
        env.observation_spec,
        network_config=value_network_config
    )

    return PPOAgent(
        observation_spec=env.observation_spec,
        action_spec=env.action_spec,
        optimizer=adam,
        actor_network=actor_network,
        value_network=value_network
    )

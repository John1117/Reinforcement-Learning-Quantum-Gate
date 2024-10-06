import os

import numpy as np
import tensorflow as tf
from pack.env2q import evolve, get_inf


class CNOTTester(object):

    def __init__(self, env, agent, driver, noise_test_stddevs, num_noise_test_samples, noise_test_interval, ideal_weight, noise_weight):
        # universal params
        self.noise_test_stddevs = noise_test_stddevs
        self.num_noise_test_samples = num_noise_test_samples
        self.noise_test_interval = noise_test_interval
        self.ideal_weight = ideal_weight
        self.noise_weight = noise_weight

        # original params for old agent
        old_action_stddevs = driver.run(env, agent, greedy=True)[3]
        self.old_X1 = env.X1_pulse
        self.old_X2 = env.X2_pulse
        self.old_J = env.J_pulse
        self.old_noise_test_infs, self.old_noise_orders = noise_test(env, [self.old_X1, self.old_X2, self.old_J], noise_test_stddevs, num_noise_test_samples)

        # initialize the new and best params to be the old ones
        self.new_X1 = self.old_X1
        self.new_X2 = self.old_X1
        self.new_J = self.old_X1
        self.new_noise_test_infs = self.old_noise_test_infs
        self.new_noise_orders = self.old_noise_orders
        self.best_X1 = self.old_X1
        self.best_X2 = self.old_X1
        self.best_J = self.old_X1
        self.best_noise_test_infs = self.old_noise_test_infs
        self.best_noise_orders = self.old_noise_orders

        self.find_best = False
        self.best_iter = 0
        self.best_weighted_inf = ideal_weight * env.ideal_inf + noise_weight * env.noise_inf

        # some lists to record params for drawing dashboard
        self.iters = [0]
        #self.infss = [[env.mean_inf, env.ideal_inf, env.noise_inf, env.weighted_inf]]
        self.mean_infs = [env.mean_inf]
        self.ideal_infs = [env.ideal_inf]
        self.noise_infs = [env.noise_inf]
        self.weighted_infs = [env.weighted_inf]
        self.stddevs = [old_action_stddevs.mean()]
        self.policy_losses = [None]
        self.value_losses = [None]

    def test(self, env, agent, driver, iter_idx):
        self.find_best = False
        self.iters.append(iter_idx)

        action_stddevs = driver.run(env, agent, greedy=True)[3]
        #self.infss.append([env.mean_inf, env.ideal_inf, env.noise_inf, env.weighted_inf])
        self.mean_infs.append(env.mean_inf)
        self.ideal_infs.append(env.ideal_inf)
        self.noise_infs.append(env.noise_inf)
        self.weighted_infs.append(env.weighted_inf)
        self.stddevs.append(action_stddevs.mean())
        self.new_X1 = env.X1_pulse
        self.new_X2 = env.X2_pulse
        self.new_J = env.J_pulse

        if iter_idx % self.noise_test_interval == 0:
            self.new_noise_test_infs, self.new_noise_orders = noise_test(env, [self.new_X1, self.new_X2, self.new_J], self.noise_test_stddevs, self.num_noise_test_samples)

        # update the best agent
        if self.best_weighted_inf > self.ideal_weight * env.ideal_inf + self.noise_weight * env.noise_inf:
            self.best_iter = iter_idx
            self.best_weighted_inf = self.ideal_weight * env.ideal_inf + self.noise_weight * env.noise_inf
            self.best_X1 = env.X1_pulse
            self.best_X2 = env.X2_pulse
            self.best_J = env.J_pulse
            self.find_best = True

            # do noise test if finding the best agent
            self.best_noise_test_infs, self.best_noise_orders = noise_test(env, [self.best_X1, self.best_X2, self.best_J], self.noise_test_stddevs, self.num_noise_test_samples)

        self.policy_losses.append(agent.loss_info['policy_gradient_loss'])
        self.value_losses.append(agent.loss_info['value_predict_loss'])

    def save(self, path_name):
        os.makedirs(path_name)
        np.save(path_name + '/new_X1_pulse.npy', self.new_X1)
        np.save(path_name + '/new_X2_pulse.npy', self.new_X2)
        np.save(path_name + '/new_J_pulse.npy', self.new_J)
        np.save(path_name + '/new_noise_test_infs.npy', self.new_noise_test_infs)
        np.save(path_name + '/new_noise_orders.npy', self.new_noise_orders)
        np.save(path_name + '/best_X1_pulse.npy', self.best_X1)
        np.save(path_name + '/best_X2_pulse.npy', self.best_X2)
        np.save(path_name + '/best_J_pulse.npy', self.best_J)
        np.save(path_name + '/best_noise_test_infs.npy', self.best_noise_test_infs)
        np.save(path_name + '/best_noise_orders.npy', self.best_noise_orders)
        np.save(path_name + '/iters.npy', self.iters)
        np.save(path_name + '/mean_infs.npy', self.mean_infs)
        np.save(path_name + '/ideal_infs.npy', self.ideal_infs)
        np.save(path_name + '/noise_infs.npy', self.noise_infs)
        np.save(path_name + '/weighted_infs.npy', self.weighted_infs)
        np.save(path_name + '/stddevs.npy', self.stddevs)
        np.save(path_name + '/policy_losses.npy', self.policy_losses)
        np.save(path_name + '/value_losses.npy', self.value_losses)


def noise_test_z1(env, pulse, stddevs, num_noise_samples):
    infs = np.zeros(stddevs.shape[0])
    for i, Z1_stddev in enumerate(stddevs):
        for Z1_noise in np.random.normal(env.Z1_const, Z1_stddev, num_noise_samples):
            Z1_pulse = np.repeat(Z1_noise, env.num_steps)
            Z2_pulse = np.repeat(env.Z2_const, env.num_steps)
            X1_pulse, X2_pulse, J_pulse = pulse
            final_gate = evolve(Z1_pulse, X1_pulse, Z2_pulse, X2_pulse, J_pulse, env.step_time)
            infs[i] += get_inf(final_gate, env.target_gate)
    infs /= num_noise_samples
    ords = np.diff(np.log10(infs)) / np.diff(np.log10(stddevs))
    return infs, ords

def noise_test(env, pulse, stddevs, num_noise_samples):
    infs = np.zeros(stddevs.shape[0])
    for i, stddev in enumerate(stddevs):
        for Z1_noise, Z2_noise in zip(np.random.normal(env.Z1_const, stddev, num_noise_samples), np.random.normal(env.Z2_const, stddev, num_noise_samples)):
            Z1_pulse = np.repeat(Z1_noise, env.num_steps)
            Z2_pulse = np.repeat(Z2_noise, env.num_steps)
            X1_pulse, X2_pulse, J_pulse = pulse
            final_gate = evolve(Z1_pulse, X1_pulse, Z2_pulse, X2_pulse, J_pulse, env.step_time)
            infs[i] += get_inf(final_gate, env.target_gate)
    infs /= num_noise_samples
    ords = np.diff(np.log10(infs)) / np.diff(np.log10(stddevs))
    return infs, ords
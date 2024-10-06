import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pack.utils import joint_first_step
matplotlib.use('TkAgg')


def set_ax(ax, title=None, xlabel=None, ylabel=None, xscale='linear', yscale='linear'):
    ax.clear()
    if title:
        ax.set_title(title, fontsize=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    if ylabel:
        ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.tick_params(axis='both', labelsize=20)
    for sp in ax.spines.values():
        sp.set_linewidth(2)
    ax.grid(True)
    return ax


class CNOTLearningDashboard(object):

    def __init__(self, name='dashboard', pulse_ticks=None, figsize=(20, 10)):
        self.pulse_ticks = pulse_ticks

        plt.ion()
        self.fig = plt.figure(num=name, figsize=figsize, tight_layout=True)
        gs = self.fig.add_gridspec(nrows=4, ncols=3)

        self.ax_inf = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_noise = self.fig.add_subplot(gs[2:4, 0])
        self.ax_order = self.fig.add_subplot(gs[2:4, 1])
        self.ax_X1 = self.fig.add_subplot(gs[0, 2])
        self.ax_X2 = self.fig.add_subplot(gs[1, 2])
        self.ax_J = self.fig.add_subplot(gs[2, 2])
        self.ax_stddev = self.fig.add_subplot(gs[3, 2])
        self.fig.show()

    def update(self, tester):
        self.plot_inf(tester.iters, tester.mean_infs, tester.ideal_infs, tester.noise_infs, tester.weighted_infs, tester.best_iter, tester.best_weighted_inf)
        self.plot_noise(tester.noise_test_stddevs, tester.new_noise_test_infs, tester.old_noise_test_infs, tester.best_noise_test_infs)
        self.plot_order(tester.noise_test_stddevs, tester.new_noise_orders, tester.old_noise_orders, tester.best_noise_orders)
        self.plot_X1(self.pulse_ticks, tester.new_X1, tester.old_X1, tester.best_X1)
        self.plot_X2(self.pulse_ticks, tester.new_X2, tester.old_X2, tester.best_X2)
        self.plot_J(self.pulse_ticks, tester.new_J, tester.old_J, tester.best_J)
        self.plot_stddev(tester.iters, tester.stddevs)
        self.fig.canvas.flush_events()

    def show(self):
        plt.ioff()
        plt.show()

    def save(self, path_name):
        plt.savefig(fname=path_name + '.png')
        plt.savefig(fname=path_name + '.pdf')

    def plot_inf(self, iters, mean_infs, ideal_infs, noise_infs, weighted_infs, best_iter, best_weighted_inf):
        ax = set_ax(self.ax_inf, title='Infidelity Learning Curve', xlabel='Iteration', ylabel='Infidelity', yscale='log')
        ax.plot(iters, weighted_infs, 'g', label='Weighted', lw=5)
        ax.plot(iters, mean_infs, 'b', label='Mean')
        ax.plot(iters, ideal_infs, 'k', label='Ideal')
        ax.plot(iters, noise_infs, 'r', label='Noise')
        ax.plot(best_iter, best_weighted_inf, 'g*', markersize=20, label='Best')
        ax.legend(fontsize=20)

    def plot_noise(self, stddevs, new_infs, old_infs, best_infs):
        ax = set_ax(self.ax_noise, title='Noise Test', xlabel='Noise Stddev', ylabel='Mean Infidelity', xscale='log', yscale='log')
        ax.set_ylim(np.nanmin([old_infs, new_infs, best_infs]) * 0.1, 1.3)

        ax.plot(stddevs, old_infs, 'c', label='Old')
        ax.plot(stddevs, new_infs, 'b', label='New')
        ax.plot(stddevs, best_infs, 'g', label='Best')
        ax.plot(stddevs, stddevs, 'k--')
        ax.plot(stddevs, stddevs ** 2, 'k--')
        ax.plot(stddevs, stddevs ** 4, 'k--')
        ax.legend(fontsize=20)

    def plot_order(self, stddevs, new_orders, old_orders, best_orders):
        ax = set_ax(self.ax_order, title='Noise Order', xlabel='Noise Stddev', ylabel='Noise Order', xscale='log')
        stddevs = (stddevs[1:] + stddevs[:-1])/2
        ax.plot(stddevs, old_orders, 'c', label='Old')
        ax.plot(stddevs, new_orders, 'b', label='New')
        ax.plot(stddevs, best_orders, 'g', label='Best')
        ax.legend(fontsize=20)

    def plot_X1(self, pulse_ticks, new_pulse, old_pulse, best_pulse):
        ax = set_ax(self.ax_X1, title='X1 Pulse', xlabel='Time', ylabel='Amplitude')
        ax.step(pulse_ticks, joint_first_step(old_pulse), c='c', label='Old')
        ax.step(pulse_ticks, joint_first_step(new_pulse), c='b', label='New')
        ax.step(pulse_ticks, joint_first_step(best_pulse), c='g', label='Best')
        ax.legend(fontsize=20)

    def plot_X2(self, pulse_ticks, new_pulse, old_pulse, best_pulse):
        ax = set_ax(self.ax_X2, title='X2 Pulse', xlabel='Time', ylabel='Amplitude')
        ax.step(pulse_ticks, joint_first_step(old_pulse), c='c', label='Old')
        ax.step(pulse_ticks, joint_first_step(new_pulse), c='b', label='New')
        ax.step(pulse_ticks, joint_first_step(best_pulse), c='g', label='Best')
        ax.legend(fontsize=20)

    def plot_J(self, pulse_ticks, new_pulse, old_pulse, best_pulse):
        ax = set_ax(self.ax_J, title='J Pulse', xlabel='Time', ylabel='Amplitude')
        ax.step(pulse_ticks, joint_first_step(old_pulse), c='c', label='Old')
        ax.step(pulse_ticks, joint_first_step(new_pulse), c='b', label='New')
        ax.step(pulse_ticks, joint_first_step(best_pulse), c='g', label='Best')
        ax.legend(fontsize=20)

    def plot_stddev(self, iters, stddevs):
        ax = set_ax(self.ax_stddev, title='Action Stddev', xlabel='Iteration', ylabel='ASD', yscale='log')
        ax.plot(iters, stddevs, 'b')

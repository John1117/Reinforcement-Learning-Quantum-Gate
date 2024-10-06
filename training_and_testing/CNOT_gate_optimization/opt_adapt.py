import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pack.env import PulseSeqEnv, X_gate, H_gate, evolve, get_inf
from pack.test import noise_test
from pack.utils import joint_first_step
matplotlib.use('TkAgg')

k = 1.5**0.5
s = 0.1
target_gate = X_gate
max_gate_time = 4
num_steps = 16
Sz_const = 1.
Sz_noise_samples = np.array([0., s * k, -s * k])
max_Sx_amp = 4.
train_ideal_weight = 1.
dtype = np.float64

maxiter = num_steps * 1000
xatol = 1e-6
fatol = 1e-14

stddevs = np.logspace(-4, 0, 21)
num_noise_samples = 100


def env_fn(Sx_pulse, env):
    env.reset()
    for Sx_coef in Sx_pulse:
        env.step(Sx_coef)
    return env.weighted_inf


num_trials = 10
ss = np.array([1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 0])
infss = np.zeros((len(ss), num_trials, len(stddevs)))
ordss = np.zeros((len(ss), num_trials, len(stddevs) - 1))
pulses = np.zeros((len(ss), num_trials, num_steps))
for i, s_ in enumerate(ss):
    for j in range(num_trials):
        train_noise_weight = s_
        env = PulseSeqEnv(
            target_gate=target_gate,
            max_gate_time=max_gate_time,
            num_steps=num_steps,
            Sz_const=Sz_const,
            Sz_noise_samples=Sz_noise_samples,
            max_Sx_amp=max_Sx_amp,
            ideal_weight=train_ideal_weight,
            noise_weight=train_noise_weight,
            dtype=dtype
        )
        if i == 0:
            x0 = np.random.uniform(low=-max_Sx_amp/2, high=max_Sx_amp/2, size=num_steps)
        else:
            x0 = pulses[i-1, j]
        res = minimize(
            fun=env_fn,
            x0=x0,
            method='Nelder-Mead',
            bounds=np.repeat([[-max_Sx_amp, max_Sx_amp]], num_steps, axis=0),
            args=(env,),
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol}
        )
        infs, ords = noise_test(env, res.x, stddevs, num_noise_samples)
        infss[i, j] = infs
        ordss[i, j] = ords
        pulses[i, j] = res.x
        print(res)
        print('---------------------------------------')


jointed_pulses = joint_first_step(pulses.T).T
pulse_ticks = np.linspace(0, max_gate_time, num_steps + 1)

fig, ax = plt.subplots(1, 3, figsize=(20, 10), tight_layout=True)

ax[0].set_xlabel('Noise stddev', fontsize=20)
ax[0].set_ylabel('Infidelity', fontsize=20)
for i, infs in enumerate(infss):
    ax[0].plot(stddevs, infs.T, alpha=(i+1)/len(ss))
    ax[0].plot(stddevs, np.exp(np.log(infs).mean(axis=0)), 'k', lw=5, alpha=(i+1)/len(ss))
ax[0].plot(stddevs, stddevs, 'k--')
ax[0].plot(stddevs, stddevs ** 2, 'k--')
ax[0].plot(stddevs, stddevs ** 4, 'k--')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_ylim((infss.min() * 0.1, None))
ax[0].tick_params(axis='both', labelsize=20)
ax[0].grid()

ax[1].set_xlabel('Noise stddev', fontsize=20)
ax[1].set_ylabel('Noise order', fontsize=20)
for i, ords in enumerate(ordss):
    ax[1].plot((stddevs[1:] + stddevs[:-1])/2, ords.T, alpha=(i+1)/len(ss))
    ax[1].plot((stddevs[1:] + stddevs[:-1])/2, ords.mean(axis=0), 'k', lw=5, alpha=(i+1)/len(ss))
ax[1].set_xscale('log')
ax[1].tick_params(axis='both', labelsize=20)
ax[1].grid()

ax[2].set_xlabel('Time', fontsize=20)
ax[2].set_ylabel('Amplitude', fontsize=20)
for i, jointed_pulse in enumerate(jointed_pulses):
    ax[2].step(pulse_ticks, jointed_pulse.T, alpha=(i+1)/len(ss))
ax[2].tick_params(axis='both', labelsize=20)
ax[2].grid()
plt.show()

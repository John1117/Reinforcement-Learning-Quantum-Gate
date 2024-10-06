import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pack.env import PulseSeqEnv, X_gate, H_gate, evolve, get_inf
from pack.test import noise_test
from pack.utils import joint_first_step
matplotlib.use('TkAgg')

s = 0.01
k = 1.5**0.5
r = (7/2.0202)**0.5

target_gate = X_gate
max_gate_time = 1
num_steps = 1
Sz_const = 0
#Sz_noise_samples = np.random.normal(loc=0., scale=s, size=50)
#Sz_noise_samples = np.array([0., s * k, -s * k])
#Sz_noise_samples = np.array([0., s * r, 0.1 * s * r, 0.01 * s * r, -s * r, -0.1 * s * r, -0.01 * s * r])
Sz_noise_samples = np.array([0.])
max_Sx_amp = 4.
train_ideal_weight = 1.
train_noise_weight = 1.
dtype = np.float64

maxiter = num_steps * 1000
xatol = 1e-6
fatol = 1e-14

stddevs = np.logspace(-4, 0, 21)
num_noise_samples = 10

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


def env_fn(Sx_pulse, env):
    env.reset()
    for Sx_coef in Sx_pulse:
        env.step(Sx_coef)
    return env.weighted_inf


num_trials = 1
success_tol = s ** 4 * max_gate_time
success_counts = 0
old_infss = []
old_ordss = []
old_pulses = []
for idx_trial in range(num_trials):
    x0 = np.random.uniform(low=-max_Sx_amp/2, high=max_Sx_amp/2, size=num_steps)
    #x0 = np.random.uniform(low=max_Sx_amp/2, high=max_Sx_amp, size=num_steps)
    res = minimize(
        fun=env_fn,
        x0=x0,
        method='Nelder-Mead',
        bounds=np.repeat([[-max_Sx_amp, max_Sx_amp]], num_steps, axis=0),
        args=(env,),
        options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol}
    )
    #infs, ords = noise_test(env, res.x, stddevs, num_noise_samples)
    #old_infss.append(infs)
    #old_ordss.append(ords)
    #old_pulses.append(res.x)
    print(res)
    print('---------------------------------------')

    if abs(res.fun) < success_tol:
        success_counts += 1

print('success rate:', np.round(success_counts / num_trials * 100, 1), '%')

"""
old_infss = np.array(old_infss)
old_ordss = np.array(old_ordss)
old_pulses = np.array(old_pulses)
old_jointed_pulses = joint_first_step(old_pulses.T).T
pulse_ticks = np.linspace(0, max_gate_time, num_steps + 1)


#--------------------------------------------------------------------------------------------------------------------------------------------------------
target_gate = H_gate
Sz_const = 1.
s = 0.01
Sz_noise_samples = np.array([0., s * k, -s * k])

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

success_tol = s ** 4 * max_gate_time
success_counts = 0
new_infss = []
new_ordss = []
new_pulses = []
for old_pulse in old_pulses:
    x0 = old_pulse
    res = minimize(
        fun=env_fn,
        x0=x0,
        method='Nelder-Mead',
        bounds=np.repeat([[-max_Sx_amp, max_Sx_amp]], num_steps, axis=0),
        args=(env,),
        options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol}
    )
    infs, ords = noise_test(env, res.x, stddevs, num_noise_samples)
    new_infss.append(infs)
    new_ordss.append(ords)
    new_pulses.append(res.x)
    print(res)
    print('---------------------------------------')

    if abs(res.fun) < success_tol:
        success_counts += 1

print('success rate:', np.round(success_counts / num_trials * 100, 1), '%')

new_infss = np.array(new_infss)
new_ordss = np.array(new_ordss)
new_pulses = np.array(new_pulses)
new_jointed_pulses = joint_first_step(new_pulses.T).T

print((old_pulses * max_gate_time / num_steps).sum(axis=1))
print((new_pulses * max_gate_time / num_steps).sum(axis=1))


fig, ax = plt.subplots(2, 3, figsize=(20, 10), tight_layout=True)

ax[0, 0].set_xlabel('Noise stddev', fontsize=20)
ax[0, 0].set_ylabel('Infidelity', fontsize=20)
ax[0, 0].plot(stddevs, old_infss.T)
ax[0, 0].plot(stddevs, np.exp(np.log(old_infss).mean(axis=0)), 'k-', lw=5)
ax[0, 0].plot(stddevs, stddevs, 'k--')
ax[0, 0].plot(stddevs, stddevs ** 2, 'k--')
ax[0, 0].plot(stddevs, stddevs ** 4, 'k--')
ax[0, 0].set_xscale('log')
ax[0, 0].set_yscale('log')
ax[0, 0].set_ylim((old_infss.min() * 0.1, None))
ax[0, 0].tick_params(axis='both', labelsize=20)
ax[0, 0].grid()

ax[0, 1].set_xlabel('Noise stddev', fontsize=20)
ax[0, 1].set_ylabel('Noise order', fontsize=20)
ax[0, 1].plot((stddevs[1:] + stddevs[:-1])/2, old_ordss.T)
ax[0, 1].plot((stddevs[1:] + stddevs[:-1])/2, old_ordss.mean(axis=0), 'k-', lw=5)
ax[0, 1].set_xscale('log')
ax[0, 1].tick_params(axis='both', labelsize=20)
ax[0, 1].grid()

ax[0, 2].set_xlabel('Time', fontsize=20)
ax[0, 2].set_ylabel('Amplitude', fontsize=20)
ax[0, 2].step(pulse_ticks, old_jointed_pulses.T)
ax[0, 2].tick_params(axis='both', labelsize=20)
ax[0, 2].grid()

ax[1, 0].set_xlabel('Noise stddev', fontsize=20)
ax[1, 0].set_ylabel('Infidelity', fontsize=20)
ax[1, 0].plot(stddevs, new_infss.T)
ax[1, 0].plot(stddevs, np.exp(np.log(new_infss).mean(axis=0)), 'k-', lw=5)
ax[1, 0].plot(stddevs, stddevs, 'k--')
ax[1, 0].plot(stddevs, stddevs ** 2, 'k--')
ax[1, 0].plot(stddevs, stddevs ** 4, 'k--')
ax[1, 0].set_xscale('log')
ax[1, 0].set_yscale('log')
ax[1, 0].set_ylim((new_infss.min() * 0.1, None))
ax[1, 0].tick_params(axis='both', labelsize=20)
ax[1, 0].grid()

ax[1, 1].set_xlabel('Noise stddev', fontsize=20)
ax[1, 1].set_ylabel('Noise order', fontsize=20)
ax[1, 1].plot((stddevs[1:] + stddevs[:-1])/2, new_ordss.T)
ax[1, 1].plot((stddevs[1:] + stddevs[:-1])/2, new_ordss.mean(axis=0), 'k-', lw=5)
ax[1, 1].set_xscale('log')
ax[1, 1].tick_params(axis='both', labelsize=20)
ax[1, 1].grid()

ax[1, 2].set_xlabel('Time', fontsize=20)
ax[1, 2].set_ylabel('Amplitude', fontsize=20)
ax[1, 2].step(pulse_ticks, new_jointed_pulses.T)
ax[1, 2].tick_params(axis='both', labelsize=20)
ax[1, 2].grid()

plt.show()"""

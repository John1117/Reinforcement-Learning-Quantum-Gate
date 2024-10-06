import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pack.env2q import TwoQubitEnv, CNOT, evolve, get_inf
from pack.utils import joint_first_step
matplotlib.use('TkAgg')


def noise_test(env, pulse, stddevs, num_noise_samples):
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


k = 1.5**0.5
s = 0.1

target_gate = CNOT
max_gate_time = 100
num_steps = 16
Z1_const = 1.0
Z1_noises = np.array([0.0, k*s, -k*s]) #, k*s, -k*s
Z2_const = 1.0
Z2_noises = np.array([0.0])
max_X1_amp = 4.0
max_X2_amp = 4.0
max_J_amp = 4.0
ideal_weight = 1.0
noise_weight = 1.0
dtype = np.float64


env = TwoQubitEnv(
    target_gate=target_gate,
    max_gate_time=max_gate_time,
    num_steps=num_steps,
    Z1_const=Z1_const,
    Z1_noises=Z1_noises,
    Z2_const=Z2_const,
    Z2_noises=Z2_noises,
    max_X1_amp=max_X1_amp,
    max_X2_amp=max_X2_amp,
    max_J_amp=max_J_amp,
    ideal_weight=ideal_weight,
    noise_weight=noise_weight,
    dtype=dtype
)

maxiter = 3 * num_steps * 1000
xatol = 1e-6
fatol = 1e-14

stddevs = np.logspace(-4, 0, 21)
mid_stddevs = (stddevs[1:] + stddevs[:-1]) / 2
num_noise_samples = 100


def env_fn(ctrl_pulse, env):
    ctrl_pulse = np.reshape(ctrl_pulse, (3, env.num_steps))
    env.reset()
    for coef in ctrl_pulse.T:
        env.step(coef)
    return env.weighted_inf


num_trials = 5
success_tol = (s * Z1_const * max_gate_time) ** 3 / 6
success_counts = 0
inf_ls = []
infss = np.zeros((num_trials, stddevs.shape[0]))
ordss = np.zeros((num_trials, mid_stddevs.shape[0]))

plt.ion()
fig, ax = plt.subplots(1, 2, num=f'T={max_gate_time}, N={num_steps}', figsize=(15, 7.5), tight_layout=True)
fig.show()
for i in range(num_trials):
    x0 = np.random.uniform(low=-max_X1_amp/2, high=max_X1_amp/2, size=3*num_steps)
    res = minimize(
        fun=env_fn,
        x0=x0,
        method='Nelder-Mead',
        bounds=np.repeat([[-max_X1_amp, max_X1_amp]], 3*num_steps, axis=0),
        args=(env,),
        options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol}
    )
    p = res.x.reshape(3, env.num_steps)
    infs, ords = noise_test(env, p, stddevs, num_noise_samples)
    infss[i] = infs
    ordss[i] = ords

    ax[0].clear()
    ax[0].plot(stddevs, infss.T)
    ax[0].plot(stddevs, np.prod(infss[0:i+1], axis=0) ** (1/(i+1)), c='k', lw=5)
    ax[0].grid(True)
    ax[0].set_xlabel('noise std', fontsize=20)
    ax[0].set_ylabel('avg inf', fontsize=20)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    ax[1].clear()
    ax[1].plot(mid_stddevs, ordss.T)
    ax[1].plot(mid_stddevs, np.mean(ordss[0:i+1], axis=0), c='k', lw=5)
    ax[1].grid(True)
    ax[1].set_xlabel('noise std', fontsize=20)
    ax[1].set_ylabel('noise ord', fontsize=20)
    ax[1].tick_params(axis='both', labelsize=20)
    ax[1].set_xscale('log')
    fig.canvas.flush_events()

    print('---------------------------------------')
    print(i+1)
    print(res.message)
    print(env.weighted_inf)
    inf_ls.append(env.weighted_inf)

    if abs(res.fun) < success_tol:
        print('###')
        success_counts += 1

print('---------------------------------------')
print('avg inf:', np.prod(inf_ls) ** (1/len(inf_ls)))
print('success rate:', success_counts / num_trials) #np.round(success_counts / num_trials * 100, 1), '%')
plt.ioff()
plt.show()

"""
num_trials = 1
ss = np.array([1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 0])
infss = np.zeros((len(ss), num_trials, len(stddevs)))
ordss = np.zeros((len(ss), num_trials, len(stddevs) - 1))
pulses = np.zeros((len(ss), num_trials, num_steps))
for i, s_ in enumerate(ss):
    for j in range(num_trials):
        train_noise_weight = s_
        env = TwoQubitEnv(
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
"""
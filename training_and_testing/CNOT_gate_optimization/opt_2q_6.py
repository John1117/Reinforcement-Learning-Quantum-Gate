import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pack.env2q import TwoQubitEnvNM, CNOT, evolve, get_inf
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
max_gate_time = 2
num_steps = 18
Z1_const = 1.0
Z1_noises = np.array([0.0]) #, k*s, -k*s
Z2_const = 1.0
Z2_noises = np.array([0.0])
max_X1_amp = 4.0
max_X2_amp = 4.0
max_J_amp = 4.0
ideal_weight = 1.0
noise_weight = 1.0
dtype = np.float64

J_coef = 1 / max_gate_time * 2


env = TwoQubitEnvNM(
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
    posi_J=True,
    ideal_weight=ideal_weight,
    noise_weight=noise_weight,
    dtype=dtype
)

maxiter = 2 * num_steps * 1000
xatol = 1e-6
fatol = 1e-14

stddevs = np.logspace(-4, 0, 11)
mid_stddevs = (stddevs[1:] + stddevs[:-1]) / 2
num_noise_samples = 30


def env_fn(ctrl_pulse, env, J_coef=1.0):
    ctrl_pulse = np.reshape(ctrl_pulse, (2, env.num_steps))
    env.reset()
    for i in range(env.num_steps):
        coef = np.array([ctrl_pulse[0, i], ctrl_pulse[1, i], J_coef])
        env.step(coef)
    return env.weighted_inf


num_trials = 5
success_tol = 1e-7 #(s * Z1_const * max_gate_time) ** 3 / 6
success_counts = 0
inf_ls = []
infss = np.zeros((num_trials, stddevs.shape[0]))
ordss = np.zeros((num_trials, mid_stddevs.shape[0]))

#plt.ion()
#fig, ax = plt.subplots(1, 2, num=f'T={max_gate_time}, N={num_steps}', figsize=(15, 7.5), tight_layout=True)
#fig.show()

bounds = np.concatenate(
    (
        np.repeat([[-max_X1_amp, max_X1_amp]], num_steps, axis=0),
        np.repeat([[-max_X2_amp, max_X2_amp]], num_steps, axis=0),
        #np.repeat([[0, max_J_amp]], num_steps, axis=0)
    ),
    axis=0
)
for i in range(num_trials):
    x0 = np.concatenate(
        (
            np.random.uniform(low=-max_X1_amp, high=max_X1_amp, size=num_steps),
            np.random.uniform(low=-max_X2_amp, high=max_X2_amp, size=num_steps),
            #np.random.uniform(low=0, high=max_J_amp, size=num_steps)
        )
    )
    res = minimize(
        fun=env_fn,
        x0=x0,
        method='Nelder-Mead',
        bounds=bounds,
        args=(env, J_coef),
        options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol}
    )
    X1X2_pulse = res.x.reshape(2, env.num_steps)
    J_pulse = np.ones((1, env.num_steps)) * J_coef
    pulse = np.concatenate((X1X2_pulse, J_pulse), axis=0)
    infs, ords = noise_test(env, pulse, stddevs, num_noise_samples)
    infss[i] = infs
    ordss[i] = ords

    """ax[0].clear()
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
    fig.canvas.flush_events()"""

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
#plt.ioff()
#plt.show()

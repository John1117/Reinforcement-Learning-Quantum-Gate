# %%
import numpy as np
import matplotlib.pyplot as plt
from pack.utils import joint_first_step
from pack.env import evolve, get_inf, H_gate

# %%
mfn_i = '/file_name'

# %%
fn_i = mfn_i + '/ideal_infs.npy'
inf_i = np.load(fn_i)
# %%
inf_i[-1]

# %%
lw = 2
plt.figure(figsize=(10, 6))
plt.plot(inf_i, 'b-', lw=lw)

fs=24
plt.xlabel('Iteration', fontsize=fs, fontname='Times New Roman')
plt.xticks(fontsize=fs, fontname='Times New Roman')

plt.ylabel('Infidelity', fontsize=fs, fontname='Times New Roman')
plt.yscale('log')
yticks = 10 ** np.array([0, -3, -6, -9, -12, -15.])
plt.yticks(ticks=yticks, fontsize=fs, fontname='Times New Roman')

plt.grid()
plt.show()


# %%
mfn_n = [
    '/file_name'
]
# %%
6845+1309+201+45+403+725

# %%
ideal_infs = np.array([])
for mfn in mfn_n:
    file_name   = mfn + '/ideal_infs.npy'
    stage_ideal_infs = np.load(file_name)
    ideal_infs = np.concatenate([ideal_infs, stage_ideal_infs])

noisy_infs = np.array([])
for mfn in mfn_n:
    file_name   = mfn + '/noise_infs.npy'
    stage_noisy_infs = np.load(file_name)
    noisy_infs = np.concatenate([noisy_infs, stage_noisy_infs])
noisy_infs[noisy_infs < 1e-15] = np.nan

mean_infs = np.array([])
for mfn in mfn_n:
    file_name   = mfn + '/mean_infs.npy'
    stage_mean_infs = np.load(file_name)
    mean_infs = np.concatenate([mean_infs, stage_mean_infs])

weighted_infs = np.array([])
for mfn in mfn_n:
    file_name   = mfn + '/weighted_infs.npy'
    stage_weighted_infs = np.load(file_name)
    weighted_infs = np.concatenate([weighted_infs, stage_weighted_infs])

# %%
ideal_infs[-1]
# %%

plt.figure(figsize=(10, 6))

plt.plot(ideal_infs, 'g', lw=2, label='Ideal')
plt.plot(noisy_infs, 'k', lw=2, label='Noisy')
plt.plot(weighted_infs, 'b', lw=2, label='Weighted')

fs=24
plt.xlabel('Iteration', fontsize=fs, fontname='Times New Roman')
plt.xticks(fontsize=fs, fontname='Times New Roman')

plt.ylabel('Infidelity', fontsize=fs, fontname='Times New Roman')
plt.yscale('log')
yticks = 10 ** np.array([0, -3, -6, -9.])
plt.yticks(ticks=yticks, fontsize=fs, fontname='Times New Roman')

plt.legend(prop={'size': fs, 'family': 'Times New Roman'})
plt.grid()
plt.show()



# %%
pulse_fn_i = mfn_i + '/best_pulse.npy'
pulse_fn_n = mfn_n[-1] + '/best_pulse.npy'

pulse_i = np.load(pulse_fn_i)
pulse_n = np.load(pulse_fn_n)

# %%

p_i = joint_first_step(pulse_i)
p_n = joint_first_step(pulse_n)
time = np.linspace(0, 1, p_i.size)


plt.figure(figsize=(10, 6))
plt.step(time, p_i, 'g-', label='Ideal', lw=lw)
plt.step(time, p_n, 'b-', label='Noisy', lw=lw)


fs = 24
fn = 'Times New Roman'
plt.xlabel('Gate time', fontsize=fs, fontname=fn)
plt.xticks(fontsize=fs, fontname=fn)

plt.ylabel('$\Omega$', fontsize=fs, fontname=fn)
#yticks = [-4, -2, 0, 2, 4]
plt.yticks(fontsize=fs, fontname=fn)
plt.ylim(-4, 4)

plt.legend(prop={'size': fs, 'family': 'Times New Roman'})
plt.grid()
plt.show()


# %%
def noise_test(target_gate, max_gate_time, num_steps, Sz_const, Sx_pulse, stddevs, num_noise_samples):
    step_time = max_gate_time / num_steps
    infs = np.zeros(stddevs.shape[0])
    for i, Sz_stddev in enumerate(stddevs):
        for Sz_noise in np.random.normal(Sz_const, Sz_stddev, num_noise_samples):
            Sz_pulse = np.repeat(Sz_noise, num_steps)
            final_gate = evolve(Sx_pulse, Sz_pulse, step_time)
            infs[i] += get_inf(final_gate, target_gate)
    infs /= num_noise_samples
    ords = np.diff(np.log10(infs)) / np.diff(np.log10(stddevs))
    return infs, ords

# %%
target_gate = H_gate
max_gate_time = 4.
num_steps = 16
Sz_const = 1.
Sx_pulse_i = pulse_i
Sx_pulse_n = pulse_n
stddevs = np.logspace(-8, 0, 21)
num_noise_samples = 10000
infs_i, ords_i = noise_test(target_gate, max_gate_time, num_steps, Sz_const, Sx_pulse_i, stddevs, num_noise_samples)
infs_n, ords_n = noise_test(target_gate, max_gate_time, num_steps, Sz_const, Sx_pulse_n, stddevs, num_noise_samples)

# %%
stddevs = np.logspace(-8, 0, 21)
J2 = 1/2 * (max_gate_time * stddevs) ** 2
J4 = 5/48 * (max_gate_time * stddevs) ** 4


plt.figure(figsize=(10, 6))
plt.plot(stddevs, J2, 'k:', lw=lw, label='Est.  $J_2$')
plt.plot(stddevs, J4, 'k--', lw=lw, label='Est.  $J_4$')
plt.plot(stddevs, infs_i, 'g-', lw=lw, label='Ideal')
plt.plot(stddevs, infs_n, 'b-', lw=lw, label='Noisy')

plt.xscale('log')
plt.xlabel('Noise standard deviation', fontsize=fs, fontname='Times New Roman')
xticks = 10 ** np.array([0, -2, -4, -6, -8.])
plt.xticks(xticks, fontsize=fs, fontname='Times New Roman')

plt.ylabel('Ensemble infidelity', fontsize=fs, fontname='Times New Roman')
plt.yscale('log')
yticks = 10 ** np.array([0, -3, -6, -9, -12, -15.])
plt.yticks(ticks=yticks, fontsize=fs, fontname='Times New Roman')
plt.ylim(yticks[-1]/3, yticks[0]*3)

plt.legend(prop={'size': fs, 'family': 'Times New Roman'})
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
stddevs = (stddevs[1:] + stddevs[:-1])/2
plt.plot(stddevs, ords_i, 'g-', lw=lw, label='Ideal')
plt.plot(stddevs, ords_n, 'b-', lw=lw, label='Noisy')

plt.xscale('log')
plt.xlabel('Noise standard deviation', fontsize=fs, fontname='Times New Roman')
xticks = 10 ** np.array([0, -2, -4, -6, -8.])
plt.xticks(xticks, fontsize=fs, fontname='Times New Roman')

plt.ylabel('Noise order', fontsize=fs, fontname='Times New Roman')
yticks = np.array([0, 1, 2, 3, 4.])
plt.yticks(yticks, fontsize=fs, fontname='Times New Roman')

plt.legend(prop={'size': fs, 'family': 'Times New Roman'})
plt.grid()
plt.show()




# %%
pulse_fn_i = mfn_n[0] + '/best_pulse.npy'
pulse_fn_n = mfn_n[-1] + '/best_pulse.npy'

pulse_i = np.load(pulse_fn_i)
pulse_n = np.load(pulse_fn_n)


# %%
p_i = joint_first_step(pulse_i)
p_n = joint_first_step(pulse_n)
time = np.linspace(0, 1, p_i.size)


plt.figure(figsize=(10, 6))
plt.step(time, p_i, 'c--', label='Before', lw=6)
plt.step(time, p_n, 'b-', label='After', lw=lw)


fs = 24
fn = 'Times New Roman'
plt.xlabel('Gate time', fontsize=fs, fontname=fn)
plt.xticks(fontsize=fs, fontname=fn)

plt.ylabel('$\Omega$', fontsize=fs, fontname=fn)
#yticks = [-4, -2, 0, 2, 4]
plt.yticks(fontsize=fs, fontname=fn)
plt.ylim(-4, 4)

plt.legend(prop={'size': fs, 'family': 'Times New Roman'})
plt.grid()
plt.show()
# %%
target_gate = H_gate
max_gate_time = 4.
num_steps = 16
Sz_const = 1.
Sx_pulse_i = pulse_i
Sx_pulse_n = pulse_n
stddevs = np.logspace(-8, 0, 21)
num_noise_samples = 10000
infs_i, ords_i = noise_test(target_gate, max_gate_time, num_steps, Sz_const, Sx_pulse_i, stddevs, num_noise_samples)
infs_n, ords_n = noise_test(target_gate, max_gate_time, num_steps, Sz_const, Sx_pulse_n, stddevs, num_noise_samples)

# %%
stddevs = np.logspace(-8, 0, 21)
J2 = 1/2 * (max_gate_time * stddevs) ** 2
J4 = 5/48 * (max_gate_time * stddevs) ** 4


plt.figure(figsize=(10, 6))
plt.plot(stddevs, J2, 'k:', lw=lw, label='Est.  $J_2$')
plt.plot(stddevs, J4, 'k--', lw=lw, label='Est.  $J_4$')
plt.plot(stddevs, infs_i, 'c-', lw=lw, label='Before')
plt.plot(stddevs, infs_n, 'b-', lw=lw, label='After')

plt.xscale('log')
plt.xlabel('Noise standard deviation', fontsize=fs, fontname='Times New Roman')
xticks = 10 ** np.array([0, -2, -4, -6, -8.])
plt.xticks(xticks, fontsize=fs, fontname='Times New Roman')

plt.ylabel('Ensemble infidelity', fontsize=fs, fontname='Times New Roman')
plt.yscale('log')
yticks = 10 ** np.array([0, -4, -7, -10.])
plt.yticks(ticks=yticks, fontsize=fs, fontname='Times New Roman')
plt.ylim(yticks[-1]/3, yticks[0]*3)

plt.legend(prop={'size': fs, 'family': 'Times New Roman'})
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
stddevs = (stddevs[1:] + stddevs[:-1])/2
plt.plot(stddevs, ords_i, 'g-', lw=lw, label='Before')
plt.plot(stddevs, ords_n, 'b-', lw=lw, label='After')

plt.xscale('log')
plt.xlabel('Noise standard deviation', fontsize=fs, fontname='Times New Roman')
xticks = 10 ** np.array([0, -2, -4, -6, -8.])
plt.xticks(xticks, fontsize=fs, fontname='Times New Roman')

plt.ylabel('Noise order', fontsize=fs, fontname='Times New Roman')
yticks = np.array([0, 1, 2, 3, 4.])
plt.yticks(yticks, fontsize=fs, fontname='Times New Roman')

plt.legend(prop={'size': fs, 'family': 'Times New Roman'})
plt.grid()
plt.show()
# %%

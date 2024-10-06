# %%
import numpy as np
import matplotlib.pyplot as plt
from pack.utils import joint_first_step
from pack.env import evolve, get_inf, X_gate, H_gate


main_file_names = [
    '/Users/john/personal/myself/Projects/MLQuantGate/record/noisy_h_32/h14/data/6845',
    '/Users/john/personal/myself/Projects/MLQuantGate/record/noisy_h_nw/h1/data/1309',
    '/Users/john/personal/myself/Projects/MLQuantGate/record/noisy_h_nw/h3/data/201',
    '/Users/john/personal/myself/Projects/MLQuantGate/record/noisy_h_nw/h5/data/45',
    '/Users/john/personal/myself/Projects/MLQuantGate/record/noisy_h_nw/h6/data/403',
    '/Users/john/personal/myself/Projects/MLQuantGate/record/noisy_h_nw/h7/data/725'
]

# %%
ideal_infs = np.array([])
for mfn in main_file_names:
    file_name   = mfn + '/ideal_infs.npy'
    stage_ideal_infs = np.load(file_name)
    ideal_infs = np.concatenate([ideal_infs, stage_ideal_infs])

noisy_infs = np.array([])
for mfn in main_file_names:
    file_name   = mfn + '/noise_infs.npy'
    stage_noisy_infs = np.load(file_name)
    noisy_infs = np.concatenate([noisy_infs, stage_noisy_infs])
noisy_infs[noisy_infs < 1e-15] = np.nan

mean_infs = np.array([])
for mfn in main_file_names:
    file_name   = mfn + '/mean_infs.npy'
    stage_mean_infs = np.load(file_name)
    mean_infs = np.concatenate([mean_infs, stage_mean_infs])

weighted_infs = np.array([])
for mfn in main_file_names:
    file_name   = mfn + '/weighted_infs.npy'
    stage_weighted_infs = np.load(file_name)
    weighted_infs = np.concatenate([weighted_infs, stage_weighted_infs])

print(ideal_infs[-1])
print(noisy_infs[-1])
print(mean_infs[-1])
print(weighted_infs[-1])


plt.figure(figsize=(12, 3))
plt.plot(ideal_infs, 'b')

fs=24
plt.xlabel('Iteration', fontsize=fs, fontname='Times New Roman')
plt.xticks(fontsize=fs, fontname='Times New Roman')

plt.ylabel('Ideal infidelity', fontsize=fs, fontname='Times New Roman')
plt.yscale('log')
yticks = 10 ** np.array([0, -5, -10, -15.])
plt.yticks(ticks=yticks, fontsize=fs, fontname='Times New Roman')

plt.grid()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(noisy_infs, 'b')

fs=24
plt.xlabel('Iteration', fontsize=fs, fontname='Times New Roman')
plt.xticks(fontsize=fs, fontname='Times New Roman')

plt.ylabel('Noisy infidelity', fontsize=fs, fontname='Times New Roman')
plt.yscale('log')
yticks = 10 ** np.array([0, -5, -10, -15.])
plt.yticks(ticks=yticks, fontsize=fs, fontname='Times New Roman')

plt.grid()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(mean_infs, 'b')

fs=24
plt.xlabel('Iteration', fontsize=fs, fontname='Times New Roman')
plt.xticks(fontsize=fs, fontname='Times New Roman')

plt.ylabel('Average infidelity', fontsize=fs, fontname='Times New Roman')
plt.yscale('log')
yticks = 10 ** np.array([0, -5, -10, -15.])
plt.yticks(ticks=yticks, fontsize=fs, fontname='Times New Roman')

plt.grid()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(weighted_infs, 'b', lw=1)

fs=24
plt.xlabel('Iteration', fontsize=fs, fontname='Times New Roman')
plt.xticks(fontsize=fs, fontname='Times New Roman')

plt.ylabel('Weighted infidelity', fontsize=fs, fontname='Times New Roman')
plt.yscale('log')
yticks = 10 ** np.array([0, -5, -10, -15.])
plt.yticks(ticks=yticks, fontsize=fs, fontname='Times New Roman')

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

file_name = '/Users/john/personal/myself/Projects/MLQuantGate/record/noisy_h_nw/h7/data/725' + '/best_pulse.npy'
pulse = np.load(file_name)
target_gate = H_gate
max_gate_time = 4.
num_steps = 16
Sz_const = 1.
Sx_pulse = pulse
stddevs = np.logspace(-8, 0, 80)
num_noise_samples = 10000
infs, ords = noise_test(target_gate, max_gate_time, num_steps, Sz_const, Sx_pulse, stddevs, num_noise_samples)

plt.figure(figsize=(10, 4))
plt.plot(stddevs, infs, 'b-')

plt.xscale('log')
plt.xlabel('Noise standard deviation', fontsize=fs, fontname='Times New Roman')
xticks = 10 ** np.array([0, -1, -2, -3, -4, -5, -6, -7, -8.])
plt.xticks(xticks, fontsize=fs, fontname='Times New Roman')

plt.ylabel('Ensemble infidelity', fontsize=fs, fontname='Times New Roman')
plt.yscale('log')
yticks = 10 ** np.array([0, -3, -6, -9, -12, -15.])
plt.yticks(ticks=yticks, fontsize=fs, fontname='Times New Roman')

plt.grid()
plt.show()

plt.figure(figsize=(10, 4))
stddevs = (stddevs[1:] + stddevs[:-1])/2
plt.plot(stddevs, ords, 'b-')

plt.xscale('log')
plt.xlabel('Noise standard deviation', fontsize=fs, fontname='Times New Roman')
xticks = 10 ** np.array([0, -1, -2, -3, -4, -5, -6, -7, -8.])
plt.xticks(xticks, fontsize=fs, fontname='Times New Roman')

plt.ylabel('Noise order', fontsize=fs, fontname='Times New Roman')
plt.yticks(fontsize=fs, fontname='Times New Roman')

plt.grid()
plt.show()
# %%


p = joint_first_step(pulse)
time = np.linspace(0, max_gate_time, p.size)


plt.figure(figsize=(10, 4))
plt.step(time, p, 'b-')

fs = 24
fn = 'Times New Roman'
plt.xlabel('Gate time', fontsize=fs, fontname=fn)
plt.xticks(fontsize=fs, fontname=fn)

plt.ylabel('Pulse amplitude', fontsize=fs, fontname=fn)
#yticks = [-4, -2, 0, 2, 4]
plt.yticks(fontsize=fs, fontname=fn)
plt.ylim(-4, 4)

plt.grid()
plt.show()

# %%

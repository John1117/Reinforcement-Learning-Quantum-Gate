# %%
import numpy as np
import matplotlib.pyplot as plt
from pack.utils import joint_first_step
from pack.env2q import evolve, get_inf, CNOT


main_file_names = [
    '/file_name'
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

plt.ylabel('Infidelity', fontsize=fs, fontname='Times New Roman')
plt.yscale('log')
yticks = 10 ** np.array([0, -5, -10, -15.])
plt.yticks(ticks=yticks, fontsize=fs, fontname='Times New Roman')

plt.grid()
plt.show()


# %%
def noise_test(target_gate, max_gate_time, num_steps, Z1_const, Z2_const, X1_pulse, X2_pulse, J_pulse, stddevs, num_noise_samples):
    step_time = max_gate_time / num_steps
    infs = np.zeros(stddevs.shape[0])
    for i, Z1_stddev in enumerate(stddevs):
        for Z1_noise in np.random.normal(Z1_const, Z1_stddev, num_noise_samples):
            Z1_pulse = np.repeat(Z1_noise, num_steps)
            Z2_pulse = np.repeat(Z2_const, num_steps)
            final_gate = evolve(Z1_pulse, X1_pulse, Z2_pulse, X2_pulse, J_pulse, step_time)
            infs[i] += get_inf(final_gate, target_gate)
    infs /= num_noise_samples
    ords = np.diff(np.log10(infs)) / np.diff(np.log10(stddevs))
    return infs, ords



# %%
target_gate = CNOT
max_gate_time = 2.
num_steps = 8
Z1_const = 1.
Z2_const = 1.
stddevs = np.logspace(-8, 0, 80)
num_noise_samples = 10000
# %%
file_name = main_file_names[-1] + '/best_X1_pulse.npy'
X1_pulse = np.load(file_name)
file_name = main_file_names[-1] + '/best_X2_pulse.npy'
X2_pulse = np.load(file_name)
file_name = main_file_names[-1] + '/best_J_pulse.npy'
J_pulse = np.load(file_name)


# %%

X1p = joint_first_step(X1_pulse)
X2p = joint_first_step(X2_pulse)
Jp = joint_first_step(J_pulse)
time = np.linspace(0, max_gate_time, X1p.size)


plt.figure(figsize=(10, 3))
plt.step(time, X1p, 'b-')

fs = 24
fn = 'Times New Roman'
plt.xlabel('Gate time', fontsize=fs, fontname=fn)
plt.xticks(fontsize=fs, fontname=fn)

plt.ylabel('$q_{X_1}$', fontsize=fs, fontname=fn)
yticks = [-4, -2, 0, 2, 4]
plt.yticks(yticks, fontsize=fs, fontname=fn)
plt.ylim(-4, 4)

plt.grid()
plt.show()

plt.figure(figsize=(10, 3))
plt.step(time, X2p, 'b-')
plt.xlabel('Gate time', fontsize=fs, fontname=fn)
plt.xticks(fontsize=fs, fontname=fn)

plt.ylabel('$q_{X_2}$', fontsize=fs, fontname=fn)
plt.yticks(yticks, fontsize=fs, fontname=fn)
plt.ylim(-4, 4)

plt.grid()
plt.show()

plt.figure(figsize=(10, 3))
plt.step(time, Jp, 'b-')
plt.xlabel('Gate time', fontsize=fs, fontname=fn)
plt.xticks(fontsize=fs, fontname=fn)

plt.ylabel('$q_{Z_1Z_2}$', fontsize=fs, fontname=fn)
plt.yticks(yticks, fontsize=fs, fontname=fn)
plt.ylim(-4, 4)

plt.grid()
plt.show()

# %%
infs, ords = noise_test(target_gate, max_gate_time, num_steps, Z1_const, Z2_const, X1_pulse, X2_pulse, J_pulse, stddevs, num_noise_samples)

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
stddevs_ = (stddevs[1:] + stddevs[:-1])/2
plt.plot(stddevs_, ords, 'b-')

plt.xscale('log')
plt.xlabel('Noise standard deviation', fontsize=fs, fontname='Times New Roman')
xticks = 10 ** np.array([0, -1, -2, -3, -4, -5, -6, -7, -8.])
plt.xticks(xticks, fontsize=fs, fontname='Times New Roman')

plt.ylabel('Noise order', fontsize=fs, fontname='Times New Roman')
plt.yticks(fontsize=fs, fontname='Times New Roman')

plt.grid()
plt.show()

# %%
plt.figure(figsize=(10, 6))
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

plt.figure(figsize=(10, 6))
stddevs_ = (stddevs[1:] + stddevs[:-1])/2
plt.plot(stddevs_, ords, 'b-')

plt.xscale('log')
plt.xlabel('Noise standard deviation', fontsize=fs, fontname='Times New Roman')
xticks = 10 ** np.array([0, -1, -2, -3, -4, -5, -6, -7, -8.])
plt.xticks(xticks, fontsize=fs, fontname='Times New Roman')

plt.ylabel('Noise order', fontsize=fs, fontname='Times New Roman')
yticks = [0, 1, 2]
plt.yticks(yticks, fontsize=fs, fontname='Times New Roman')

plt.grid()
plt.show()
# %%

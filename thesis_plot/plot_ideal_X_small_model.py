# %%
import numpy as np
import matplotlib.pyplot as plt
from pack.utils import joint_first_step
from pack.env import evolve, get_inf, X_gate, H_gate

# %%
main_file_name = '/file_name'

# %%
file_name = main_file_name + '/ideal_infs.npy'
infs = np.load(file_name)

plt.figure(figsize=(10, 6))
plt.plot(infs, 'b-')

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
file_name = main_file_name + '/best_pulse.npy'
pulse = np.load(file_name)
print(pulse[0])
print(1-pulse)

p = joint_first_step(pulse)
time = np.linspace(0, 1, p.size)


plt.figure(figsize=(10, 6))
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

target_gate = X_gate
max_gate_time = 1.
num_steps = 8
Sz_const = 0.
Sx_pulse = pulse
stddevs = np.logspace(-8, 0, 80)
num_noise_samples = 10000
infs, ords = noise_test(target_gate, max_gate_time, num_steps, Sz_const, Sx_pulse, stddevs, num_noise_samples)

plt.figure(figsize=(10, 6))
plt.plot(stddevs, infs, 'b-')

plt.xscale('log')
plt.xlabel('Noise standard deviation', fontsize=fs, fontname='Times New Roman')
xticks = 10 ** np.array([0, -2, -4, -6, -8.])
plt.xticks(xticks, fontsize=fs, fontname='Times New Roman')

plt.ylabel('Ensemble infidelity', fontsize=fs, fontname='Times New Roman')
plt.yscale('log')
yticks = 10 ** np.array([0, -3, -6, -9, -12, -15.])
plt.yticks(ticks=yticks, fontsize=fs, fontname='Times New Roman')

plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
stddevs = (stddevs[1:] + stddevs[:-1])/2
plt.plot(stddevs, ords, 'b-')

plt.xscale('log')
plt.xlabel('Noise standard deviation', fontsize=fs, fontname='Times New Roman')
xticks = 10 ** np.array([0, -2, -4, -6, -8.])
plt.xticks(xticks, fontsize=fs, fontname='Times New Roman')

plt.ylabel('Noise order', fontsize=fs, fontname='Times New Roman')
yticks = np.array([0, 1, 2.])
plt.yticks(yticks, fontsize=fs, fontname='Times New Roman')

plt.grid()
plt.show()
# %%

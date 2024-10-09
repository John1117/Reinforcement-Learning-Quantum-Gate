# %%
import numpy as np
import matplotlib.pyplot as plt
from pack.utils import joint_first_step
from pack.env import evolve, get_inf, X_gate

# %%
mfn_na = '/file_name'
mfn_a = '/file_name'


# %%
inf_fn_na = mfn_na + '/ideal_infs.npy'
inf_fn_a = mfn_a + '/ideal_infs.npy'

inf_na = np.load(inf_fn_na)
inf_a = np.load(inf_fn_a)


# %%

lw = 2
plt.figure(figsize=(10, 6))
plt.plot(inf_na, 'g-', lw=lw, label='No adapt.')
plt.plot(inf_a, 'b-', lw=lw, label='Adapt.')

fs=24
fn = 'Times New Roman'
plt.xlabel('Iteration', fontsize=fs, fontname=fn)
plt.xscale('log')
xticks = 10 ** np.array([0, 1, 2, 3])
plt.xticks(xticks, fontsize=fs, fontname=fn)

plt.ylabel('Infidelity', fontsize=fs, fontname=fn)
plt.yscale('log')
yticks = 10 ** np.array([0, -3, -6, -9, -12, -15.])
plt.yticks(yticks, fontsize=fs, fontname=fn)

plt.legend(prop={'size': fs, 'family': 'Times New Roman'})
plt.grid()
plt.show()


# %%
pulse_fn_na = mfn_na + '/best_pulse.npy'
pulse_fn_a = mfn_a + '/best_pulse.npy'

pulse_na = np.load(pulse_fn_na)
pulse_a = np.load(pulse_fn_a)

# %%

p_na = joint_first_step(pulse_na)
p_a = joint_first_step(pulse_a)
time = np.linspace(0, 1, p_na.size)


plt.figure(figsize=(10, 6))
plt.step(time, p_na, 'g-', lw=lw, label='No adapt.')
plt.step(time, p_a, 'b-', lw=lw, label='Adapt.')

lw = 2
fs = 24
fn = 'Times New Roman'
plt.xlabel('Gate time', fontsize=fs, fontname=fn)
plt.xticks(fontsize=fs, fontname=fn)

plt.ylabel('$\Omega$', fontsize=fs, fontname=fn)
d = 1e-6
plt.ylim(1-d, 1+d)
plt.ticklabel_format(axis='y', style='sci', useOffset=1.0, useMathText=True)
yticks = [1-d, 1-0.5*d, 1, 1+0.5*d, 1+d]
plt.yticks(yticks, fontsize=fs, fontname=fn)
ax = plt.gca()
ax.yaxis.get_offset_text().set_fontsize(fs)
ax.yaxis.get_offset_text().set_fontname(fn)

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

target_gate = X_gate
max_gate_time = 1.
num_steps = 1
Sz_const = 0.
Sx_pulse_na = pulse_na
Sx_pulse_a = pulse_a
stddevs = np.logspace(-8, 0, 21)
num_noise_samples = 10000
infs_na, ords_na = noise_test(target_gate, max_gate_time, num_steps, Sz_const, Sx_pulse_na, stddevs, num_noise_samples)
infs_a, ords_a = noise_test(target_gate, max_gate_time, num_steps, Sz_const, Sx_pulse_a, stddevs, num_noise_samples)

# %%
stddevs = np.logspace(-8, 0, 21)
J2 = 1/2 * (max_gate_time * stddevs) ** 2
J4 = 5/48 * (max_gate_time * stddevs) ** 4


plt.figure(figsize=(10, 6))
plt.plot(stddevs, J2, 'k:', lw=lw, label='Est.  $J_2$')
plt.plot(stddevs, J4, 'k--', lw=lw, label='Est.  $J_4$')
plt.plot(stddevs, infs_na, 'g-', lw=lw, label='No adapt.')
plt.plot(stddevs, infs_a, 'b-', lw=lw, label='Adapt.')

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
plt.plot(stddevs, ords_na, 'g-', lw=lw, label='No adapt.')
plt.plot(stddevs, ords_a, 'b-', lw=lw, label='Adapt.')

plt.xscale('log')
plt.xlabel('Noise standard deviation', fontsize=fs, fontname='Times New Roman')
xticks = 10 ** np.array([0, -2, -4, -6, -8.])
plt.xticks(xticks, fontsize=fs, fontname='Times New Roman')

plt.ylabel('Noise order', fontsize=fs, fontname='Times New Roman')
yticks = np.array([0, 1, 2.])
plt.yticks(yticks, fontsize=fs, fontname='Times New Roman')

plt.legend(prop={'size': fs, 'family': 'Times New Roman'})
plt.grid()
plt.show()
# %%

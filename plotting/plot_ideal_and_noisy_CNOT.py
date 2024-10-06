# %%
import numpy as np
import matplotlib.pyplot as plt
from pack.utils import joint_first_step
from pack.env2q import evolve, get_inf, CNOT

# %%
mfn_i = [
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/ideal_T4_N16/c1/data/3901',
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/ideal_T4_N16/c6/data/1941',
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/ideal_T4_N16/c7/data/10065',
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/ideal_T4_N16/c11/data/8130',
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/ideal_T4_N16/c13/data/18747'
]

# %%
3901+1941+10065+8130+18747


# %%
ideal_infs = np.array([])
for mfn in mfn_i:
    file_name   = mfn + '/ideal_infs.npy'
    stage_ideal_infs = np.load(file_name)
    ideal_infs = np.concatenate([ideal_infs, stage_ideal_infs])

noisy_infs = np.array([])
for mfn in mfn_i:
    file_name   = mfn + '/noise_infs.npy'
    stage_noisy_infs = np.load(file_name)
    noisy_infs = np.concatenate([noisy_infs, stage_noisy_infs])
noisy_infs[noisy_infs < 1e-15] = np.nan

mean_infs = np.array([])
for mfn in mfn_i:
    file_name   = mfn + '/mean_infs.npy'
    stage_mean_infs = np.load(file_name)
    mean_infs = np.concatenate([mean_infs, stage_mean_infs])

weighted_infs = np.array([])
for mfn in mfn_i:
    file_name   = mfn + '/weighted_infs.npy'
    stage_weighted_infs = np.load(file_name)
    weighted_infs = np.concatenate([weighted_infs, stage_weighted_infs])

# %%
ideal_infs[-1]
# %%
plt.figure(figsize=(10, 6))

plt.plot(ideal_infs, 'b', lw=2)

fs=24
plt.xlabel('Iteration', fontsize=fs, fontname='Times New Roman')
plt.xticks(fontsize=fs, fontname='Times New Roman')

plt.ylabel('Infidelity', fontsize=fs, fontname='Times New Roman')
plt.yscale('log')
yticks = 10 ** np.array([0, -3, -6, -9.])
plt.yticks(ticks=yticks, fontsize=fs, fontname='Times New Roman')

plt.grid()
plt.show()


# %%
mfn_n = [
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/noisy_z1z2/c3/data/8597',
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/noisy_z1z2/c11/data/19905',
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/noisy_z1z2/c14/data/14444',
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/noisy_z1z2/c16/data/407',
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/noisy_z1z2/c17/data/17422',
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/noisy_z1z2/c27/data/19522',
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/noisy_z1z2/c30/data/1795',
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/noisy_z1z2/c31/data/3078',
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/noisy_z1z2/c32/data/9606',
    '/Users/john/personal/myself/Projects/MLQuantGate/cnot_record/noisy_z1z2/c34/data/13995'
]
# %%
8597+19905+14444+407+17422+19522+1795+3078+9606+13005
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
X1_pulse_fn_i = mfn_i[-1] + '/best_X1_pulse.npy'
X1_pulse_i = np.load(X1_pulse_fn_i)

X2_pulse_fn_i = mfn_i[-1] + '/best_X2_pulse.npy'
X2_pulse_i = np.load(X2_pulse_fn_i)

J_pulse_fn_i = mfn_i[-1] + '/best_J_pulse.npy'
J_pulse_i = np.load(J_pulse_fn_i)

X1_pulse_fn_n = mfn_n[-1] + '/best_X1_pulse.npy'
X1_pulse_n = np.load(X1_pulse_fn_n)

X2_pulse_fn_n = mfn_n[-1] + '/best_X2_pulse.npy'
X2_pulse_n = np.load(X2_pulse_fn_n)

J_pulse_fn_n = mfn_n[-1] + '/best_J_pulse.npy'
J_pulse_n = np.load(J_pulse_fn_n)


# %%

x1p_i = joint_first_step(X1_pulse_i)
x2p_i = joint_first_step(X2_pulse_i)
jp_i = joint_first_step(J_pulse_i)

x1p_n = joint_first_step(X1_pulse_n)
x2p_n = joint_first_step(X2_pulse_n)
jp_n = joint_first_step(J_pulse_n)

time = np.linspace(0, 1, x1p_i.size)

# %%

lw = 2
sz = (10, 4)
plt.figure(figsize=sz)
plt.step(time, x1p_i, 'g-', label='Ideal', lw=lw)
plt.step(time, x1p_n, 'b-', label='Noisy', lw=lw)


fs = 24
fn = 'Times New Roman'
plt.xlabel('Gate time', fontsize=fs, fontname=fn)
plt.xticks(fontsize=fs, fontname=fn)

plt.ylabel('$\Omega_1$', fontsize=fs, fontname=fn)
yticks = [-4, -2, 0, 2, 4]
plt.yticks(yticks, fontsize=fs, fontname=fn)
plt.ylim(-4, 4)

plt.legend(prop={'size': fs, 'family': 'Times New Roman'})
plt.grid()
plt.show()

plt.figure(figsize=sz)
plt.step(time, x2p_i, 'g-', label='Ideal', lw=lw)
plt.step(time, x2p_n, 'b-', label='Noisy', lw=lw)


fs = 24
fn = 'Times New Roman'
plt.xlabel('Gate time', fontsize=fs, fontname=fn)
plt.xticks(fontsize=fs, fontname=fn)

plt.ylabel('$\Omega_2$', fontsize=fs, fontname=fn)
yticks = [-4, -2, 0, 2, 4]
plt.yticks(yticks, fontsize=fs, fontname=fn)
plt.ylim(-4, 4)

plt.grid()
plt.show()

plt.figure(figsize=sz)
plt.step(time, jp_i, 'g-', label='Ideal', lw=lw)
plt.step(time, jp_n, 'b-', label='Noisy', lw=lw)


fs = 24
fn = 'Times New Roman'
plt.xlabel('Gate time', fontsize=fs, fontname=fn)
plt.xticks(fontsize=fs, fontname=fn)

plt.ylabel('$J$', fontsize=fs, fontname=fn)
yticks = [-4, -2, 0, 2, 4]
plt.yticks(yticks, fontsize=fs, fontname=fn)
plt.ylim(-4, 4)

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
max_gate_time = 4.
num_steps = 16
Z1_const = 1.
Z2_const = 1.
stddevs = np.logspace(-8, 0, 21)
num_noise_samples = 100
infs_i, ords_i = noise_test(target_gate, max_gate_time, num_steps, Z1_const, Z2_const, X1_pulse_i, X2_pulse_i, J_pulse_i, stddevs, num_noise_samples)
infs_n, ords_n = noise_test(target_gate, max_gate_time, num_steps, Z1_const, Z2_const, X1_pulse_n, X2_pulse_n, J_pulse_n, stddevs, num_noise_samples)

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
yticks = 10 ** np.array([0, -2, -4, -6, -8, -10.])
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
yticks = np.array([0, 1, 2, 3.])
plt.yticks(yticks, fontsize=fs, fontname='Times New Roman')

plt.legend(prop={'size': fs, 'family': 'Times New Roman'})
plt.grid()
plt.show()

# %%
print(infs_i[0])
print(infs_n[0])



# %%
# %%
X1_pulse_fn_i = mfn_n[1] + '/best_X1_pulse.npy'
X1_pulse_i = np.load(X1_pulse_fn_i)

X2_pulse_fn_i = mfn_n[1] + '/best_X2_pulse.npy'
X2_pulse_i = np.load(X2_pulse_fn_i)

J_pulse_fn_i = mfn_n[1] + '/best_J_pulse.npy'
J_pulse_i = np.load(J_pulse_fn_i)

X1_pulse_fn_n = mfn_n[-1] + '/best_X1_pulse.npy'
X1_pulse_n = np.load(X1_pulse_fn_n)

X2_pulse_fn_n = mfn_n[-1] + '/best_X2_pulse.npy'
X2_pulse_n = np.load(X2_pulse_fn_n)

J_pulse_fn_n = mfn_n[-1] + '/best_J_pulse.npy'
J_pulse_n = np.load(J_pulse_fn_n)


# %%

x1p_i = joint_first_step(X1_pulse_i)
x2p_i = joint_first_step(X2_pulse_i)
jp_i = joint_first_step(J_pulse_i)

x1p_n = joint_first_step(X1_pulse_n)
x2p_n = joint_first_step(X2_pulse_n)
jp_n = joint_first_step(J_pulse_n)

# %%

lw = 2
sz = (10, 4)
plt.figure(figsize=sz)
plt.step(time, x1p_i, 'c--', label='Before', lw=6)
plt.step(time, x1p_n, 'b-', label='After', lw=lw)


fs = 24
fn = 'Times New Roman'
plt.xlabel('Gate time', fontsize=fs, fontname=fn)
plt.xticks(fontsize=fs, fontname=fn)

plt.ylabel('$\Omega_1$', fontsize=fs, fontname=fn)
yticks = [-4, -2, 0, 2, 4]
plt.yticks(yticks, fontsize=fs, fontname=fn)
plt.ylim(-4, 4)

plt.legend(prop={'size': fs, 'family': 'Times New Roman'})
plt.grid()
plt.show()

plt.figure(figsize=sz)
plt.step(time, x2p_i, 'c--', label='Ideal', lw=6)
plt.step(time, x2p_n, 'b-', label='Noisy', lw=lw)


fs = 24
fn = 'Times New Roman'
plt.xlabel('Gate time', fontsize=fs, fontname=fn)
plt.xticks(fontsize=fs, fontname=fn)

plt.ylabel('$\Omega_2$', fontsize=fs, fontname=fn)
yticks = [-4, -2, 0, 2, 4]
plt.yticks(yticks, fontsize=fs, fontname=fn)
plt.ylim(-4, 4)

plt.grid()
plt.show()

plt.figure(figsize=sz)
plt.step(time, jp_i, 'c--', label='Ideal', lw=6)
plt.step(time, jp_n, 'b-', label='Noisy', lw=lw)


fs = 24
fn = 'Times New Roman'
plt.xlabel('Gate time', fontsize=fs, fontname=fn)
plt.xticks(fontsize=fs, fontname=fn)

plt.ylabel('$J$', fontsize=fs, fontname=fn)
yticks = [-4, -2, 0, 2, 4]
plt.yticks(yticks, fontsize=fs, fontname=fn)
plt.ylim(-4, 4)

plt.grid()
plt.show()


# %%
stddevs = np.logspace(-8, 0, 21)
num_noise_samples = 10000
infs_i, ords_i = noise_test(target_gate, max_gate_time, num_steps, Z1_const, Z2_const, X1_pulse_i, X2_pulse_i, J_pulse_i, stddevs, num_noise_samples)
infs_n, ords_n = noise_test(target_gate, max_gate_time, num_steps, Z1_const, Z2_const, X1_pulse_n, X2_pulse_n, J_pulse_n, stddevs, num_noise_samples)

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

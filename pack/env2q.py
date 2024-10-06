import numpy as np
import scipy.linalg as la
from pack.utils import DataSpec, matrix_dot

I = np.array([[1, 0], [0, 1]], np.complex128)
Z = np.array([[1, 0], [0, -1]], np.complex128)
X = np.array([[0, 1], [1, 0]], np.complex128)
H = np.array([[1, 1], [1, -1]], np.complex128) * 2 ** -.5
I2 = np.kron(I, I)
X1 = np.kron(X, I)
Z1 = np.kron(Z, I)
X2 = np.kron(I, X)
Z2 = np.kron(I, Z)
ZZ = np.kron(Z, Z)
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], np.complex128)


def evolve(
        Z1_pulse: np.ndarray,
        X1_pulse: np.ndarray,
        Z2_pulse: np.ndarray,
        X2_pulse: np.ndarray,
        J_pulse: np.ndarray,
        dt=1.0,
        init_gate=I2
):
    final_gate = init_gate
    for Z1_coef, X1_coef, Z2_coef, X2_coef, J_coef in zip(Z1_pulse, X1_pulse, Z2_pulse, X2_pulse, J_pulse):
        h = (X1_coef*X1 + Z1_coef*Z1 + X2_coef*X2 + Z2_coef*Z2 + J_coef*ZZ)/2
        final_gate = la.expm(-1j * np.pi * dt * h) @ final_gate
    return final_gate


def get_inf(
        final_gate: np.ndarray,
        target_gate: np.ndarray
):
    final_gate_norm = matrix_dot(final_gate, final_gate)
    gate_dot = matrix_dot(final_gate, target_gate)
    inf = 1 - gate_dot ** 2 / (target_gate.shape[0] * final_gate_norm)
    return max(inf, 1e-15)


class TwoQubitEnv(object):

    def __init__(
            self,
            target_gate=CNOT,
            max_gate_time=1.0,
            num_steps=1,
            Z1_const=0.0,
            Z1_noises=np.array([0.0]),
            Z2_const=0.0,
            Z2_noises=np.array([0.0]),
            max_X1_amp=4.0,
            max_X2_amp=4.0,
            max_J_amp=4.0,
            posi_J=False,
            fix_J=False,
            fix_J_value=None,
            ideal_weight=1.0,
            noise_weight=1.0,
            dtype=np.float64
    ):
        # gate info: gate type, gate time, num of step pulses, time in each step pulse
        self.target_gate = target_gate
        self.max_gate_time = max_gate_time
        self.num_steps = num_steps
        self.step_time = max_gate_time / num_steps
        self.gate_time = 0.0

        # pulse coef info
        self.Z1_const = Z1_const
        self.Z1_noises = Z1_noises
        self.Z2_const = Z2_const
        self.Z2_noises = Z2_noises
        self.num_of_Z1_noises = self.Z1_noises.shape[0]
        self.num_of_Z2_noises = self.Z2_noises.shape[0]
        self.max_X1_amp = max_X1_amp
        self.max_X2_amp = max_X2_amp
        self.max_J_amp = max_J_amp
        self.posi_J = posi_J
        self.fix_J = fix_J
        self.fix_J_value = fix_J_value

        # initial pulses
        self.Z1_pulses = Z1_const + np.repeat(Z1_noises.reshape(-1, 1), num_steps, axis=1)
        self.Z2_pulses = Z2_const + np.repeat(Z2_noises.reshape(-1, 1), num_steps, axis=1)

        self.X1_pulse = np.zeros(num_steps, dtype)
        self.X2_pulse = np.zeros(num_steps, dtype)
        if fix_J:
            self.J_pulse = np.ones(num_steps, dtype) * fix_J_value
        else:
            self.J_pulse = np.zeros(num_steps, dtype)

        # weights for making reward from infs
        self.infs = np.ones(self.num_of_Z1_noises * self.num_of_Z2_noises, dtype)
        self.ideal_weight = ideal_weight
        self.noise_weight = noise_weight
        self.mean_inf = self.infs.mean()
        self.ideal_inf = self.infs[0]
        self.noise_inf = self.mean_inf - self.ideal_inf
        self.weighted_inf = self.ideal_weight * self.ideal_inf + self.noise_weight * self.noise_inf

        # date type of env
        self.dtype = dtype

        # env spec: observation spec (max_gate_timeBD), action spec, reward spec
        if posi_J:
            self.observation_spec = DataSpec(
                shape=num_steps * 3 + 1,
                dtype=dtype,
                mins=[-max_X1_amp for _ in range(num_steps)] + [-max_X2_amp for _ in range(num_steps)] + [0 for _ in range(num_steps)] + [0],
                maxs=[max_X1_amp for _ in range(num_steps)] + [max_X2_amp for _ in range(num_steps)] + [max_J_amp for _ in range(num_steps)] + [max_gate_time])

            self.action_spec = DataSpec(
                shape=3,
                dtype=dtype,
                mins=[-max_X1_amp, -max_X2_amp, 0],
                maxs=[max_X1_amp, max_X2_amp, max_J_amp])

        elif fix_J:
            self.observation_spec = DataSpec(
                shape=num_steps * 2 + 1,
                dtype=dtype,
                mins=[-max_X1_amp for _ in range(num_steps)] + [-max_X2_amp for _ in range(num_steps)] + [0],
                maxs=[max_X1_amp for _ in range(num_steps)] + [max_X2_amp for _ in range(num_steps)] + [max_gate_time])

            self.action_spec = DataSpec(
                shape=2,
                dtype=dtype,
                mins=[-max_X1_amp, -max_X2_amp],
                maxs=[max_X1_amp, max_X2_amp])
        else:
            self.observation_spec = DataSpec(
                shape=num_steps * 3 + 1,
                dtype=dtype,
                mins=[-max_X1_amp for _ in range(num_steps)] + [-max_X2_amp for _ in range(num_steps)] + [-max_J_amp for _ in range(num_steps)] + [0],
                maxs=[max_X1_amp for _ in range(num_steps)] + [max_X2_amp for _ in range(num_steps)] + [max_J_amp for _ in range(num_steps)] + [max_gate_time])

            self.action_spec = DataSpec(
                shape=3,
                dtype=dtype,
                mins=[-max_X1_amp, -max_X2_amp, -max_J_amp],
                maxs=[max_X1_amp, max_X2_amp, max_J_amp])

        self.reward_spec = DataSpec(
            shape=1,
            dtype=dtype,
            mins=0,
            maxs=15)

        # initial vars that occur in reset fn
        self.step_idx = 0
        self.end = False

    def reset(self):
        self.gate_time = 0.0
        self.X1_pulse = np.zeros(self.num_steps, self.dtype)
        self.X2_pulse = np.zeros(self.num_steps, self.dtype)
        if not self.fix_J:
            self.J_pulse = np.zeros(self.num_steps, self.dtype)
        self.infs = np.ones(self.num_of_Z1_noises * self.num_of_Z2_noises, self.dtype)
        self.mean_inf = self.infs.mean()
        self.ideal_inf = self.infs[0]
        self.noise_inf = self.mean_inf - self.ideal_inf
        self.weighted_inf = self.ideal_weight * self.ideal_inf + self.noise_weight * self.noise_inf
        self.step_idx = 0
        self.end = False
        return self.get_observation()

    def step(self, action):
        if self.end:
            self.reset()
        action = np.clip(np.squeeze(action), self.action_spec.mins, self.action_spec.maxs)
        # add the latest step pulse to Sx pulse
        self.X1_pulse[self.step_idx] = action[0]
        self.X2_pulse[self.step_idx] = action[1]
        if not self.fix_J:
            self.J_pulse[self.step_idx] = action[2]

        self.step_idx += 1
        self.gate_time = self.step_idx * self.step_time
        observation = self.get_observation()

        # if env is at the final step (end = True), do evolution and calculate inf for each sz pulse
        self.end = True if self.step_idx == self.num_steps else False
        if self.end:
            self.set_infs()
            reward = self.get_final_reward()
        else:
            reward = np.array([[0]], self.dtype)
        return observation, reward

    def set_infs(self):
        i = 0
        for Z1_pulse in self.Z1_pulses:
            for Z2_pulse in self.Z2_pulses:
                final_gate = evolve(Z1_pulse, self.X1_pulse, Z2_pulse, self.X2_pulse, self.J_pulse, self.step_time)
                self.infs[i] = get_inf(final_gate, self.target_gate)
                i += 1

    def get_observation(self):
        if self.fix_J:
            return np.concatenate((self.X1_pulse, self.X2_pulse, [self.gate_time]), dtype=self.dtype).reshape(1, self.observation_spec.shape)
        else:
            return np.concatenate((self.X1_pulse, self.X2_pulse, self.J_pulse, [self.gate_time]), dtype=self.dtype).reshape(1, self.observation_spec.shape)

    def get_final_reward(self):
        self.mean_inf = self.infs.mean()
        self.ideal_inf = self.infs[0]
        self.noise_inf = self.mean_inf - self.ideal_inf
        self.weighted_inf = self.ideal_weight * self.ideal_inf + self.noise_weight * self.noise_inf
        return -np.log10([[self.weighted_inf]], dtype=self.dtype)


def evolve_NM(
        Z1_pulse: np.ndarray,
        X1_pulse: np.ndarray,
        Z2_pulse: np.ndarray,
        X2_pulse: np.ndarray,
        J_pulse: np.ndarray,
        dt=1.0,
        init_gate=I2
):
    final_gate = init_gate
    for Z1_coef, X1_coef, Z2_coef, X2_coef, J_coef in zip(Z1_pulse, X1_pulse, Z2_pulse, X2_pulse, J_pulse):
        h = (X1_coef*X1 + Z1_coef*Z1 + X2_coef*X2 + Z2_coef*Z2 + J_coef*ZZ)/2
        final_gate = la.expm(-1j * np.pi * dt * h) @ final_gate
    return final_gate


def get_inf_NM(
        final_gate: np.ndarray,
        target_gate: np.ndarray
):
    final_gate_norm = matrix_dot(final_gate, final_gate)
    gate_dot = matrix_dot(final_gate, target_gate)
    inf = 1 - gate_dot ** 2 / (target_gate.shape[0] * final_gate_norm)
    return max(inf, 1e-15)


class TwoQubitEnvNM(object):

    def __init__(
            self,
            target_gate=CNOT,
            max_gate_time=1.0,
            num_steps=1,
            Z1_const=0.0,
            Z1_noises=np.array([0.0]),
            Z2_const=0.0,
            Z2_noises=np.array([0.0]),
            max_X1_amp=4.0,
            max_X2_amp=4.0,
            max_J_amp=4.0,
            posi_J=False,
            ideal_weight=1.0,
            noise_weight=1.0,
            dtype=np.float64
    ):
        # gate info: gate type, gate time, num of step pulses, time in each step pulse
        self.target_gate = target_gate
        self.max_gate_time = max_gate_time
        self.num_steps = num_steps
        self.step_time = max_gate_time / num_steps
        self.gate_time = 0.0

        # pulse coef info
        self.Z1_const = Z1_const
        self.Z1_noises = Z1_noises
        self.Z2_const = Z2_const
        self.Z2_noises = Z2_noises
        self.num_of_Z1_noises = self.Z1_noises.shape[0]
        self.num_of_Z2_noises = self.Z2_noises.shape[0]
        self.max_X1_amp = max_X1_amp
        self.max_X2_amp = max_X2_amp
        self.max_J_amp = max_J_amp

        # initial pulses
        self.Z1_pulses = Z1_const + np.repeat(Z1_noises.reshape(-1, 1), num_steps, axis=1)
        self.Z2_pulses = Z2_const + np.repeat(Z2_noises.reshape(-1, 1), num_steps, axis=1)

        self.X1_pulse = np.zeros(num_steps, dtype)
        self.X2_pulse = np.zeros(num_steps, dtype)
        self.J_pulse = np.zeros(num_steps, dtype)

        # weights for making reward from infs
        self.infs = np.ones(self.num_of_Z1_noises * self.num_of_Z2_noises, dtype)
        self.ideal_weight = ideal_weight
        self.noise_weight = noise_weight
        self.mean_inf = self.infs.mean()
        self.ideal_inf = self.infs[0]
        self.noise_inf = self.mean_inf - self.ideal_inf
        self.weighted_inf = self.ideal_weight * self.ideal_inf + self.noise_weight * self.noise_inf

        # date type of env
        self.dtype = dtype

        # env spec: observation spec (max_gate_timeBD), action spec, reward spec
        self.observation_spec = DataSpec(
            shape=num_steps * 3 + 1,
            dtype=dtype,
            mins=[-max_X1_amp for _ in range(num_steps)] + [-max_X2_amp for _ in range(num_steps)] + [0 if posi_J else -max_J_amp for _ in range(num_steps)] + [0],
            maxs=[max_X1_amp for _ in range(num_steps)] + [max_X2_amp for _ in range(num_steps)] + [max_J_amp for _ in range(num_steps)] + [max_gate_time])

        self.action_spec = DataSpec(
            shape=3,
            dtype=dtype,
            mins=[-max_X1_amp, -max_X2_amp, 0 if posi_J else -max_J_amp],
            maxs=[max_X1_amp, max_X2_amp, max_J_amp])

        self.reward_spec = DataSpec(
            shape=1,
            dtype=dtype,
            mins=0,
            maxs=15)

        # initial vars that occur in reset fn
        self.step_idx = 0
        self.end = False

    def reset(self):
        self.gate_time = 0.0
        self.X1_pulse = np.zeros(self.num_steps, self.dtype)
        self.X2_pulse = np.zeros(self.num_steps, self.dtype)
        self.J_pulse = np.zeros(self.num_steps, self.dtype)
        self.infs = np.ones(self.num_of_Z1_noises * self.num_of_Z2_noises, self.dtype)
        self.mean_inf = self.infs.mean()
        self.ideal_inf = self.infs[0]
        self.noise_inf = self.mean_inf - self.ideal_inf
        self.weighted_inf = self.ideal_weight * self.ideal_inf + self.noise_weight * self.noise_inf
        self.step_idx = 0
        self.end = False
        return self.get_observation()

    def step(self, action):
        if self.end:
            self.reset()
        action = np.clip(np.squeeze(action), self.action_spec.mins, self.action_spec.maxs)
        # add the latest step pulse to Sx pulse
        self.X1_pulse[self.step_idx] = action[0]
        self.X2_pulse[self.step_idx] = action[1]

        self.J_pulse[self.step_idx] = action[2]

        self.step_idx += 1
        self.gate_time = self.step_idx * self.step_time
        observation = self.get_observation()

        # if env is at the final step (end = True), do evolution and calculate inf for each sz pulse
        self.end = True if self.step_idx == self.num_steps else False
        if self.end:
            self.set_infs()
            reward = self.get_final_reward()
        else:
            reward = np.array([[0]], self.dtype)
        return observation, reward

    def set_infs(self):
        i = 0
        for Z1_pulse in self.Z1_pulses:
            for Z2_pulse in self.Z2_pulses:
                final_gate = evolve_NM(Z1_pulse, self.X1_pulse, Z2_pulse, self.X2_pulse, self.J_pulse, self.step_time)
                self.infs[i] = get_inf_NM(final_gate, self.target_gate)
                i += 1

    def get_observation(self):
        return np.concatenate((self.X1_pulse, self.X2_pulse, self.J_pulse, [self.gate_time]), dtype=self.dtype).reshape(1, self.observation_spec.shape)

    def get_final_reward(self):
        self.mean_inf = self.infs.mean()
        self.ideal_inf = self.infs[0]
        self.noise_inf = self.mean_inf - self.ideal_inf
        self.weighted_inf = self.ideal_weight * self.ideal_inf + self.noise_weight * self.noise_inf
        return -np.log10([[self.weighted_inf]], dtype=self.dtype)


if __name__ == '__main__':
    env = TwoQubitEnvNM(max_gate_time=4, num_steps=16)


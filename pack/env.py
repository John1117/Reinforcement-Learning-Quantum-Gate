import numpy as np
from pack.utils import DataSpec, matrix_dot

X_gate = np.array([[0, 1], [1, 0]], np.complex128)
H_gate = np.array([[1, 1], [1, -1]], np.complex128)*2**-.5

Id = np.array([[1, 0], [0, 1]], np.complex128)
Sx = np.array([[0, 1], [1, 0]], np.complex128)
Sz = np.array([[1, 0], [0, -1]], np.complex128)


def evolve(
        Sx_pulse: np.ndarray,
        Sz_pulse: np.ndarray,
        dt=1.0,
        init_gate=Id
):
    final_gate = init_gate
    for Sx_coef, Sz_coef in zip(Sx_pulse, Sz_pulse):
        n_vec = np.array([Sx_coef, 0, Sz_coef])
        n_norm = np.linalg.norm(n_vec)
        n_hat = n_vec if n_norm == 0 else n_vec / n_norm
        theta = -n_norm * dt * np.pi / 2
        n_dot_S = (n_hat[0] * Sx + n_hat[2] * Sz)
        final_gate = (np.cos(theta) * Id + 1j * np.sin(theta) * n_dot_S) @ final_gate
    return final_gate


def get_inf(
        final_gate: np.ndarray,
        target_gate: np.ndarray
):
    final_gate_norm = matrix_dot(final_gate, final_gate)
    gate_dot = matrix_dot(final_gate, target_gate)
    inf = 1 - gate_dot ** 2 / (2 * final_gate_norm)
    return max(inf, 1e-15)


class BaseStepPulseEnv(object):
    
    def __init__(
            self,
            target_gate=X_gate,
            max_gate_time=1.0,
            num_steps=10,
            Sz_const=0.0,
            Sz_noise_samples=np.array([0.0, 0.1, -0.1]),
            max_Sx_amp=4.0,
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
        self.Sz_const = Sz_const
        self.Sz_noise_samples = Sz_noise_samples
        self.num_noise_samples = self.Sz_noise_samples.shape[0]
        self.max_Sx_amp = max_Sx_amp
        # initial pulses
        self.Sz_pulses = Sz_const + np.repeat(Sz_noise_samples.reshape(-1, 1), num_steps, axis=1)
        self.Sx_pulse = np.zeros(num_steps, dtype)

        # weights for making reward from infs
        self.infs = np.ones(self.num_noise_samples, dtype)
        self.ideal_weight = ideal_weight
        self.noise_weight = noise_weight
        self.mean_inf = self.infs.mean()
        self.ideal_inf = self.infs[0]
        self.noise_inf = self.mean_inf - self.ideal_inf
        self.weighted_inf = self.ideal_weight * self.ideal_inf + self.noise_weight * self.noise_inf
        
        # date type of env
        self.dtype = dtype

        #env spec: observation spec (TBD), action spec, reward spec
        self.observation_spec = None
        self.action_spec = DataSpec(
            shape=1,
            dtype=dtype,
            mins=-max_Sx_amp,
            maxs=max_Sx_amp)

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
        self.Sx_pulse = np.zeros(self.num_steps, self.dtype)
        self.infs = np.ones(self.num_noise_samples, self.dtype)
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

        # add the latest step pulse to Sx pulse
        self.Sx_pulse[self.step_idx] = np.clip(np.squeeze(action), self.action_spec.mins, self.action_spec.maxs)

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
        for idx, Sz_pulse in enumerate(self.Sz_pulses):
            final_gate = evolve(self.Sx_pulse, Sz_pulse, self.step_time)
            self.infs[idx] = get_inf(final_gate, self.target_gate)

    def get_observation(self):
        pass

    def get_final_reward(self):
        self.mean_inf = self.infs.mean()
        self.ideal_inf = self.infs[0]
        self.noise_inf = self.mean_inf - self.ideal_inf
        self.weighted_inf = self.ideal_weight * self.ideal_inf + self.noise_weight * self.noise_inf
        return -np.log10([[self.weighted_inf]], dtype=self.dtype)


class PulseSeqEnv(BaseStepPulseEnv):

    def __init__(
            self,
            target_gate=X_gate,
            max_gate_time=1.0,
            num_steps=10,
            Sz_const=0.0,
            Sz_noise_samples=np.array([0.0, 0.1, -0.1]),
            max_Sx_amp=4.0,
            ideal_weight=1.0,
            noise_weight=1.0,
            dtype=np.float64
    ):
        super().__init__(
            target_gate,
            max_gate_time,
            num_steps,
            Sz_const,
            Sz_noise_samples,
            max_Sx_amp,
            ideal_weight,
            noise_weight,
            dtype
        )
        self.observation_spec = DataSpec(
            shape=num_steps + 1,
            dtype=dtype,
            mins=[-max_Sx_amp] * num_steps + [0],
            maxs=[max_Sx_amp] * num_steps + [max_gate_time])

    def get_observation(self):
        return np.concatenate((self.Sx_pulse, [self.gate_time]), dtype=self.dtype).reshape(1, self.observation_spec.shape)


class PrePulseEnv(BaseStepPulseEnv):

    def __init__(
            self,
            target_gate=X_gate,
            max_gate_time=1.0,
            num_steps=10,
            Sz_const=0.0,
            Sz_noise_samples=np.array([0.0, 0.1, -0.1]),
            max_Sx_amp=4.0,
            ideal_weight=1.0,
            noise_weight=1.0,
            dtype=np.float64
    ):
        super().__init__(
            target_gate,
            max_gate_time,
            num_steps,
            Sz_const,
            Sz_noise_samples,
            max_Sx_amp,
            ideal_weight,
            noise_weight,
            dtype
        )
        self.observation_spec = DataSpec(
            shape=2,
            dtype=dtype,
            mins=[-max_Sx_amp, 0],
            maxs=[max_Sx_amp, max_gate_time])

    def get_observation(self):
        return np.concatenate(([self.Sx_pulse[self.step_idx - 1]], [self.gate_time]), dtype=self.dtype).reshape(1, self.observation_spec.shape)


# TODO: Fourier transform of pulse as observation

if __name__ == '__main__':
    base_env = BaseStepPulseEnv()
    seq_env = PulseSeqEnv()
    pre_env = PrePulseEnv()

    base_obs0 = base_env.reset()
    seq_obs0 = seq_env.reset()
    pre_obs0 = pre_env.reset()
    print(base_obs0, seq_obs0, pre_obs0)

    for i in range(1, 20):
        print(i)
        base_obs, base_rwd = base_env.step(i)
        seq_obs, seq_rwd = seq_env.step(i)
        pre_obs, pre_rwd = pre_env.step(i)
        print(base_obs, seq_obs, pre_obs)
        print(base_rwd, seq_rwd, pre_rwd)

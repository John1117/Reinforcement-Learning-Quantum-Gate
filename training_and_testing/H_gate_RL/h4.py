import numpy as np
import tensorflow as tf
from pack.env import PulseSeqEnv, X_gate, H_gate
from pack.agent import build_ppo
from pack.train import train_agent
from pack.utils import load_network

s = 0.01
k = 1.5 ** 0.5

target_gate = H_gate
max_gate_time = 4
num_steps = 16
Sz_const = 1.
Sz_noise_samples = np.array([0., k*s, -k*s])
max_Sx_amp = 4.
train_ideal_weight = 1.
train_noise_weight = 1.
dtype = np.float64

init_learning_rate = 3e-4
actor_network_config = (32, 32)
value_network_config = (32, 32)
init_action_stddev = np.array(1e-3, dtype)

max_iter = 2000
num_collects = 30
num_epochs = 5
batch_size = None

test_ideal_weight = train_ideal_weight
test_noise_weight = train_noise_weight

noise_test_interval = 100
num_noise_test_samples = 300

revolution_interval = np.inf
adapt_interval = np.inf
adaptive_learning_rate_exponent = None
adaptive_action_stddev_exponent = None
max_learning_rate = init_learning_rate
max_action_stddev = init_action_stddev
if adaptive_action_stddev_exponent:
    train_stddev = False
else:
    train_stddev = True

save_interval = 100
board_update_interval = 50

main_path = '/Users/john/personal/myself/projects/machine_learning_quantum_gate/record/noisy_h_32_blr'
model_name = 'h4'
path_name = main_path + '/' + model_name


tf.keras.backend.set_floatx('float64')
with tf.device('/cpu'):
    env = PulseSeqEnv(
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

    agent = build_ppo(
        env=env,
        init_learning_rate=init_learning_rate,
        actor_network_config=actor_network_config,
        value_network_config=value_network_config,
        init_action_stddev=init_action_stddev,
        train_stddev=train_stddev
    )
    old_main_path = '/Users/john/personal/myself/projects/machine_learning_quantum_gate/record/noisy_h_32'
    agent = load_network(agent, old_main_path + '/h14/network/6845', use_stddev_proj_network=False)

    train_agent(
        env=env,
        agent=agent,
        max_iter=max_iter,
        num_collects=num_collects,
        num_epochs=num_epochs,
        batch_size=batch_size,
        ideal_weight=test_ideal_weight,
        noise_weight=test_noise_weight,
        noise_test_interval=noise_test_interval,
        num_noise_test_samples=num_noise_test_samples,
        revolution_interval=revolution_interval,
        adapt_interval=adapt_interval,
        adaptive_learning_rate_exponent=adaptive_learning_rate_exponent,
        adaptive_action_stddev_exponent=adaptive_action_stddev_exponent,
        max_learning_rate=max_learning_rate,
        max_action_stddev=max_action_stddev,
        save_interval=save_interval,
        path_name=path_name,
        board_update_interval=board_update_interval,
        model_name=model_name
    )

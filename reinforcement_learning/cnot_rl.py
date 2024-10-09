import numpy as np
import tensorflow as tf
from pack.env2q import TwoQubitEnv, CNOT
from pack.agent import build_ppo
from pack.train2q import train_agent_2q
from pack.utils import load_network

s = 0.01
k = 1.5 ** 0.5

target_gate = CNOT
max_gate_time = 4
num_steps = 16
Z1_const = 1.0
Z1_noises = np.array([0.0, k*s, -k*s]) #, k*s, -k*s
Z2_const = 1.0
Z2_noises = np.array([0.0, k*s, -k*s])
max_X1_amp = 4.0
max_X2_amp = 4.0
max_J_amp = 4.0
fix_J = False
fix_J_value = 1 / max_gate_time * 2
train_ideal_weight = 1.0
train_noise_weight = 0
dtype = np.float64

init_learning_rate = 3e-11
actor_network_config = (128, 128)
value_network_config = (128, 128)
init_action_stddev = np.array(3e-6, dtype)

max_iter = 20000
num_collects = 30
num_epochs = 5
batch_size = None

test_ideal_weight = train_ideal_weight
test_noise_weight = train_noise_weight

noise_test_interval = 100
num_noise_test_samples = 50

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
board_update_interval = 100

main_path = '/Users/...'
model_name = 'model_name'
path_name = None # main_path + '/' + model_name


tf.keras.backend.set_floatx('float64')
with tf.device('/cpu'):
    env = TwoQubitEnv(
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
        fix_J=fix_J,
        fix_J_value=fix_J_value,
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
    # old_main_path = '/Users/...'
    # agent = load_network(agent, old_main_path + '/old_model_name', use_stddev_proj_network=False)

    train_agent_2q(
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

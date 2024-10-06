import os
import numpy as np
from pack.driver import EpisodeDriver
from pack.buffer import ReplayBuffer
from pack.plot2q import CNOTLearningDashboard
from pack.test2q import CNOTTester
from pack.utils import save_network, load_network


def train_agent_2q(
        env=None,
        agent=None,
        max_iter=3000,
        num_collects=10,
        num_epochs=10,
        batch_size=None,
        ideal_weight=1.0,
        noise_weight=1.0,
        noise_test_interval=100,
        num_noise_test_samples=100,
        revolution_interval=1,
        adapt_interval=1,
        adaptive_learning_rate_exponent=0.5,
        adaptive_action_stddev_exponent=0.5,
        max_learning_rate=3e-4,
        max_action_stddev=0.3,
        save_interval=100,
        path_name=None,
        board_update_interval=10,
        model_name=None
):
    buffer = ReplayBuffer()
    driver = EpisodeDriver()

    # set the pulse ticks for plotting pulse
    pulse_ticks = np.linspace(0, env.max_gate_time, env.num_steps + 1)
    board = CNOTLearningDashboard(model_name, pulse_ticks)

    # set the range of noise stddevs for noise test
    noise_test_stddevs = np.logspace(-4, 0, 17)
    tester = CNOTTester(env, agent, driver, noise_test_stddevs, num_noise_test_samples, noise_test_interval, ideal_weight, noise_weight)

    # make a new dir to save networks and fig
    if path_name:
        os.makedirs(path_name)
        os.makedirs(path_name + '/fig')
        os.makedirs(path_name + '/network')
        os.makedirs(path_name + '/data')
        #board.save(path_name + '/fig/0')
        save_network(agent, path_name + '/network/0')
        tester.save(path_name + '/data/0')

    # start the training-testing iters
    for iter_idx in range(1, max_iter + 1):
        # clear replay buffer
        buffer.clear()

        # collect data to buffer by driving previous agent in env
        driver.collect(env, agent, buffer, num_collects)

        # train agent with collected data in buffer
        agent.train(buffer, num_epochs, batch_size)

        # test agent and record some (best) vars to lists
        tester.test(env, agent, driver, iter_idx)

        # save agent's networks and board when finding the best or meeting save interval
        if tester.find_best or iter_idx % save_interval == 0:
            board.update(tester)
            if path_name:
                board.save(path_name + '/fig/' + str(iter_idx))
                save_network(agent, path_name + '/network/' + str(iter_idx))
                tester.save(path_name + '/data/' + str(iter_idx))

        # plot the result for every board flush interval
        elif iter_idx % board_update_interval == 0 or iter_idx == 1:
            board.update(tester)

        # revolute agent
        if iter_idx % revolution_interval == 0:
            agent = load_network(agent, path_name + '/network/' + str(tester.best_iter), use_stddev_proj_network=True)

        # adapt learning rate and action stddev
        if adaptive_learning_rate_exponent and iter_idx % adapt_interval == 0:
            # if use the best agent, then use the best inf to update the lr and action stddev
            if iter_idx % revolution_interval == 0:
                scale_learning_rate = tester.weighted_infs[tester.best_iter] ** adaptive_learning_rate_exponent
            else:
                scale_learning_rate = tester.weighted_infs[-1] ** adaptive_learning_rate_exponent
            learning_rate = np.clip(scale_learning_rate, 1e-15, max_learning_rate)
            agent.set_learning_rate(learning_rate)

        if adaptive_action_stddev_exponent and iter_idx % adapt_interval == 0:
            if iter_idx % revolution_interval == 0:
                scale_action_stddev = tester.weighted_infs[tester.best_iter] ** adaptive_action_stddev_exponent
            else:
                scale_action_stddev = tester.weighted_infs[-1] ** adaptive_action_stddev_exponent
            #print(scale_action_stddev)
            action_stddev = np.clip(scale_action_stddev, 1e-15, max_action_stddev)
            agent.set_action_stddev(action_stddev)
    board.show()

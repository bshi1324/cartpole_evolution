#!/usr/bin/env python3
import gym
import time
import numpy as np

from lib import model

MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 10000
NOISE_STD = 0.01
LEARNING_RATE = 0.001


def evaluate(env, net):
    obs = env.reset()
    reward = 0.0
    steps = 0
    done = False
    while not done:
        obs = np.array(obs)
        act_prob = net(obs)
        actions = np.argmax(act_prob, axis=1)
        obs, r, done, _ = env.step(actions[0])
        reward += r
        steps += 1
    return reward, steps


def sample_noise(net):
    pos = []
    neg = []
    for p in net.parameters:
        noise = np.random.normal(size=p.shape).astype(np.float32)
        pos.append(noise)
        neg.append(- noise)
    return pos, neg


def eval_with_noise(env, net, noise):
    old_params = net.parameters
    for p, p_n, in zip(net.parameters, noise):
        p += NOISE_STD * p_n
    r, s = evaluate(env, net)
    net.parameters = old_params
    return r, s


def train(net, batch_noise, batch_reward):
    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    s = np.std(norm_reward)
    if abs(s) > 1e-6:
        norm_reward /= s

    weighted_noise = None
    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n
    for p, p_update in zip(net.parameters, weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        p += LEARNING_RATE * update


def main():
    env = gym.make('CartPole-v0')

    net = model.Net(env.observation_space.shape[0], env.action_space.n)

    step_idx = 0
    while True:
        t_start = time.time()
        batch_noise = []
        batch_reward = []
        batch_steps = 0
        for _ in range(MAX_BATCH_EPISODES):
            noise, neg_noise = sample_noise(net)
            batch_noise.append(noise)
            batch_noise.append(neg_noise)
            reward, steps = eval_with_noise(env, net, noise)
            batch_reward.append(reward)
            batch_steps += steps
            reward, steps = eval_with_noise(env, net, neg_noise)
            batch_reward.append(reward)
            batch_steps += steps
            if batch_steps > MAX_BATCH_STEPS:
                break

        step_idx += 1
        m_reward = np.mean(batch_reward)
        if m_reward > 199:
            print('Solved in {:d} steps'.format(step_idx))
            break

        train(net, batch_noise, batch_reward)

        speed = batch_steps / (time.time() - t_start)
        print("%d: reward=%.2f, speed=%.2f f/s" % (step_idx, m_reward, speed))

        
if __name__ == '__main__':
    main()
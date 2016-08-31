#!/usr/bin/python
#Copy-Past: ftw

import gym
import os

env = gym.make('Copy-v0')
obs = env.reset()

for move in range(1000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample())
    print(obs, reward, done, info)
    raw_input("Press any key to continue")
    os.system('clear')

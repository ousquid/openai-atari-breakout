#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
from gym import spaces
from gym.spaces.box import Box
from wrap import make_env

import cv2

class Environment:
    def __init__(self):
        envs = [make_env(ENV_NAME, i) for i in range(NUM_PROCESSES)]
        self.env = SubprocVecEnv(envs)
        
        self.actor_critic = Net(self.env.action_space.n)
        self.global_brain = Brain(self.actor_critic)
        
        # (4, 84, 84)
        obs_shape = envs.observation_space.shape
        self.obs_shape = (obs_shape[0] * NUM_STACK_FRAME, *obs_shape[1:])
        

    def actions():
        pass
    
    def reward():
        pass
    
    def run(self):
        # (16, 4, 84, 84)
        current_obs = np.zeros([NUM_PROCESSES, *obs_shape])
        episode_rewards = np.zeros([NUM_PROCESSES, 1])
        final_rewards = np.zeros([NUM_PROCESSES, 1])

        # torch.Size([16, 1, 84, 84])
        obs = envs.reset()
        # frameの先頭に最新のobsを格納
        current_obs[:, :1] = obs  
        
        storage = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape, current_obs)

        for j in tqdm(range(NUM_UPDATES)):
            for step in range(NUM_ADVANCED_STEP):
                #with torch.no_grad():
                cpu_actions = self.actor_critic.act(storage.observations[step])[0]
                
                # obs size:(16, 1, 84, 84)
                obs, reward, done, info = self.env.step(cpu_actions)

                reward = reward.reshape(-1,1)
                episode_rewards += reward

                done = done.reshape(-1,1)
                final_rewards[done] = episode_rewards
                episode_rewards[done] = 0

                # 現在の状態をdone時には全部0にする
                current_obs[done.reshape(-1,1,1,1)] = 0

                # frameをstackする
                current_obs[:, 1:] = current_obs[:, :-1] # 2～4番目に1～3番目を上書き
                current_obs[:, :1] = obs  # 1番目に最新のobsを格納

                # メモリオブジェクトに今stepのtransitionを挿入
                storage.insert(current_obs, cpu_action, reward, done)

            # advancedした最終stepの状態から予想する状態価値を計算
            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.observations[-1]).detach()

            # 全stepの割引報酬和を計算して、rolloutsの変数returnsを更新
            rollouts.compute_returns(next_value)

            # ネットワークとrolloutの更新
            global_brain.update(rollouts)
            rollouts.after_update()

            # ログ：途中経過の出力
            if j % 100 == 0:
                print("finished frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".
                      format(j*NUM_PROCESSES*NUM_ADVANCED_STEP,
                             final_rewards.mean(),
                             final_rewards.median(),
                             final_rewards.min(),
                             final_rewards.max()))

            # 結合パラメータの保存
            if j % 12500 == 0:
                torch.save(global_brain.actor_critic.state_dict(),
                           'weight_'+str(j)+'.pth')
        
        # 実行ループの終了
        torch.save(global_brain.actor_critic.state_dict(), 'weight_end.pth')

if __name__=="__main__":
    breakout_env = Environment()
    breakout_env.run()
#!/usr/bin/env python3
# coding: utf-8

import cv2
import numpy as np
from collections import deque
from tqdm import tqdm

import gym
import keras
from wrap import make_env
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from a2c.const import NUM_PROCESSES, NUM_ADVANCED_STEP, NUM_UPDATES, \
                        ENV_NAME, NUM_STACK_FRAME, LR, ALPHA, EPS
from a2c.net import create_a2c_net
from a2c.brain import Brain
from a2c.storage import RolloutStorage
import theano
theano.config.optimizer="fast_compile"
#theano.config.exception_verbosity="high"

class Environment:
    def __init__(self):
        envs = [make_env(ENV_NAME, i) for i in range(NUM_PROCESSES)]
        self.env = SubprocVecEnv(envs)

        # (4, 84, 84)
        obs_shape = self.env.observation_space.shape
        self.obs_shape = (obs_shape[0] * NUM_STACK_FRAME, *obs_shape[1:])
        
        print("obs_shape:", self.obs_shape)
        print("action_num:", self.env.action_space.n)
        self.actor_critic = create_a2c_net(self.obs_shape, self.env.action_space.n)
        
        optm = keras.optimizers.RMSprop(lr=LR, rho=ALPHA, epsilon=EPS)
        self.actor_critic.compile(
            optimizer=optm, loss="mean_squared_error", metrics=["mean_absolute_error"])
        
        self.global_brain = Brain(self.actor_critic)
        
        

    def actions():
        pass
    
    def reward():
        pass
    
    def run(self):
        # (16, 4, 84, 84)
        current_obs = np.zeros([NUM_PROCESSES, *self.obs_shape])
        episode_rewards = np.zeros([NUM_PROCESSES, 1])
        final_rewards = np.zeros([NUM_PROCESSES, 1])

        # torch.Size([16, 1, 84, 84])
        obs = self.env.reset()
        # frameの先頭に最新のobsを格納
        current_obs[:, :1] = obs  
        
        storage = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES, self.obs_shape, current_obs)

        for j in tqdm(range(NUM_UPDATES)):
            for step in range(NUM_ADVANCED_STEP):
                #with torch.no_grad():
                _, cpu_actions = self.actor_critic.predict(storage.observations[step] / 255)
                action = np.argmax(np.array([np.random.multinomial(1, x) for x in cpu_actions]), axis=1)
                
                # obs size:(16, 1, 84, 84)
                obs, reward, done, info = self.env.step(action)

                reward = reward.reshape(-1,1)
                episode_rewards += reward
                
                final_rewards[done] = episode_rewards[done]
                episode_rewards[done] = 0

                # 現在の状態をdone時には全部0にする
                current_obs[done] = 0

                # frameをstackする
                current_obs[:, 1:] = current_obs[:, :-1] # 2～4番目に1～3番目を上書き
                current_obs[:, :1] = obs  # 1番目に最新のobsを格納

                # メモリオブジェクトに今stepのtransitionを挿入
                storage.insert(current_obs, action, reward, done)

            # advancedした最終stepの状態から予想する状態価値を計算
            #with torch.no_grad():
            input_obs = storage.observations[-1] / 255
            next_value, _ = self.actor_critic.predict(input_obs)

            # 全stepの割引報酬和returnsを計算
            storage.compute_discounted_rewards(next_value)

            # ネットワークとstorageの更新
            self.global_brain.update(storage)
            storage.after_update()

            # ログ：途中経過の出力
            if j % 100 == 0:
                print("finished frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".
                      format(j*NUM_PROCESSES*NUM_ADVANCED_STEP,
                             final_rewards.mean(), np.median(final_rewards),
                             final_rewards.min(),final_rewards.max()))

            # 結合パラメータの保存
            if j % 12500 == 0:
                self.actor_critic.save('weight_'+str(j)+'.pth')
        
        # 実行ループの終了
        self.actor_critic.save('weight_end.pth')


if __name__=="__main__":
    breakout_env = Environment()
    breakout_env.run()
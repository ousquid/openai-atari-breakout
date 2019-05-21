import numpy as np
from .const import GAMMA

class RolloutStorage:
    '''Advantage学習するためのメモリクラス'''
    def __init__(self, num_steps, num_processes, obs_shape, current_obs):
        self.num_steps = num_steps
        self.observations = np.zeros([num_steps + 1, num_processes, *obs_shape])
        self.observations[0] = current_obs
        
        self.masks = np.ones([num_steps + 1, num_processes, 1])
        self.rewards = np.zeros([num_steps, num_processes, 1])
        self.actions = np.zeros([num_steps, num_processes, 1]).astype('int64')

        self.discounted_rewards = np.zeros([num_steps + 1, num_processes, 1])
        self.index = 0

    def insert(self, current_obs, action, reward, mask):
        '''次のindexにtransitionを格納する'''
        self.observations[self.index + 1] = current_obs
        self.masks[self.index + 1] = mask.reshape(-1,1)
        self.rewards[self.index] = reward
        self.actions[self.index] = action.reshape(-1,1)

        self.index = (self.index + 1) % self.num_steps

    def after_update(self):
        '''Advantageするstep数が完了したら、最新のものをindex0に格納'''
        self.observations[0] = self.observations[-1]
        self.masks[0] = self.masks[-1]

    def compute_discounted_rewards(self, next_value):
        '''Advantageするステップ中の各ステップの割引報酬和を計算する'''

        # 注意：5step目から逆向きに計算しています
        # 注意：5step目はAdvantage1となる。4ステップ目はAdvantage2となる。・・・
        self.discounted_rewards[-1] = next_value
        for ad_step in reversed(range(self.num_steps)):
            self.discounted_rewards[ad_step] = \
                self.discounted_rewards[ad_step + 1] * GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]

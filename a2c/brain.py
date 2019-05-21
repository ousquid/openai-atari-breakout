import os
import keras
import keras.backend as K
from keras.optimizers import RMSprop

from .const import LR, EPS, ALPHA, VALUE_LOSS_COEF, ENTROPY_COEF, MAX_GRAD_NORM, \
    NUM_ADVANCED_STEP, NUM_PROCESSES

class Brain:
    def __init__(self, actor_critic, filename=""):
        self.actor_critic = actor_critic  

        if filename and os.path.exists(filename):
            self.actor_critic.load(filename)


    def evaluate_actions(self, x, actions):
        '''
        状態xから状態価値、実際の行動actionsのlog確率とエントロピーを求めます
        # https://github.com/germain-hug/Deep-RL-Keras/blob/master/A2C/actor.py#L29
        
        '''
        value, actor_output = self.actor_critic.output
        log_probs = K.log(actor_output)  
        
        # a[i][j] = log_probs[i][actions[i][j]]
        action_log_probs = K.gather(K.transpose(log_probs), actions)
        
        # dim=1で行動の種類方向に計算
        dist_entropy = -K.mean(K.sum(log_probs * actor_output, axis=1)) 
        # 軸やばい # meanやばい-> size 1?
    
        return value, action_log_probs, dist_entropy
        
        
    def update(self, storage):
        '''advanced計算した5つのstepの全てを使って更新します'''
        
        def get_updates(action_pl, discounted_r):
            # torch.Size([4, 84, 84])
            values, action_log_probs, dist_entropy = \
                self.evaluate_actions(
                    self.actor_critic.input,
                    action_pl
                )
     
            # storage.observations[:-1].view(-1, *obs_shape) torch.Size([80, 4, 84, 84])
            # storage.actions.view(-1, 1) torch.Size([80, 1])
            # values torch.Size([80, 1])
            # action_log_probs torch.Size([80, 1])
            # dist_entropy torch.Size([])
    
            # torch.Size([5, 16, 1])
            values = K.reshape(values, [num_steps, num_processes, 1] ) 
            action_log_probs = K.reshape(action_log_probs, [num_steps, num_processes, 1] )
            
            # torch.Size([5, 16, 1])
            advantages = discounted_r - values
            value_loss = K.mean(K.pow(advantages, 2) )
            action_gain = K.mean(K.stop_gradient(advantages) * action_log_probs)

            total_loss = (value_loss * VALUE_LOSS_COEF -
                          action_gain - dist_entropy * ENTROPY_COEF)
                          
            return RMSprop(lr=LR, epsilon=0.1, rho=0.99, clipnorm=MAX_GRAD_NORM).get_updates(
                (self.actor_critic.trainable_weights, [], total_loss) )

        obs_shape = storage.observations.size()[2:]  
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES
    
        self.action_pl = K.placeholder(shape=(None, self.actor_critic.output_shape[1]))
        self.discounted_r = K.placeholder(shape=(None,))

        K_func = K.function(
            [self.actor_critic.input, self.action_pl, self.discounted_r],
            [], 
            updates=get_updates(self.action_pl, self.discounted_r))

        K_func([storage.observations[:-1].reshape(-1, *obs_shape),
                storage.actions.reshape(-1, 1),
                storage.discounted_rewards[:-1]])

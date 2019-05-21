from keras.layers import Input, Dense, Conv2d
from keras.models import Model

def Net(num_process, obs_shape, n_out):
    # This returns a tensor
    inputs = Input(shape=[num_process, *obs_shape])
    
    x = Conv2d(32, 8, strides=4, activation='relu')(inputs/255.0)
    x = Conv2d(64, 4, strides=2, activation='relu')(x)
    x = Conv2d(64, 3, strides=1, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)

    cri = Dense(1)(x)
    act = Dense(n_out, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=[cri, act])
    return model


def act(self, x):
    '''状態xから行動を確率的に求めます'''
    value, actor_output = self(x)  
    
    # dim=1で行動の種類方向に計算
    probs = F.softmax(actor_output, dim=1)    
    action = probs.multinomial(num_samples=1)

    return action

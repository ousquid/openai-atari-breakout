from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model

def create_a2c_net(obs_shape, n_out):
    # This returns a tensor (None, 1, 84, 84)
    inputs = Input(shape=obs_shape)
    
    x = Conv2D(32, 8, strides=4, activation='relu')(inputs)
    x = Conv2D(64, 4, strides=2, activation='relu')(x)
    x = Conv2D(64, 3, strides=1, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)

    cri = Dense(1)(x)
    act = Dense(n_out, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=[cri, act])
    return model
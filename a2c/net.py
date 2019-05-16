
def init(module, gain):
    '''層の結合パラメータを初期化する関数を定義'''
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module

class Flatten(nn.Module):
    '''コンボリューション層の出力画像を1次元に変換する層を定義'''

    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self, n_out):
        super(Net, self).__init__()

        # 結合パラメータの初期化関数
        def init_(module): return init(
            module, gain=nn.init.calculate_gain('relu'))

        # コンボリューション層の定義
        self.conv = nn.Sequential(
            # 画像サイズの変化84*84→20*20
            init_(nn.Conv2d(NUM_STACK_FRAME, 32, kernel_size=8, stride=4)),
            # stackするflameは4画像なのでinput=NUM_STACK_FRAME=4である、出力は32とする、
            # sizeの計算  size = (Input_size - Kernel_size + 2*Padding_size)/ Stride_size + 1

            nn.ReLU(),
            # 画像サイズの変化20*20→9*9
            init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, kernel_size=3, stride=1)),  # 画像サイズの変化9*9→7*7
            nn.ReLU(),
            Flatten(),  # 画像形式を1次元に変換
            init_(nn.Linear(64 * 7 * 7, 512)),  # 64枚の7×7の画像を、512次元のoutputへ
            nn.ReLU()
        )

        # 結合パラメータの初期化関数
        def init_(module): 
            return init(module, gain=1.0)

        # Criticの定義
        self.critic = init_(nn.Linear(512, 1))  # 状態価値なので出力は1つ

        # 結合パラメータの初期化関数
        def init_(module): 
            return init(module, gain=0.01)

        # Actorの定義
        self.actor = init_(nn.Linear(512, n_out))  # 行動を決めるので出力は行動の種類数

        # ネットワークを訓練モードに設定
        self.train()


    def forward(self, x):
        '''ネットワークのフォワード計算を定義します'''
        input = x / 255.0  # 画像のピクセル値0-255を0-1に正規化する
        conv_output = self.conv(input)  # Convolution層の計算
        critic_output = self.critic(conv_output)  # 状態価値の計算
        actor_output = self.actor(conv_output)  # 行動の計算

        return critic_output, actor_output


    def act(self, x):
        '''状態xから行動を確率的に求めます'''
        value, actor_output = self(x)  
        # 190324
        # self(x)の動作について、以下のリンクの最下部のFAQに解説を補足しました。
        # https://github.com/YutaroOgawa/Deep-Reinforcement-Learning-Book
        
        probs = F.softmax(actor_output, dim=1)    # dim=1で行動の種類方向に計算
        action = probs.multinomial(num_samples=1)

        return action


    def get_value(self, x):
        '''状態xから状態価値を求めます'''
        value, actor_output = self(x)
        # 190324
        # self(x)の動作について、以下のリンクの最下部のFAQに解説を補足しました。
        # https://github.com/YutaroOgawa/Deep-Reinforcement-Learning-Book
        
        return value


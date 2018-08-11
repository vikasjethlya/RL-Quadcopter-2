from keras import layers, models, optimizers
from keras import backend as K

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        # Input layer
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Hidden layers (state)
        L2 = 0.1
        net_states = layers.Dense(units=512, kernel_regularizer=layers.regularizers.l2(L2))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation("relu")(net_states)
        net_states = layers.Dense(units=256, kernel_regularizer=layers.regularizers.l2(L2))(net_states)

        # Hidden layers (action)
        net_actions = layers.Dense(units=256,kernel_regularizer=layers.regularizers.l2(L2))(actions)

        # Hidden layers (both)
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Output layer
        out = layers.Dense(units=1,
                           name='q_values',
                           kernel_initializer=layers.initializers.RandomUniform(minval=-0.3, maxval=0.3))(net)

        # Model
        self.model = models.Model(inputs=[states, actions], outputs=out)

        # Compile
        self.model.compile(optimizer=optimizers.Adam(lr=L2), loss='mse')

        # Action gradients
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()],
        outputs=K.gradients(out, actions))
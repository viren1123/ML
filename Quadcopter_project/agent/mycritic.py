from keras import layers, models, optimizers
from keras import backend as K

class MyCritic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):
        
        states = layers.Input(shape=(self.state_size,))
        actions = layers.Input(shape=(self.action_size,))

        states_layers = layers.Dense(32, activation='relu')(states)
        states_layers = layers.Dense(64, activation='relu')(states_layers)

        actions_layers = layers.Dense(32, activation='relu')(actions)
        actions_layers = layers.Dense(64, activation='relu')(actions_layers)

        combined_layers = layers.Add()([states_layers, actions_layers])
        combined_layers = layers.Activation('relu')(combined_layers)

        Q_values = layers.Dense(1)(combined_layers)

        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        action_gradients = K.gradients(Q_values, actions)

        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
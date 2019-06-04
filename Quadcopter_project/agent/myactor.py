from keras import layers, models, optimizers
from keras import backend as K

class MyActor:
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.build_model()

    def build_model(self):
       
        states = layers.Input(shape=(self.state_size,))
        net = layers.Dense(32, activation='relu')(states)
        net = layers.Dense(64, activation='relu')(net)
        net = layers.Dense(128, activation='relu')(net)

        actions_prob = layers.Dense(self.action_size, activation='sigmoid')(net)
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low)(actions_prob)

        self.model = models.Model(inputs=states, outputs=actions)

        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
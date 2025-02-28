# Adopted from https://github.com/sachinruk/blog
# License information https://github.com/sachinruk/blog/blob/6cff65295f617da6c33995128c47304f22e83026/LICENSE
# Taken from https://github.com/sachinruk/blog/blob/6cff65295f617da6c33995128c47304f22e83026/_notebooks/2021-02-15-Tensorflow-Learning-Rate-Finder.ipynb
import tensorflow as tf


class LRFind(tf.keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, n_rounds):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_up = (max_lr / min_lr) ** (1 / n_rounds)
        self.lrs = []
        self.losses = []

    def on_train_begin(self, logs=None):
        self.weights = self.model.get_weights()
        self.model.optimizer.lr = self.min_lr

    def on_train_batch_end(self, batch, logs=None):
        self.lrs.append(self.model.optimizer.lr.numpy())
        self.losses.append(logs["loss"])
        self.model.optimizer.lr = self.model.optimizer.lr * self.step_up
        if self.model.optimizer.lr > self.max_lr:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.model.set_weights(self.weights)
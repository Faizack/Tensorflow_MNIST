import tensorflow as tf
from tensorflow import keras
import numpy as np
import copy
import collections
import torch
from collections import OrderedDict
# import tensorflow_federated as tff


class Model():
    '''
    Contains all machine learning functionality
    '''
    # static, might have to be calculated dynamically
    batch_size = 64
    epochs = 3

    def __init__(self, num_workers, idx, model, optimizer, device, topk, isEvil=False):
        self.num_workers = num_workers
        self.idx = idx
        self.model = model
        self.optimizer = optimizer
        self.DEVICE = device
        self.topk = topk
        self.isEvil = isEvil

        # this would be generic in a real application
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
        x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
        self.train_loader = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train))
        self.train_loader = self.train_loader.shuffle(
            len(x_train)).batch(self.batch_size)

        self.test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        self.test_loader = self.test_loader.shuffle(
            len(x_test)).batch(self.batch_size)

        self.garbage = tf.random.uniform((64, 28, 28, 1))

        # find the datasets indices
        # also this would not be implemented like this in the real application
        # the users would use an 80/20 random split of their own dataset for training/validating
        self.num_train_batches = len(self.train_loader) // self.num_workers
        self.num_test_batches = len(self.test_loader) // self.num_workers
        # start idx
        self.start_idx_train = self.num_test_batches * self.idx
        self.start_idx_test = self.num_test_batches * self.idx


  

    def average(self, state_dicts):

        print("Averaging")

        weights = [model.get_weights() for model in state_dicts]
        avg_weights = [np.mean(w, axis=0) for w in zip(*weights)]

        # create a new model with the same architecture as the first model
        super_model = tf.keras.models.clone_model(state_dicts[0])

    # set the averaged weights in the new model
        super_model.set_weights(avg_weights)
        # print("super model",super_model)
        return super_model
        # return state_dicts[0]


    

    def adapt_current_model(self, avg_state_dict):
        self.model=avg_state_dict

    def train(self):
        # self.model.train()
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        for epoch in range(self.epochs):
            for idx, (data, target) in enumerate(self.train_loader):
                if idx >= self.start_idx_train and idx < self.start_idx_train + self.num_train_batches:
                    if self.isEvil:
                        data = self.garbage
                        target = tf.random.uniform((self.batch_size,), maxval=10, dtype=tf.int32)
                   
                    self.model.train_on_batch(data, target)
            print(f'Finished epoch {epoch} of worker {self.idx}')
        return self.model

    def rank_models(self, sorted_models):
        return [self.num_workers - idx for idx in range(len(sorted_models))]

    def get_top_k(self, sorted_models):
        return [models[2] for models in sorted_models][-self.topk:]

    def test(self):
        test_loss = 0
        correct = 0
        for idx, (data, target) in enumerate(self.test_loader):
            if idx >= self.start_idx_test and idx < self.start_idx_test + self.num_test_batches:
                output = self.model(data)
                if target.shape[-1] == 1:
                    target = tf.squeeze(target, axis=-1)
                test_loss += tf.keras.losses.sparse_categorical_crossentropy(
                    target, output).numpy().mean()
                pred = tf.argmax(output, axis=1)
                correct += tf.reduce_sum(tf.cast(tf.equal(pred,
                                         tf.cast(target, tf.int64)), tf.float32))

        test_loss /= (self.num_test_batches * self.batch_size)
        print('\nTest set: , Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, self.num_test_batches * self.batch_size,
            100. * correct / (self.num_test_batches * self.batch_size)))

        return correct / (self.num_test_batches*self.batch_size)

    def eval(self, model_state_dicts):
        res = []
        for idx, m in enumerate(model_state_dicts):
            # self.model.set_weights(m.get_weights())
            acc = self.test()
            res.append((acc, idx, m))
            # print("Res", res)

        sorted_models = sorted(res, key=lambda t: t[0])
        return self.rank_models(sorted_models), self.get_top_k(sorted_models), res

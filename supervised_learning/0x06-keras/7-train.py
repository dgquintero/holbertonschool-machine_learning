#!/usr/bin/env python3
"""function optimize_model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    Function train the model using early stopping
        Arguments:
            network: model to train
            data: contain the input data
            labels: containing the labels of data
            batch_size: size of the batch used for mini-batch gd
            epochs: number of passes through data for mini-batch gd
            verbose: boolean if output should be printed during training
            shuffle: boolean that determines whether to shuffle the batches
                every epoch
            early_stopping: is a boolean that indicates whether early stopping
                should be use
            patience: is the patience used for early stopping
                every epoch
            learning_rate_decay: boolean that indicates whether learning
                rate decay should be used
            alpha: initial lr
            decay_rate: decay rate
        Returns: History object
    """

    def scheduler(epoch):
        """Function to get the lr of each epoch"""
        return alpha / (1 + decay_rate * epoch)

    callback = []
    ES = K.callbacks.EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   patience=patience)
    LRD = K.callbacks.LearningRateScheduler(scheduler, verbose=1)

    if validation_data and early_stopping:
        callback.append(ES)

    if validation_data and learning_rate_decay:
        callback.append(LRD)

    history = network.fit(x=data, y=labels, epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callback,
                          batch_size=batch_size, verbose=verbose,
                          shuffle=shuffle)
    return history

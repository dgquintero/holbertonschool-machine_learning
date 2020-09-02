#!/usr/bin/env python3
"""function optimize_model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
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
        Returns: History object
    """
    callback_es = []
    ES = K.callbacks.EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   patience=patience)
    if validation_data and early_stopping:
        callback_es.append(ES)

    history = network.fit(x=data, y=labels, epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callback_es,
                          batch_size=batch_size, verbose=verbose,
                          shuffle=shuffle)
    return history

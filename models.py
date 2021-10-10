import tensorflow as tf
import numpy as np


from custom_layers import *
import hyperparameters




def create_model(network_type, n_heads, hidden_units, alpha_sparse, use_bias_sparse=True):
    tf.keras.backend.clear_session()
    
    hidden_initializers = tf.keras.initializers.RandomUniform(minval=-np.sqrt(6/hidden_units)/hyperparameters.OMEGA_0,
    														  maxval=np.sqrt(6/hidden_units)/hyperparameters.OMEGA_0)

    
    if network_type == "fourier":
        X = tf.keras.layers.Input(shape=(hyperparameters.INPUT_COORDINATES,))
        x = FourierLayer(hyperparameters.FOURIER_UNITS, hyperparameters.SCALE)(X)
        x = tf.keras.layers.Dense(hidden_units, activation='relu')(x)

       	for counter in range(hyperparameters.BASE_N_HLAYERS - 1):
            x = tf.keras.layers.Dense(hidden_units, activation='relu')(x)

        head_output = tf.keras.layers.Dense(1, kernel_initializer=hidden_initializers)(x)
    elif network_type == "siren":
        X = tf.keras.layers.Input(shape=(hyperparameters.INPUT_COORDINATES,))
        x = SineLayer(hyperparameters.INPUT_COORDINATES, hidden_units, is_first=True, omega_0=hyperparameters.OMEGA_0)(X)

        for counter in range(hyperparameters.BASE_N_HLAYERS - 1):
            x = SineLayer(hidden_units, hidden_units, is_first=False)(x)

        head_output = tf.keras.layers.Dense(1, kernel_initializer=hidden_initializers)(x)
    elif network_type == "multi-head":
        X = tf.keras.layers.Input(shape=(hyperparameters.INPUT_COORDINATES,))
        x = tf.keras.layers.Dense(hidden_units, activation='relu')(X)

        for counter in range(hyperparameters.BASE_N_HLAYERS - 1):
            x = tf.keras.layers.Dense(hidden_units, activation='relu')(x)

        head_output = PureSparseLayer(units=n_heads,
                                      alpha=alpha_sparse,
                                      use_bias=use_bias_sparse)(x)
    else:
        raise NameError('network_type must be "siren", "multihead" or "fourier"')

        

    model = tf.keras.models.Model(X, head_output)

    
    return model
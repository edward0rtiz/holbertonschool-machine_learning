#!/usr/bin/env python3
""" Variational Autoencoder
"""

import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    function that creates a variational autoencoder
    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers:  list containing the number of nodes for each hidden
                        layer in the encoder, respectively
        latent_dims: integer containing the dimensions of the latent space
                     representation
    Returns: encoder, decoder, auto
    """

    X_input = K.Input(shape=(input_dims,))
    hidden_ly = K.layers.Dense(units=hidden_layers[0], activation='relu')
    Y_prev = hidden_ly(X_input)
    for i in range(1, len(hidden_layers)):
        hidden_ly = K.layers.Dense(units=hidden_layers[i],
                                   activation='relu')
        Y_prev = hidden_ly(Y_prev)
    latent_ly = K.layers.Dense(units=latent_dims, activation=None)
    z_mean = latent_ly(Y_prev)
    z_log_sigma = latent_ly(Y_prev)

    def sampling(args):
        """Sampling similar points in latent space"""
        z_mean, z_stand_des = args
        batch = K.backend.shape(z_mean)[0]
        dim = K.backend.int_shape(z_mean)[1]
        epsilon = K.backend.random_normal(shape=(batch, dim))
        return z_mean + K.backend.exp(z_stand_des) * epsilon

    z = K.layers.Lambda(sampling, output_shape=(latent_dims,))([z_mean,
                                                                z_log_sigma])
    encoder = K.Model(X_input, [z, z_mean, z_log_sigma])

    X_decode = K.Input(shape=(latent_dims,))
    hidden_ly_deco = K.layers.Dense(units=hidden_layers[-1], activation='relu')
    Y_prev = hidden_ly_deco(X_decode)
    for j in range(len(hidden_layers) - 2, -1, -1):
        hidden_ly_deco = K.layers.Dense(units=hidden_layers[j],
                                        activation='relu')
        Y_prev = hidden_ly_deco(Y_prev)
    last_ly = K.layers.Dense(units=input_dims, activation='sigmoid')
    output = last_ly(Y_prev)
    decoder = K.Model(X_decode, output)

    e_output = encoder(X_input)[-1]
    d_output = decoder(e_output)
    auto = K.Model(X_input, d_output)

    def vae_loss(x, x_decoder_mean):
        x_loss = K.backend.binary_crossentropy(x, x_decoder_mean)
        kl_loss = - 0.5 * K.backend.mean(1 + z_log_sigma -
                                         K.backend.square(z_mean) -
                                         K.backend.exp(z_log_sigma), axis=-1)
        return x_loss + kl_loss

    auto.compile(loss=vae_loss, optimizer='adam')
    return encoder, decoder, auto

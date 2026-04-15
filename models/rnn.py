import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

#Modele RNN naive
def build_rnn_v1(input_shape):
    inputs  = keras.layers.Input(shape=input_shape)
    x = layers.LSTM(24)(inputs)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="RNN_Baseline")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

#Modele avec regulation:dropout
def build_rnn_v2(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(units=24)(inputs)
    x = layers.Dropout(0.5)(x) 
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="RNN_V2_Dropout")
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

#modele final avec recall et Early Stopping
def build_rnn_final(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(24, name="LSTM_Layer")(inputs)
    x = layers.Dropout(0.5, name="Dropout_Layer")(x)
    outputs = layers.Dense(1, activation='sigmoid', name="Output_Layer")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="RNN_V3_Final")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    return model


# Backward-compatible API used by legacy scripts
def build_rnn(input_shape):
    return build_rnn_final(input_shape)
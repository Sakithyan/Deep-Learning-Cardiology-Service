import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

#exemples: baseline
def build_rnn_v1(input_shape):
    #test: premiere version pour comparaison
    inputs  = keras.layers.Input(shape=input_shape)
    #exemples: LSTM pour sequence ECG
    x = layers.LSTM(24)(inputs)
    #api: sortie binaire
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="RNN_Baseline")
    #api: compile standard binaire
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

#fixe: ajoute dropout
def build_rnn_v2(input_shape):
    #fixe: meme idee que v1, mais plus robuste
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(units=24)(inputs)
    #fixe: regularisation
    x = layers.Dropout(0.5)(x) 
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="RNN_V2_Dropout")  
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

#api: version finale
def build_rnn_final(input_shape):
    #api: version utilisee par build_rnn
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(24, name="LSTM_Layer")(inputs)
    x = layers.Dropout(0.5, name="Dropout_Layer")(x)
    outputs = layers.Dense(1, activation='sigmoid', name="Output_Layer")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="RNN_V3_Final")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        #api: suivi precision + rappel
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    return model


#api: compat legacy
def build_rnn(input_shape):
    return build_rnn_final(input_shape)
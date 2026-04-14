import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Modele MLP naive
def build_mlp_v1(input_shape):
    inputs = keras.layers.Input(shape=input_shape)
    x = layers.Dense(24, activation='relu')(inputs)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP_Baseline")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Modele MLP avec regulation: dropout
def build_mlp_v2(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(24, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x) 
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP_V2_Dropout")
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Modele MLP final avec recall et Early Stopping
def build_mlp_final(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(24, activation='relu', name="Hidden_Layer")(inputs)
    x = layers.Dropout(0.5, name="Dropout_Layer")(x)
    outputs = layers.Dense(1, activation='sigmoid', name="Output_Layer")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP_V3_Final")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    return model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# MLP baseline (naive)
def build_mlp_v1(input_shape):
    inputs = keras.layers.Input(shape=input_shape)
    x = layers.Dense(24, activation="relu")(inputs)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP_Baseline")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# MLP regularized with dropout
def build_mlp_v2(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(24, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP_V2_Dropout")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Final MLP version with recall metric
def build_mlp_final(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(24, activation="relu", name="Hidden_Layer")(inputs)
    x = layers.Dropout(0.5, name="Dropout_Layer")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="Output_Layer")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP_V3_Final")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall")],
    )
    return model


# Backward-compatible API used by legacy scripts
def build_mlp(input_shape):
    return build_mlp_final(input_shape)

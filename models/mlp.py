import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#exemples: baseline dense
def build_mlp_v1(input_shape):
    #test: version la plus simple pour reference
    inputs = keras.layers.Input(shape=input_shape)
    #exemples: 1 couche cachee
    x = layers.Dense(24, activation="relu")(inputs)
    #api: sortie binaire
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP_Baseline")
    #api: compile standard binaire
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


#fixe: ajoute dropout
def build_mlp_v2(input_shape):
    #fixe: meme base que v1 avec regularisation
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(24, activation="relu")(inputs)
    #fixe: regularisation
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP_V2_Dropout")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


#api: version finale
def build_mlp_final(input_shape):
    #api: garde la structure simple + metric recall
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(24, activation="relu", name="Hidden_Layer")(inputs)
    x = layers.Dropout(0.5, name="Dropout_Layer")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="Output_Layer")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP_V3_Final")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        #api: suivi precision + rappel
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall")],
    )
    return model


#api: compat legacy
def build_mlp(input_shape):
    return build_mlp_final(input_shape)

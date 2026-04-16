import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_naive(input_shape):
    #exemples: baseline Conv1D simple
    #test: point de depart pour comparer les autres versions
    inputs = layers.Input(shape=input_shape)

    #exemples: extraction locale puis reduction
    x = layers.Conv1D(filters=16, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    #exemples: on passe en vecteur avant la dense
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    #api: classification binaire
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="CNN_NAIVE")
    #api: compile standard binaire
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_regularized(input_shape):
    #fixe: version regularisee
    #fixe: utile quand la baseline sur-apprend
    inputs = layers.Input(shape=input_shape)
    #fix: L2 + BN pour stabiliser
    x = layers.Conv1D(
        filters=16,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    #fixe: 2e bloc conv pour enrichir les features
    x = layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    #fixe: dropout contre overfitting
    x = layers.Flatten()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.25)(x)
    #api: meme tete de sortie pour comparer proprement
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="CNN_REGULARIZED")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(input_shape):
    #api: entree standard du projet
    return build_cnn_regularized(input_shape)

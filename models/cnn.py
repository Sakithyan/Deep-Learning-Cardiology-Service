import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_naive(input_shape):
    # Version simple (baseline)
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(filters=16, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="CNN_NAIVE")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_regularized(input_shape):
    # Version avec regulation pour limiter l'overfitting
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(
        filters=16,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="CNN_REGULARIZED")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(input_shape):
    # Par defaut on garde la version regularized
    return build_cnn_regularized(input_shape)

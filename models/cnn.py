import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)

    # MODIFIER/SUPPRIMER ICI : Ajoutez vos couches cachées selon votre architecture CNN
    #RAPPEL : LE PROF VEUX DU FONCTIONNEL PAS DU SÉQUENTIEL
    x = layers.Flatten()(inputs) # Exemple minimal
    # ------------------------------

    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs, name="CNN")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
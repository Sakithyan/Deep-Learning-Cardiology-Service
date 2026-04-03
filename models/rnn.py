import tensorflow as tf
from tensorflow.keras import layers, models

def build_rnn(input_shape):
    inputs = layers.Input(shape=input_shape)

    # MODIFIER/SUPPRIMER ICI : Ajoutez vos couches cachées selon votre architecture RNN
    #RAPPEL : LE PROF VEUX DU FONCTIONNEL PAS DU SÉQUENTIEL
    x = layers.SimpleRNN(1)(inputs) # Exemple minimal
    # ------------------------------

    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs, name="RNN")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
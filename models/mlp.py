import tensorflow as tf
from tensorflow.keras import layers, models

def build_mlp(input_shape):
    inputs = layers.Input(shape=input_shape)

    # MODIFIER/SUPPRIMER ICI : Ajoutez vos couches cachées selon votre architecture MLP
    #RAPPEL : LE PROF VEUX DU FONCTIONNEL PAS DU SÉQUENTIEL
    
    x = layers.Dense(1, activation='relu')(inputs) # Exemple minimal
    # ------------------------------

    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs, name="MLP")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
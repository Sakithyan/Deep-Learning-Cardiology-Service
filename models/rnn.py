import tensorflow as tf
from tensorflow.keras import layers, models

def build_rnn(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(24, return_sequences=False, name="LSTM_Layer")(inputs)
    x = layers.Dropout(0.5, name="Dropout_Layer")(x)
    outputs = layers.Dense(1, activation='sigmoid', name="Output_Layer")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="RNN_LSTM_Cardio")

    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    return model


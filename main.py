import matplotlib.pyplot as plt
import numpy as np
from core.data_loader import load_ecg_dataset
from models.mlp import build_mlp
from models.cnn import build_cnn
from models.rnn import build_rnn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
from sklearn.decomposition import PCA
import time

def run_experiment(model_type="mlp"):
    # 1. Configuration des hyperparamètres selon le modèle
    # On suit les consignes du TP LSTM pour le RNN
    if model_type == "rnn":
        epochs = 2000
        batch_size = 256
        is_3d = True
    else:
        epochs = 100 
        batch_size = 16
        is_3d = (model_type == "cnn")

    # 2. Chargement des données
    x_train, x_test, y_train, y_test = load_ecg_dataset(is_3d=is_3d)
    
    # 3. Construction du modèle
    input_shape = x_train.shape[1:]
    if model_type == "mlp":
        model = build_mlp(input_shape)
    elif model_type == "cnn":
        model = build_cnn(input_shape)
    elif model_type == "rnn":
        model = build_rnn(input_shape)

    # 4. Entraînement avec mesure du temps 
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_split=0.2, verbose=1)
    train_time = time.time() - start_time

    # 5. Affichage des courbes Accuracy/Loss 
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(f'Accuracy - {model_type}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f'Loss - {model_type}')
    plt.legend()
    plt.show()

    # 6. Évaluation et Étude de complexité
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n--- RAPPORT {model_type.upper()} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Temps d'entraînement: {train_time:.2f}s")
    print(f"Nombre de paramètres: {model.count_params()}")

    # 7. Analyses spécifiques au RNN/LSTM
    if model_type == "rnn":
        # Matrice de Confusion 
        y_pred = (model.predict(x_test) > 0.5).astype("int32")
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot()
        plt.title("Matrice de Confusion RNN")
        plt.show()

        # Mesure RMSE (Demandé dans le cours pour les séries temporelles)
        # RMSE = racine de la moyenne des erreurs au carré
        test_predict = model.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, test_predict))
        print(f"RMSE (Performance prédiction): {rmse:.4f}")

        # Analyse PCA 
        # On utilise le nom de la couche défini dans rnn.py
        try:
            from tensorflow.keras.models import Model
            feature_extractor = Model(inputs=model.input, 
                                      outputs=model.get_layer("LSTM_Layer").output)
            features = feature_extractor.predict(x_train)
            pca = PCA(n_components=2)
            components = pca.fit_transform(features)
            plt.scatter(components[:, 0], components[:, 1], c=y_train, cmap='coolwarm')
            plt.title("PCA des caractéristiques LSTM")
            plt.show()
        except:
            print("Note: Nom de couche LSTM_Layer non trouvé pour la PCA.")

if __name__ == "__main__":
    # Il suffit de changer le nom ici pour comparer
    run_experiment("rnn")
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from core.data_loader import load_ecg_dataset
from models.rnn import build_rnn_final as build_rnn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
# Imports des différents modèles
from models.rnn import build_rnn_final as build_rnn
# from models.cnn import build_cnn_final as build_cnn  # À décommenter quand tu les auras
# from models.mlp import build_mlp_final as build_mlp

def run_simple_experiment(model_type="RNN"):
    print(f"\n>>> DEMARRAGE DE L'EXPERIENCE : {model_type} <<<")
    
    # ADAPTATION DES DONNÉES
    # Le RNN et CNN ont besoin de 3D (batch, temps, 1), le MLP de 2D (batch, temps)
    is_3d = True if model_type in ["RNN", "CNN"] else False
    x_train, x_test, y_train, y_test = load_ecg_dataset(is_3d=is_3d)

    #CONSTRUCTION DU MODÈLE
    if model_type == "RNN":
        model = build_rnn(input_shape=x_train.shape[1:])
        layer_for_pca = "LSTM_Layer" # Nom de la couche pour la PCA
    elif model_type == "CNN":
        # model = build_cnn(input_shape=x_train.shape[1:])
        layer_for_pca = "Conv_Final"
        pass 
    elif model_type == "MLP":
        # model = build_mlp(input_shape=x_train.shape[1:])
        layer_for_pca = "Dense_Hidden"
        pass

    model.summary()

    # ENTRAÎNEMENT 
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True 
    )

    print(">>> Debut de l'entrainement...")
    start_train = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=500,
        batch_size=32,
        validation_split=0.2, 
        callbacks=[early_stop],
        verbose=1
    )
    training_duration = time.time() - start_train

    # SAUVEGARDE ET COMPLEXITÉ 
    # Sauvegarde pour la mesure de taille (contrainte vieux PC)
    model_path = 'best_rnn_model.h5'
    model.save(model_path)
    model_size_kb = os.path.getsize(model_path) / 1024

    # Mesure du temps d'inférence (vitesse de diagnostic)
    start_inf = time.time()
    y_probs = model.predict(x_test, verbose=0)
    inference_time_ms = ((time.time() - start_inf) / len(x_test)) * 1000

    # ÉVALUATION FINALE 
    print("\n" + "="*40)
    print("RAPPORT TECHNIQUE POUR LA CLINIQUE")
    loss, accuracy, recall = model.evaluate(x_test, y_test, verbose=0)
    print(f"Precision (Accuracy) : {accuracy*100:.2f}%")
    print(f"Securité (Recall)    : {recall*100:.2f}%")
    print(f"Nombre de parametres : {model.count_params()}")
    print(f"Taille du modele    : {model_size_kb:.2f} Ko")
    print(f"Temps d'inference   : {inference_time_ms:.4f} ms/ECG")
    print("="*40)

    # VISUALISATIONS
    
    # Graphique de la perte (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Evolution de la Perte (Loss)')
    plt.legend()

    # Graphique de la précision (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Evolution de la Précision (Accuracy)')
    plt.legend()

    plt.show()

    # B. Matrice de Confusion (Interprétation Clinique)
    # Seuil à 0.5 car sortie Sigmoid
    y_pred = (y_probs > 0.5).astype(int)
    ax = plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Infarctus', 'Normal'])
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='.2f')
    plt.title("Matrice de Confusion")

    # C. PCA (Visualisation de l'Espace Latent)
    plt.subplot(1, 2, 2)
    try:
        # On extrait les caractéristiques de la couche LSTM 
        feat_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer("LSTM_Layer").output)
        features = feat_extractor.predict(x_train, verbose=0)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', alpha=0.7)
        plt.title("Espace Latent (PCA)")
        plt.colorbar(label="0: Infarctus, 1: Normal")
    except Exception as e:
        print(f"\nNote: PCA échouée (vérifiez le nom de la couche LSTM). Erreur: {e}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simple_experiment()
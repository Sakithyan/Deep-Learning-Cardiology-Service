import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from core.data_loader import load_ecg_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# Imports des différents modèles
from models.rnn import build_rnn_final as build_rnn
# from models.cnn import build_cnn_final as build_cnn  # À décommenter 
# from models.mlp import build_mlp_final as build_mlp  # À décommenter

def run_simple_experiment(model_type="RNN", n_runs=5):
    print(f"\n>>> DEMARRAGE DE L'EXPERIENCE : {model_type} ({n_runs} lancements) <<<")
    
    # ADAPTATION DES DONNÉES
    is_3d = True if model_type in ["RNN", "CNN"] else False
    x_train, x_test, y_train, y_test = load_ecg_dataset(is_3d=is_3d)

    # Listes pour stocker les résultats de robustesse
    all_acc = [] #accuracy
    all_rec = [] #recall
    all_inf = [] #inference

    # BOUCLE DE ROBUSTESSE
    for i in range(n_runs):
        print(f"\n--- Lancement {i+1}/{n_runs} ---")
        
        # CONSTRUCTION DU MODÈLE (doit être dans la boucle pour repartir de zéro)
        if model_type == "RNN":
            model = build_rnn(input_shape=x_train.shape[1:])
            layer_for_pca = "LSTM_Layer"
        elif model_type == "CNN":
            # model = build_cnn(input_shape=x_train.shape[1:])
            layer_for_pca = "Conv_Final"
            pass 
        elif model_type == "MLP":
            # model = build_mlp(input_shape=x_train.shape[1:])
            layer_for_pca = "Dense_Hidden"
            pass

        if i == 0: model.summary() # On affiche le résumé seulement au premier tour


        # ENTRAÎNEMENT 
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True 
        )
        history = model.fit(
            x_train, y_train,
            epochs=500, batch_size=32,
            validation_split=0.2, callbacks=[early_stop],
            verbose=0 
        )

        # ÉVALUATION ET MESURES
        loss, accuracy,recall = model.evaluate(x_test, y_test, verbose=0)
        
        start_inf = time.time()
        y_probs = model.predict(x_test, verbose=0)
        inference_time_ms = ((time.time() - start_inf) / len(x_test)) * 1000

        # Stockage
        all_acc.append(accuracy)
        all_rec.append(recall)
        all_inf.append(inference_time_ms)
        #print(f"Run {i+1} terminé - Acc: {accuracy*100:.2f}%")
        print(f"Run {i+1} terminé - Acc: {accuracy*100:.2f}% | Recall: {recall*100:.2f}%")

    # SAUVEGARDE DU DERNIER MODÈLE
    model_path = f'best_{model_type.lower()}_model.h5'
    model.save(model_path)
    model_size_kb = os.path.getsize(model_path) / 1024

    # ÉVALUATION FINALE (MOYENNE + ECART-TYPE)
    print("\n" + "="*40)
    print("RAPPORT TECHNIQUE POUR LA CLINIQUE")
    print(f"Modele : {model_type} ({n_runs} runs)")
    print(f"Precision (Accuracy) : {np.mean(all_acc)*100:.2f}% (+/- {np.std(all_acc)*100:.2f}%)")
    print(f"Securite (Recall)    : {np.mean(all_rec)*100:.2f}% (+/- {np.std(all_rec)*100:.2f}%)")
    print(f"Nombre de parametres : {model.count_params()}")
    print(f"Taille du modele     : {model_size_kb:.2f} Ko")
    print(f"Temps d'inference    : {np.mean(all_inf):.4f} ms/ECG")
    print("="*40)

    # VISUALISATIONS (du dernier run pour l'exemple)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Evolution de la Perte (Loss)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Evolution de la Précision (Accuracy)')
    plt.legend()
    plt.show()

    # Matrice de Confusion
    y_pred = (y_probs > 0.5).astype(int)
    plt.figure(figsize=(14, 6))
    ax = plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Infarctus', 'Normal'])
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='.2f')
    plt.title(f"Matrice de Confusion ({model_type})")

    # PCA (Dynamique selon le modèle)
    plt.subplot(1, 2, 2)
    try:
        feat_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_for_pca).output)
        features = feat_extractor.predict(x_train, verbose=0)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', alpha=0.7)
        plt.title(f"Espace Latent (PCA - {model_type})")
        plt.colorbar(label="0: Infarctus, 1: Normal")
    except Exception as e:
        print(f"\nNote: PCA échouée. Erreur: {e}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simple_experiment(model_type="RNN", n_runs=5)
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from core.data_loader import load_ecg_dataset
# from models.rnn import build_rnn_final as build_rnn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.decomposition import PCA
# Imports des différents modèles
# from models.rnn import build_rnn_final as build_rnn
# from models.cnn import build_cnn_final as build_cnn  # À décommenter quand tu les auras
from models.mlp import build_mlp_v1, build_mlp_v2, build_mlp_final

def run_mlp_experiment(mlp_builder, model_name, show_prediction_curve=False):
    print(f"\n>>> DEMARRAGE DE L'EXPERIENCE : {model_name} <<<")
    
    # ADAPTATION DES DONNÉES (MLP utilise 2D)
    is_3d = False
    x_train, x_test, y_train, y_test = load_ecg_dataset(is_3d=is_3d)

    # CONSTRUCTION DU MODÈLE
    model = mlp_builder(input_shape=x_train.shape[1:])
    layer_for_pca = "Hidden_Layer"

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
    model_path = f'best_{model_name}_model.h5'
    model.save(model_path)
    model_size_kb = os.path.getsize(model_path) / 1024

    # Mesure du temps d'inférence
    start_inf = time.time()
    y_probs = model.predict(x_test, verbose=0)
    inference_time_ms = ((time.time() - start_inf) / len(x_test)) * 1000

    # ÉVALUATION FINALE 
    print("\n" + "="*40)
    print(f"RAPPORT : {model_name}")
    print("="*40)
    
    # Gérer les différentes métriques selon le modèle
    results = model.evaluate(x_test, y_test, verbose=0)
    if len(results) == 3:  # loss, accuracy, recall
        loss, accuracy, recall = results
        print(f"Precision (Accuracy) : {accuracy*100:.2f}%")
        print(f"Securité (Recall)    : {recall*100:.2f}%")
    else:  # loss, accuracy (sans recall)
        loss, accuracy = results
        print(f"Precision (Accuracy) : {accuracy*100:.2f}%")
    
    print(f"Loss                 : {loss:.4f}")
    print(f"Nombre de parametres : {model.count_params()}")
    print(f"Taille du modele     : {model_size_kb:.2f} Ko")
    print(f"Temps d'inference    : {inference_time_ms:.4f} ms/ECG")
    print(f"Durée entrainement   : {training_duration:.2f} secondes")
    print("="*40)

    # VISUALISATIONS
    plt.figure(figsize=(14, 5))
    
    # Graphique de la perte (Loss)
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Perte (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Graphique de la précision (Accuracy)
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name} - Précision (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # Matrice de Confusion
    plt.subplot(1, 3, 3)
    y_pred = (y_probs > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Infarctus', 'Normal'])
    disp.plot(ax=plt.gca(), cmap=plt.cm.Blues, values_format='.2f')
    plt.title(f'{model_name} - Matrice de Confusion')

    plt.tight_layout()
    plt.show()

    if show_prediction_curve:
        plot_mlp_final_prediction_curve(y_probs.flatten(), y_test)

    return {
        'model_name': model_name,
        'accuracy': accuracy if 'accuracy' in locals() else None,
        'recall': recall if 'recall' in locals() else None,
        'loss': loss,
        'params': model.count_params(),
        'size_kb': model_size_kb,
        'inference_time_ms': inference_time_ms
    }


def run_simple_experiment(model_type="MLP"):
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
        model = build_mlp_final(input_shape=x_train.shape[1:])
        layer_for_pca = "Hidden_Layer"

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


def plot_mlp_final_prediction_curve(y_probs, y_test):
    """
    Affiche une courbe ROC et la distribution des prédictions
    du MLP final pour les cas 'malade' (Infarctus) vs 'pas malade' (Normal).
    """
    print("\n>>> TRACÉ DE LA COURBE DE DIAGNOSTIC - MLP FINAL <<<\n")

    # Calcul de la courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # Créer les visualisations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Graphique 1: Courbe ROC ---
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('Taux de Faux Positifs (Spécificité)')
    axes[0].set_ylabel('Taux de Vrais Positifs (Sensibilité)')
    axes[0].set_title('Courbe ROC - MLP Final\n(Malade vs Pas Malade)')
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(alpha=0.3)

    # --- Graphique 2: Distribution des probabilités ---
    y_probs_healthy = y_probs[y_test == 0]  # Pas malade (Normal)
    y_probs_sick = y_probs[y_test == 1]     # Malade (Infarctus)

    axes[1].hist(y_probs_healthy, bins=30, alpha=0.7, label='Pas Malade (Normal)', color='blue', edgecolor='black')
    axes[1].hist(y_probs_sick, bins=30, alpha=0.7, label='Malade (Infarctus)', color='red', edgecolor='black')
    axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Seuil de décision (0.5)')
    axes[1].set_xlabel('Probabilité Prédite (Score du Modèle)')
    axes[1].set_ylabel('Nombre d\'échantillons')
    axes[1].set_title('Distribution des Prédictions du MLP Final\n(Malade vs Pas Malade)')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Afficher les statistiques
    print("\n" + "="*60)
    print("STATISTIQUES DE DIAGNOSTIC - MLP FINAL")
    print("="*60)
    print(f"AUC (Area Under Curve) : {roc_auc:.4f}")
    print(f"Nombre d'échantillons 'Pas Malade' : {len(y_probs_healthy)}")
    print(f"Nombre d'échantillons 'Malade' : {len(y_probs_sick)}")
    print(f"\nMoyenne de probabilité 'Pas Malade' : {y_probs_healthy.mean():.4f}")
    print(f"Moyenne de probabilité 'Malade' : {y_probs_sick.mean():.4f}")
    print(f"\nÉcart-type 'Pas Malade' : {y_probs_healthy.std():.4f}")
    print(f"Écart-type 'Malade' : {y_probs_sick.std():.4f}")
    print("="*60)

if __name__ == "__main__":
    # Tester tous les MLP et ne tracer les prédictions que pour le MLP final
    results = []

    print("\n" + "="*60)
    print("EXPERIENCES MLP: ANALYSE POUR TOUS LES ALGORITHMES")
    print("="*60)

    # Analyse MLP v1
    result_v1 = run_mlp_experiment(build_mlp_v1, "MLP_v1_Baseline")
    results.append(result_v1)

    # Analyse MLP v2
    result_v2 = run_mlp_experiment(build_mlp_v2, "MLP_v2_Dropout")
    results.append(result_v2)

    # Analyse et prédiction MLP final
    result_final = run_mlp_experiment(build_mlp_final, "MLP_Final", show_prediction_curve=True)
    results.append(result_final)

    # Résumé comparatif des trois MLP
    print("\n" + "="*60)
    print("RÉSUMÉ COMPARATIF DES MLP")
    print("="*60)
    print(f"{'Modèle':<25} {'Accuracy':<12} {'Loss':<10} {'Params':<10} {'Taille Ko':<10}")
    print("-"*70)
    for r in results:
        acc = f"{r['accuracy']*100:.2f}%" if r['accuracy'] is not None else "N/A"
        print(f"{r['model_name']:<25} {acc:<12} {r['loss']:<10.4f} {r['params']:<10} {r['size_kb']:<10.2f}")
    print("="*60)
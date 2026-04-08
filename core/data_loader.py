import pandas as pd
import numpy as np

def load_ecg_dataset(is_3d=False):
    """
    Charge et prépare le dataset ECG200 
    is_3d : True pour CNN/RNN (3D), False pour MLP (2D)
    """
    
    url_train = "https://maxime-devanne.com/datasets/ECG200/ECG200_TRAIN.tsv"
    url_test = "https://maxime-devanne.com/datasets/ECG200/ECG200_TEST.tsv"

    
    train_df = pd.read_csv(url_train, sep='\t', header=None)
    test_df = pd.read_csv(url_test, sep='\t', header=None)

    
    y_train = train_df.iloc[:, 0].values
    x_train = train_df.iloc[:, 1:].values
    
    y_test = test_df.iloc[:, 0].values
    x_test = test_df.iloc[:, 1:].values

    # 4. Nettoyage des labels : {-1, 1} -> {0, 1}
    y_train = np.where(y_train == -1, 0, 1)
    y_test = np.where(y_test == -1, 0, 1)

    # 5. Normalisation 
    # On calcule la moyenne et l'écart-type sur le TRAIN uniquement
    mean = x_train.mean()
    std = x_train.std()
    
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # 6. Adaptation de la dimension 
    if is_3d:
        # Passage de (N, 96) à (N, 96, 1) pour Conv1D ou LSTM
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    # Test du chargement pour le MLP 
    x_train, x_test, y_train, y_test = load_ecg_dataset(is_3d=False)
    
    print("--- VERIFICATION MLP (2D) ---")
    print(f"X_train shape: {x_train.shape}") # Doit être (100, 96)
    print(f"Labels uniques: {np.unique(y_train)}") # Doit être [0. 1.]
    print(f"Moyenne (0?): {x_train.mean():.3f}")
    print(f"Ecart-type (1?): {x_train.std():.3f}")
    
    # Test du chargement pour CNN/RNN (3D)
    x_train_3d, _, _, _ = load_ecg_dataset(is_3d=True)
    print("\n--- VERIFICATION CNN/RNN (3D) ---")
    print(f"X_train_3d shape: {x_train_3d.shape}") # Doit être (100, 96, 1)
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(name="PWD", test_size=0.2, random_state=42):
    """
    Carga un dataset desde datasets/PWD o datasets/WPD.

    Parámetros:
        name (str): "PWD" o "WPD"
        test_size (float): proporción de test
        random_state (int): semilla para split reproducible

    Retorna:
        X_train, X_test, y_train, y_test
    """
    path = f"datasets/{name}/nombre del dataset}" # aca pone el dataset que quieras cargar
    df = pd.read_csv(path)

    # Se asume que la columna objetivo se llama "target"
    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

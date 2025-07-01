import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedKFold

def load_arff_dataset(name="WPD"):
    """
    Carga y retorna X, y desde un archivo .arff dentro de /data/WPD o /data/PWD
    """
    base_path = "data/"
    if name == "WPD":
        path = base_path + "WPD/dataset2.arff"
    elif name == "PWD":
        path = base_path + "PWD/dataset1.arff"
    else:
        raise ValueError("El nombre debe ser 'WPD' o 'PWD'")

    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    # Decodificar strings si es necesario
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8")

    X = df.iloc[:, :-1].astype(float)
    y = df.iloc[:, -1].astype(int)
    return X, y


def load_spam_dataset(path="data/spam.csv"):
    """
    Carga el dataset de spam en formato CSV
    """
    df = pd.read_csv(path, encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df


def get_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Retorna X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def get_kfold(X, y, n_splits=5, random_state=42):
    """
    Retorna un objeto StratifiedKFold
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# loader.py
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def load_dataset(name="PWD"):
    if name == "PWD":
        path = "datasets/PWD/Training Dataset.arff"
    elif name == "WPD":
        path = "datasets/WPD/PhishingData.arff"
    else:
        raise ValueError("Nombre de dataset inválido")

    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    
    # Asegurar que los datos no sean byte strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8")

    # Suposición: la última columna es la clase
    X = df.iloc[:, :-1].astype(float)
    y = df.iloc[:, -1]
    return X, y

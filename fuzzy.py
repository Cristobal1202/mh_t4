import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Dividir en entrenamiento y prueba (reseteando índices)
X = df['message']
y = df['label']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_raw = X_train_raw.reset_index(drop=True)
X_test_raw = X_test_raw.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Vectorizar texto
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train_raw)
X_test_vec = vectorizer.transform(X_test_raw)

# Calcular frecuencia relativa de palabras por clase
spam_rows = y_train[y_train == 1].index
ham_rows = y_train[y_train == 0].index

spam_sum = X_train_vec[spam_rows].sum(axis=0)
ham_sum = X_train_vec[ham_rows].sum(axis=0)

spam_freq = np.asarray(spam_sum).flatten()
ham_freq = np.asarray(ham_sum).flatten()
total_freq = spam_freq + ham_freq

# Calcular grado de pertenencia difusa (evitando divisiones por cero)
with np.errstate(divide='ignore', invalid='ignore'):
    spam_ratio = np.divide(spam_freq, total_freq, out=np.zeros_like(spam_freq, dtype=float), where=total_freq != 0)


# Predicción basada en promedio de pertenencias difusas
def fuzzy_predict(X_matrix, threshold=0.5):
    preds = []
    for row in X_matrix:
        word_indices = row.indices
        ratios = spam_ratio[word_indices]
        spam_score = np.mean(ratios) if len(ratios) > 0 else 0
        preds.append(1 if spam_score >= threshold else 0)
    return np.array(preds)

# Evaluar modelo
y_pred = fuzzy_predict(X_test_vec)

print("Reporte de Clasificación (Modelo Difuso):\n")
print(classification_report(y_test, y_pred))
print("Matriz de Confusión:\n")
print(confusion_matrix(y_test, y_pred))

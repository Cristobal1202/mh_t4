import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
import random
from collections import deque

# ----------------------------
# Cargar y preparar datos
# ----------------------------
df = pd.read_csv("phishing.csv")

print("Columnas cargadas:", df.columns.tolist())

# Filtrar valores ambiguos
df = df[df['Result'] != -1]
df['Result'] = df['Result'].map({1:1, 0:0})

X = df.drop(columns=["Result"])
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Evaluación de máscara
# ----------------------------
def feature_eval(X_train, X_test, y_train, y_test, mask):
    selected_features = X_train.columns[mask == 1]
    if len(selected_features) == 0:
        return 0.0
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train[selected_features], y_train)
    preds = clf.predict(X_test[selected_features])
    return accuracy_score(y_test, preds)

# ----------------------------
# Metaheurísticas
# ----------------------------

def simulated_annealing(max_iter=1000, initial_temp=1.0, cooling_rate=0.995):
    n_features = X_train.shape[1]
    current_mask = np.ones(n_features, dtype=int)
    best_mask = current_mask.copy()
    best_score = feature_eval(X_train, X_test, y_train, y_test, current_mask)
    current_score = best_score
    temp = initial_temp
    history = [best_score]

    for iter in range(max_iter):
        candidate_mask = current_mask.copy()
        flip = random.randint(0, n_features-1)
        candidate_mask[flip] = 1 - candidate_mask[flip]
        candidate_score = feature_eval(X_train, X_test, y_train, y_test, candidate_mask)

        if candidate_score > current_score:
            current_mask = candidate_mask
            current_score = candidate_score
            if candidate_score > best_score:
                best_score = candidate_score
                best_mask = candidate_mask
        else:
            prob = np.exp((candidate_score - current_score)/temp)
            if random.random() < prob:
                current_mask = candidate_mask
                current_score = candidate_score
        temp *= cooling_rate
        history.append(best_score)

        if iter % 100 == 0:
            print(f"[SA] Iter {iter}, Best Acc: {best_score:.4f}")
    return best_mask, best_score, history

def hill_climbing(max_iter=1000):
    n_features = X_train.shape[1]
    current_mask = np.ones(n_features, dtype=int)
    best_mask = current_mask.copy()
    best_score = feature_eval(X_train, X_test, y_train, y_test, current_mask)
    history = [best_score]

    for iter in range(max_iter):
        candidate_mask = current_mask.copy()
        flip = random.randint(0, n_features-1)
        candidate_mask[flip] = 1 - candidate_mask[flip]
        candidate_score = feature_eval(X_train, X_test, y_train, y_test, candidate_mask)

        if candidate_score >= best_score:
            current_mask = candidate_mask
            best_score = candidate_score
            best_mask = candidate_mask
        history.append(best_score)

        if iter % 100 == 0:
            print(f"[HC] Iter {iter}, Best Acc: {best_score:.4f}")
    return best_mask, best_score, history

def tabu_search(max_iter=1000, tabu_size=50):
    n_features = X_train.shape[1]
    current_mask = np.ones(n_features, dtype=int)
    best_mask = current_mask.copy()
    best_score = feature_eval(X_train, X_test, y_train, y_test, current_mask)
    tabu_list = deque(maxlen=tabu_size)
    history = [best_score]

    for iter in range(max_iter):
        candidate_mask = current_mask.copy()
        flip = random.randint(0, n_features-1)
        candidate_mask[flip] = 1 - candidate_mask[flip]

        if tuple(candidate_mask) in tabu_list:
            continue

        candidate_score = feature_eval(X_train, X_test, y_train, y_test, candidate_mask)

        if candidate_score > best_score:
            best_score = candidate_score
            best_mask = candidate_mask
            current_mask = candidate_mask
        else:
            current_mask = candidate_mask

        tabu_list.append(tuple(candidate_mask))
        history.append(best_score)

        if iter % 100 == 0:
            print(f"[TS] Iter {iter}, Best Acc: {best_score:.4f}")
    return best_mask, best_score, history

# ----------------------------
# Ejecutar
# ----------------------------
print("\n=== SA ===")
sa_mask, sa_acc, sa_history = simulated_annealing()
print("\n=== HC ===")
hc_mask, hc_acc, hc_history = hill_climbing()
print("\n=== TS ===")
ts_mask, ts_acc, ts_history = tabu_search()

# ----------------------------
# Métricas finales
# ----------------------------
clf = LogisticRegression(max_iter=200)

# SA
clf.fit(X_train[X_train.columns[sa_mask==1]], y_train)
sa_pred = clf.predict(X_test[X_test.columns[sa_mask==1]])
sa_metrics = {
    "accuracy": accuracy_score(y_test, sa_pred),
    "recall": recall_score(y_test, sa_pred),
    "precision": precision_score(y_test, sa_pred),
    "f1": f1_score(y_test, sa_pred)
}

# HC
clf.fit(X_train[X_train.columns[hc_mask==1]], y_train)
hc_pred = clf.predict(X_test[X_test.columns[hc_mask==1]])
hc_metrics = {
    "accuracy": accuracy_score(y_test, hc_pred),
    "recall": recall_score(y_test, hc_pred),
    "precision": precision_score(y_test, hc_pred),
    "f1": f1_score(y_test, hc_pred)
}

# TS
clf.fit(X_train[X_train.columns[ts_mask==1]], y_train)
ts_pred = clf.predict(X_test[X_test.columns[ts_mask==1]])
ts_metrics = {
    "accuracy": accuracy_score(y_test, ts_pred),
    "recall": recall_score(y_test, ts_pred),
    "precision": precision_score(y_test, ts_pred),
    "f1": f1_score(y_test, ts_pred)
}

print("\n=== Métricas finales ===")
print(f"SA: {sa_metrics}")
print(f"HC: {hc_metrics}")
print(f"TS: {ts_metrics}")

# ----------------------------
# Graficar convergencia
# ----------------------------
plt.figure(figsize=(10,6))
plt.plot(sa_history, label="Simulated Annealing", color="blue", linewidth=2)
plt.plot(hc_history, label="Hill Climbing", color="green", linewidth=2)
plt.plot(ts_history, label="Tabu Search", color="orange", linewidth=2)
plt.xlabel("Generation Number")
plt.ylabel("Fitness Value (Accuracy)")
plt.title("Convergence of Metaheuristics on Phishing Dataset")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# ----------------------------
# Graficar radar chart mejorado
# ----------------------------
labels = ["accuracy", "recall", "precision", "f1"]
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Convertir a listas + cerrar polígono
sa_values = list(sa_metrics.values()) + [sa_metrics["accuracy"]]
hc_values = list(hc_metrics.values()) + [hc_metrics["accuracy"]]
ts_values = list(ts_metrics.values()) + [ts_metrics["accuracy"]]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# SA
ax.plot(angles, sa_values, color="blue", linewidth=2.5, marker='o', markersize=5, label="Simulated Annealing")
ax.fill(angles, sa_values, color="blue", alpha=0.15)

# HC
ax.plot(angles, hc_values, color="green", linewidth=2.5, marker='s', markersize=5, label="Hill Climbing")
ax.fill(angles, hc_values, color="green", alpha=0.15)

# TS
ax.plot(angles, ts_values, color="orange", linewidth=2.5, marker='^', markersize=5, label="Tabu Search")
ax.fill(angles, ts_values, color="orange", alpha=0.15)

ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)
ax.set_ylim(0, 1)
ax.set_title("Metaheuristic Performance Radar Chart", fontsize=14)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

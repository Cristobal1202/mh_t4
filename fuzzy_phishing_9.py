import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import random
from collections import deque, defaultdict
import warnings
from src.optimizers.boa import butterfly_optimization
from src.optimizers.hc import hill_climbing
from src.optimizers.sa import simulated_annealing
from src.optimizers.ts import tabu_search

# --- Ignorar advertencias para una salida más limpia ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# -------------------------------------
# 1. Cargar y Preparar los Datos
# -------------------------------------
try:
    df = pd.read_csv("phishing.csv")
except FileNotFoundError:
    print("Error: 'phishing.csv' no encontrado. Por favor, asegúrese de que el archivo esté en el directorio correcto.")
    exit()

# Filtrar valores y mapear la variable objetivo
df = df[df['Result'] != -1]
df['Result'] = df['Result'].map({1: 1, 0: 0})

X = df.drop(columns=["Result"])
y = df["Result"]

# División única de datos para asegurar que todos los algoritmos se prueben en el mismo conjunto
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------------------
# 2. Configuración de la Comparación Rigurosa
# -------------------------------------
N_RUNS = 10  # Número de ejecuciones para promediar resultados
MAX_EVALUATIONS = 100  # Presupuesto de evaluación de fitness
POP_SIZE_BOA = 20

# -------------------------------------
# 3. Función de Evaluación (Fitness)
# -------------------------------------

def feature_eval(mask, X_train, X_test, y_train, y_test):
    selected_features = X_train.columns[np.array(mask, dtype=bool)]
    if len(selected_features) == 0:
        return 0.0
    clf = LogisticRegression(max_iter=200, solver='liblinear')
    clf.fit(X_train[selected_features], y_train)
    preds = clf.predict(X_test[selected_features])
    return accuracy_score(y_test, preds)

# -------------------------------------
# 4. Wrapper functions for imported metaheuristics
# -------------------------------------

def sa_wrapper(X_train, X_test, y_train, y_test, max_evals, **kwargs):
    """Wrapper for simulated annealing"""
    def objective_func(mask):
        # Convert continuous values to binary mask
        binary_mask = np.round(np.clip(mask, 0, 1)).astype(int)
        return feature_eval(binary_mask, X_train, X_test, y_train, y_test)
    
    n_features = X_train.shape[1]
    sa = simulated_annealing(
        obj_function=objective_func,
        dim=n_features,
        lower_bound=0.0,
        upper_bound=1.0,
        max_iter=max_evals,
        **kwargs
    )
    best_solution, best_fitness, history = sa.optimize()
    best_mask = np.round(np.clip(best_solution, 0, 1)).astype(int)
    return best_mask, best_fitness, history

def hc_wrapper(X_train, X_test, y_train, y_test, max_evals, **kwargs):
    """Wrapper for hill climbing"""
    hc = hill_climbing(max_iter=max_evals, **kwargs)
    best_mask, best_fitness, history = hc.optimize(X_train, y_train)
    return best_mask, best_fitness, history

def ts_wrapper(X_train, X_test, y_train, y_test, max_evals, **kwargs):
    """Wrapper for tabu search"""
    ts = tabu_search(max_iter=max_evals, **kwargs)
    best_mask, best_fitness, history = ts.run(X_train, y_train)
    return best_mask, best_fitness, history

def boa_wrapper(X_train, X_test, y_train, y_test, max_iter, pop_size, **kwargs):
    """Wrapper for BOA"""
    def objective_func(mask):
        # Convert continuous values to binary mask
        binary_mask = np.round(np.clip(mask, 0, 1)).astype(int)
        return feature_eval(binary_mask, X_train, X_test, y_train, y_test)
    
    n_features = X_train.shape[1]
    boa = butterfly_optimization(
        obj_function=objective_func,
        dim=n_features,
        lower_bound=0.0,
        upper_bound=1.0,
        max_iter=max_iter,
        pop_size=pop_size,
        **kwargs
    )
    best_solution, best_fitness = boa.optimize()
    history = boa.get_fitness_history()
    best_mask = np.round(np.clip(best_solution, 0, 1)).astype(int)
    return best_mask, best_fitness, history


# -------------------------------------
# 5. Bucle de Ejecución y Recolección de Resultados
# -------------------------------------
results = defaultdict(lambda: defaultdict(list))
algorithms = {
    "SA": sa_wrapper,
    "HC": hc_wrapper,
    "TS": ts_wrapper,
    "BOA": boa_wrapper
}

print("Iniciando comparación rigurosa de metaheurísticas...")

for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"\n--- KFold {fold_idx + 1}/{kf.get_n_splits()} ---")
    X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
    y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]
    
    for name, algorithm in algorithms.items():
        print(f"Algoritmo: {name}")
        
        # Ejecutar múltiples veces por cada fold
        for run in range(N_RUNS):
            if name == "BOA":
                result = algorithm(
                    X_train_kf, X_test_kf, y_train_kf, y_test_kf,
                    max_iter=MAX_EVALUATIONS // POP_SIZE_BOA,
                    pop_size=POP_SIZE_BOA
                )
            else:
                result = algorithm(
                    X_train_kf, X_test_kf, y_train_kf, y_test_kf,
                    max_evals=MAX_EVALUATIONS
                )

            # Extract results (format may vary from imported functions)
            if isinstance(result, tuple) and len(result) >= 3:
                best_mask, best_fitness, history = result[:3]
            else:
                # Handle case where imported function returns different format
                best_mask = result if not isinstance(result, tuple) else result[0]
                best_fitness = 0
                history = [0] * MAX_EVALUATIONS

            # Ensure best_mask is binary
            if hasattr(best_mask, '__iter__'):
                best_mask = np.array(best_mask, dtype=int)
            else:
                # If single value, create random mask as fallback
                best_mask = np.random.randint(0, 2, X_train_kf.shape[1])

            # Guardar historial
            results[name]["histories"].append(history)

            # Calcular métricas finales
            selected_features = X_train_kf.columns[np.array(best_mask, dtype=bool)]
            if len(selected_features) > 0:
                clf = LogisticRegression(max_iter=200, solver='liblinear')
                clf.fit(X_train_kf[selected_features], y_train_kf)
                final_preds = clf.predict(X_test_kf[selected_features])
                results[name]["metrics"].append({
                    "accuracy": accuracy_score(y_test_kf, final_preds),
                    "recall": recall_score(y_test_kf, final_preds),
                    "precision": precision_score(y_test_kf, final_preds),
                    "f1": f1_score(y_test_kf, final_preds)
                })
            else:
                results[name]["metrics"].append({
                    "accuracy": 0, "recall": 0, "precision": 0, "f1": 0
                })


# -------------------------------------
# 6. Procesamiento y Presentación de Resultados
# -------------------------------------
summary_data = []
for name in algorithms.keys():
    metrics_df = pd.DataFrame(results[name]["metrics"])
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()
    summary_data.append({
        "Algorithm": name,
        "Avg. Accuracy": f"{mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}",
        "Avg. F1-Score": f"{mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}",
        "Avg. Precision": f"{mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}",
        "Avg. Recall": f"{mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}",
    })
    results[name]["avg_metrics"] = mean_metrics # Guardar para el gráfico de radar

summary_df = pd.DataFrame(summary_data)
print("\n=== Resumen de Rendimiento Promedio (10 Ejecuciones) ===")
print(summary_df.to_string(index=False))

# -------------------------------------
# 7. Gráfico de Evolución Promedio con Desviación Estándar
# -------------------------------------
plt.figure(figsize=(12, 7))
colors = {"SA": "blue", "HC": "green", "TS": "orange", "BOA": "red"}

for name in algorithms.keys():
    histories_list = results[name]["histories"]
    
    # Ensure all histories have the same length by padding or truncating
    max_length = MAX_EVALUATIONS
    normalized_histories = []
    
    for history in histories_list:
        if len(history) < max_length:
            # Pad with the last value
            padded_history = list(history) + [history[-1]] * (max_length - len(history))
        elif len(history) > max_length:
            # Truncate to max_length
            padded_history = history[:max_length]
        else:
            padded_history = history
        normalized_histories.append(padded_history)
    
    # Convert to numpy array now that all histories have the same length
    histories = np.array(normalized_histories)
    mean_history = histories.mean(axis=0)
    std_history = histories.std(axis=0)
    
    # Create evaluation steps
    evals = np.arange(len(mean_history))
    
    plt.plot(evals, mean_history, label=name, color=colors[name], linewidth=2)
    plt.fill_between(evals, mean_history - std_history, mean_history + std_history,
                     color=colors[name], alpha=0.15)

plt.xlabel("Número de Evaluaciones de Fitness")
plt.ylabel("Mejor Fitness (Accuracy)")
plt.title("Convergencia Promedio de Metaheurísticas (+/- Desv. Est.)")
plt.legend(loc='lower right')
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlim(0, MAX_EVALUATIONS)
plt.ylim(bottom=0.85) # Ajustar para mejor visualización
plt.tight_layout()
plt.show()


# -------------------------------------
# 8. Gráfico de Radar con Métricas Promedio
# -------------------------------------
metrics_labels = ["accuracy", "recall", "precision", "f1"]
metric_names = ["Accuracy", "Recall", "Precision", "F1-Score"]
algorithms_list = list(algorithms.keys())
num_algorithms = len(algorithms_list)
angles = np.linspace(0, 2 * np.pi, num_algorithms, endpoint=False).tolist()
angles += angles[:1]  # cerrar el radar

# Preparar datos: una lista por cada métrica
metric_data = {metric: [] for metric in metrics_labels}
for metric in metrics_labels:
    for alg in algorithms_list:
        avg_value = results[alg]["avg_metrics"][metric]
        metric_data[metric].append(avg_value)
    metric_data[metric].append(metric_data[metric][0])  # cerrar el círculo

# Graficar
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

colors_metric = {
    "accuracy": "blue",
    "recall": "green",
    "precision": "orange",
    "f1": "red"
}

for metric, color in colors_metric.items():
    ax.plot(angles, metric_data[metric], label=metric.capitalize(), color=color, linewidth=2, marker='o')
    ax.fill(angles, metric_data[metric], color=color, alpha=0.15)

ax.set_thetagrids(np.degrees(angles[:-1]), algorithms_list, fontsize=12)
ax.set_ylim(0, 1.0)
ax.set_title("Comparación de Métricas por Metaheurística", fontsize=16, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()
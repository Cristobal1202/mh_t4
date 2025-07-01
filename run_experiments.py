# experiments/run_experiments.py
from src.data_loader.loader import load_arff_dataset
from src.optimizers.boa import BOA
from src.evaluation.objective_function import objective_function
import matplotlib.pyplot as plt
import numpy as np
import os

# Crear directorio de resultados si no existe
os.makedirs('results', exist_ok=True)

# Cargar datos
X, y = load_arff_dataset("WPD")

# Dimensión del vector de parámetros
dim = 9 * 3 * 3  # 9 variables, 3 etiquetas, 3 valores cada una

# Envolver la función objetivo con los datos
def wrapped_obj(vector):
    return objective_function(vector, X, y)

# Ejecutar BOA
boa = BOA(
    obj_function=wrapped_obj,
    dim=dim,
    lower_bound=0.0,
    upper_bound=1.0,
    pop_size=20,
    max_iter=50,
    verbose=True
)

best_vector, best_fitness = boa.optimize()

print("\nMejor accuracy alcanzado:", best_fitness, "%")

# Crear gráfico de evolución de fitness
if hasattr(boa, 'fitness_history'):
    plt.figure(figsize=(10, 6))
    iterations = range(1, len(boa.fitness_history) + 1)
    fitness_values = boa.fitness_history
    
    plt.plot(iterations, fitness_values, 'g-', linewidth=2, label='Best Fitness', marker='o', markersize=4)
    plt.xlabel('Generación')
    plt.ylabel('Fitness (Accuracy %)')
    plt.title('Evolución del Fitness durante la Optimización BOA')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 100)  # Set y-axis limits from 0 to 100%
    plt.tight_layout()
    
    # Mostrar estadísticas
    print(f"\nEstadísticas del fitness:")
    print(f"Fitness inicial: {fitness_values[0]:.4f}%")
    print(f"Fitness final: {fitness_values[-1]:.4f}%")
    print(f"Mejora total: {fitness_values[-1] - fitness_values[0]:.4f}%")
    print(f"Número de generaciones: {len(fitness_values)}")
    
    # Guardar el gráfico
    plt.savefig('results/boa_fitness_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Gráfico guardado en: results/boa_fitness_evolution.png")
else:
    print("No se encontró historial de fitness para graficar")

import numpy as np
import matplotlib.pyplot as plt

class BOA:
    def __init__(self, obj_function, dim, lower_bound, upper_bound, 
                 pop_size=20, max_iter=100, 
                 sensory_modality=0.01, power_exponent=0.1, switch_prob=0.8,
                 verbose=False):
        """
        Implementación del Butterfly Optimization Algorithm (BOA)

        obj_function: función objetivo que recibe un vector y retorna un fitness
        dim: número de dimensiones del problema
        lower_bound: límite inferior de búsqueda (float o array)
        upper_bound: límite superior de búsqueda (float o array)
        """
        self.obj_function = obj_function
        self.dim = dim
        self.lower_bound = np.array([lower_bound] * dim) if isinstance(lower_bound, (int, float)) else np.array(lower_bound)
        self.upper_bound = np.array([upper_bound] * dim) if isinstance(upper_bound, (int, float)) else np.array(upper_bound)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.c = sensory_modality
        self.a = power_exponent
        self.p = switch_prob
        self.verbose = verbose

        # Historial de fitness por iteración
        self.fitness_history = []

    def _fragrance(self, I):
        return self.c * (I ** self.a)

    def optimize(self):
        # Inicializar población aleatoria
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([self.obj_function(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for t in range(self.max_iter):
            new_population = []
            for i in range(self.pop_size):
                fragrance = self._fragrance(fitness[i])
                r = np.random.rand()
                if r < self.p:
                    # Movimiento hacia la mejor mariposa
                    new_pos = population[i] + fragrance * (best_solution - population[i]) * np.random.rand()
                else:
                    # Movimiento aleatorio
                    j, k = np.random.choice(self.pop_size, 2, replace=False)
                    new_pos = population[i] + fragrance * (population[j] - population[k]) * np.random.rand()

                # Límite de búsqueda
                new_pos = np.clip(new_pos, self.lower_bound, self.upper_bound)
                new_population.append(new_pos)

            # Actualizar población y fitness
            population = np.array(new_population)
            fitness = np.array([self.obj_function(ind) for ind in population])
            current_best_idx = np.argmin(fitness)

            # Actualizar mejor solución
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = population[current_best_idx].copy()

            self.fitness_history.append(best_fitness)

            if self.verbose:
                print(f"Iteración {t+1} | Mejor fitness: {best_fitness:.5f}")

        return best_solution, best_fitness

    def get_fitness_history(self):
        return self.fitness_history


def sphere(x):
    return np.sum(x**2)

boa = BOA(obj_function=sphere, dim=5, lower_bound=-5, upper_bound=5, 
          pop_size=30, max_iter=100, verbose=True)

best_sol, best_fit = boa.optimize()
print("Mejor solución encontrada:", best_sol)
print("Fitness:", best_fit)


plt.plot(boa.get_fitness_history())
plt.title("Convergencia del Fitness - BOA")
plt.xlabel("Iteración")
plt.ylabel("Fitness")
plt.grid()
plt.show()
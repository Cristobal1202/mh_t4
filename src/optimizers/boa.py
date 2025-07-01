import numpy as np

class BOA:
    def __init__(self, obj_function, dim, lower_bound, upper_bound, 
                 pop_size=20, max_iter=100, 
                 sensory_modality=0.01, power_exponent=0.1, switch_prob=0.8,
                 verbose=False):
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

        self.fitness_history = []
        self.best_solution = None
        self.best_fitness = None

    def _fragrance(self, I):
        return self.c * (I ** self.a)

    def optimize(self):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([self.obj_function(ind if self.dim > 1 else ind[0]) for ind in population])
        best_idx = np.argmax(fitness)  # Changed to argmax for maximization
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for t in range(self.max_iter):
            new_population = []
            for i in range(self.pop_size):
                fragrance = self._fragrance(fitness[i])
                r = np.random.rand()
                if r < self.p:
                    new_pos = population[i] + fragrance * (best_solution - population[i]) * np.random.rand()
                else:
                    j, k = np.random.choice(self.pop_size, 2, replace=False)
                    new_pos = population[i] + fragrance * (population[j] - population[k]) * np.random.rand()

                new_pos = np.clip(new_pos, self.lower_bound, self.upper_bound)
                new_population.append(new_pos)

            population = np.array(new_population)
            fitness = np.array([self.obj_function(ind if self.dim > 1 else ind[0]) for ind in population])
            current_best_idx = np.argmax(fitness)  # Changed to argmax for maximization

            if fitness[current_best_idx] > best_fitness:  # Changed to > for maximization
                best_fitness = fitness[current_best_idx]
                best_solution = population[current_best_idx].copy()

            self.fitness_history.append(best_fitness)

            if self.verbose:
                print(f"Iteración {t+1} | Mejor fitness: {best_fitness:.5f}")

        self.best_solution = best_solution
        self.best_fitness = best_fitness
        return best_solution, best_fitness

    def get_fitness_history(self):
        return self.fitness_history

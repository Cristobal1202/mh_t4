import numpy as np

class simulated_annealing:
    def __init__(self, obj_function, dim, lower_bound, upper_bound,
                 max_iter=100, initial_temp=1000, cooling_rate=0.95):
        self.obj_function = obj_function
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def optimize(self):
        current_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        current_score = self.obj_function(current_solution)
        best_solution = current_solution.copy()
        best_score = current_score

        temp = self.initial_temp
        history = []

        for iter in range(self.max_iter):
            candidate_solution = current_solution + np.random.uniform(-1, 1, self.dim) * temp
            candidate_solution = np.clip(candidate_solution, self.lower_bound, self.upper_bound)
            candidate_score = self.obj_function(candidate_solution)

            if candidate_score > current_score or np.random.rand() < np.exp((candidate_score - current_score) / temp):
                current_solution = candidate_solution
                current_score = candidate_score

            if candidate_score > best_score:
                best_solution = candidate_solution.copy()
                best_score = candidate_score

            history.append(best_score)

            temp *= self.cooling_rate

            if iter % 20 == 0:
                print(f"[SA] Iter {iter}, Best Acc: {best_score:.4f}")

        return best_solution, best_score, history
# run_experiments.py
from metaheuristicas.boa import BOA
from datasets.loader import load_dataset
from metaheuristicas.utils import fitness_function

X_train, X_test, y_train, y_test = load_dataset()
boa = BOA(...)
y_pred = boa.optimize(fitness_function, X_train, y_train)

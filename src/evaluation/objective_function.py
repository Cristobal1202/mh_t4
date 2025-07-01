import numpy as np
from sklearn.metrics import accuracy_score
from src.fuzzy_model.model import FuzzyClassifier
from src.fuzzy_model.rules import rules

VARIABLES = [
    "SFH", "popUpWidnow", "SSLfinal_State", "Request_URL",
    "URL_of_Anchor", "web_traffic", "URL_Length",
    "age_of_domain", "having_IP_Address"
]

LABELS = ["low", "medium", "high"]
VALUES = [-1, 0, 1]

def vector_to_membership_params(vector):
    """
    Convierte el vector plano a un diccionario membership_params compatible con el modelo
    """
    membership_params = {}
    idx = 0
    for var in VARIABLES:
        membership_params[var] = {}
        for label in LABELS:
            membership_params[var][label] = {}
            for val in VALUES:
                membership_params[var][label][val] = np.clip(vector[idx], 0.0, 1.0)
                idx += 1
    return membership_params

def objective_function(vector, X, y_true):
    """
    Función objetivo para BOA. Retorna accuracy*100
    Fitness va de 0 a 100, donde 100 es el mejor (100% accuracy)
    """
    params = vector_to_membership_params(vector)
    model = FuzzyClassifier(rules, params)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy * 100.0

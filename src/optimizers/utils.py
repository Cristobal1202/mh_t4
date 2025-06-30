from sklearn.metrics import f1_score

def fitness_function(y_true, y_pred):
    return f1_score(y_true, y_pred)

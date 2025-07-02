import numpy as np
from sklearn.linear_model import LogisticRegression

class hill_climbing:
    def __init__(self, max_iter=100):
        self.max_iter = max_iter

    def optimize(self, X_train, y_train):
        n_features = X_train.shape[1]
        current_mask = np.ones(n_features, dtype=int)
        best_mask = current_mask.copy()
        best_score = 0.0
        history = []

        for iter in range(self.max_iter):
            candidate_mask = current_mask.copy()
            feature_to_flip = np.random.randint(0, n_features)
            candidate_mask[feature_to_flip] = 1 - candidate_mask[feature_to_flip]

            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train[X_train.columns[candidate_mask == 1]], y_train)
            score = clf.score(X_train[X_train.columns[candidate_mask == 1]], y_train)

            if score > best_score:
                best_score = score
                best_mask = candidate_mask

            history.append(best_score)

            if iter % 20 == 0:
                print(f"[HC] Iter {iter}, Best Acc: {best_score:.4f}")

        return best_mask, best_score, history
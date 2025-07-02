from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import deque

class tabu_search:
    def __init__(self, max_iter=1000, tabu_size=50):
        self.max_iter = max_iter
        self.tabu_size = tabu_size

    def run(self, X_train, y_train):
        n_features = X_train.shape[1]
        current_mask = np.ones(n_features, dtype=int)
        best_mask = current_mask.copy()
        
        # Calculate initial score
        clf = LogisticRegression(max_iter=200, solver='liblinear')
        clf.fit(X_train[X_train.columns[current_mask == 1]], y_train)
        best_score = clf.score(X_train[X_train.columns[current_mask == 1]], y_train)
        
        history = [best_score]  # Start with initial score
        tabu_list = deque(maxlen=self.tabu_size)
        tabu_list.append(tuple(current_mask))  # Add initial solution to tabu

        for iter in range(self.max_iter):
            candidate_masks = []
            for i in range(n_features):
                candidate_mask = current_mask.copy()
                candidate_mask[i] = 1 - candidate_mask[i]
                if tuple(candidate_mask) not in tabu_list:
                    candidate_masks.append(candidate_mask)

            # If no non-tabu candidates, choose the best from all neighbors
            if not candidate_masks:
                for i in range(n_features):
                    candidate_mask = current_mask.copy()
                    candidate_mask[i] = 1 - candidate_mask[i]
                    candidate_masks.append(candidate_mask)

            scores = []
            for candidate_mask in candidate_masks:
                selected_features = X_train.columns[candidate_mask == 1]
                if len(selected_features) > 0:
                    clf = LogisticRegression(max_iter=200, solver='liblinear')
                    clf.fit(X_train[selected_features], y_train)
                    score = clf.score(X_train[selected_features], y_train)
                else:
                    score = 0.0
                scores.append(score)

            best_candidate_index = np.argmax(scores)
            best_candidate_mask = candidate_masks[best_candidate_index]
            best_candidate_score = scores[best_candidate_index]

            # Always move to the best candidate (tabu search characteristic)
            current_mask = best_candidate_mask
            
            # Update global best if better
            if best_candidate_score > best_score:
                best_score = best_candidate_score
                best_mask = best_candidate_mask

            tabu_list.append(tuple(best_candidate_mask))
            history.append(best_score)

            if iter % 20 == 0:
                print(f"[TS] Iter {iter}, Best Acc: {best_score:.4f}")

        return best_mask, best_score, history
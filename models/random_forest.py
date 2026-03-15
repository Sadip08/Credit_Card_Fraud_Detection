import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=15, max_depth=6, max_samples=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.trees = []

    def fit(self, X, y):
        n_samples = int(len(X) * self.max_samples)
        for _ in range(self.n_trees):
            idx = np.random.choice(len(X), n_samples, replace=True) # Bootstrap sampling
            X_boot = X.iloc[idx].copy()
            y_boot = y.iloc[idx].copy()

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X): #predict classes (0 or 1)
        predictions = np.zeros((len(X), self.n_trees))
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        return np.round(np.mean(predictions, axis=1)).astype(int)
    
    def predict_proba(self, X):
        """
        Return probability estimates in sklearn format: 
        shape = (n_samples, 2) → [P(class=0), P(class=1)]
        """
        n_samples = len(X)
        tree_probs = np.zeros((n_samples, self.n_trees))
        
        for i, tree in enumerate(self.trees):
            tree_probs[:, i] = tree.predict(X)   # Each tree returns 0 or 1
        
        # Average probability of fraud (class 1)
        prob_class1 = np.mean(tree_probs, axis=1)
        prob_class0 = 1 - prob_class1
        
        # Return in standard sklearn format
        return np.column_stack((prob_class0, prob_class1))
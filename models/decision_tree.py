import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=6, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _entropy(self, y):
        """Calculate entropy using log base 2"""
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        p = p[p > 0]  # avoid log(0)
        return -np.sum(p * np.log2(p))

    def _information_gain(self, y, y_left, y_right):
        """Calculate Information Gain"""
        parent_entropy = self._entropy(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
            
        weighted_entropy = (n_left / n) * self._entropy(y_left) + (n_right / n) * self._entropy(y_right)
        return parent_entropy - weighted_entropy

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds[::10]:  # sampling every 10th unique value as threshold
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                
                if len(y[left_idx]) < self.min_samples_split or len(y[right_idx]) < self.min_samples_split:
                    continue
                    
                gain = self._information_gain(y, y[left_idx], y[right_idx])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return Node(value=np.bincount(y).argmax())

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=np.bincount(y).argmax())

        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature, threshold, left, right)

    def fit(self, X, y):
        X = X.values if hasattr(X, 'values') else np.array(X)
        y = y.values if hasattr(y, 'values') else np.array(y)
        self.tree = self._build_tree(X, y)

    def _predict_single(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        X = X.values if hasattr(X, 'values') else np.array(X)
        return np.array([self._predict_single(x, self.tree) for x in X])
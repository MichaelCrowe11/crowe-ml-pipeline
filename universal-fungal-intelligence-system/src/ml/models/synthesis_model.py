from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

class SynthesisModel:
    def __init__(self, data: np.ndarray, target: np.ndarray):
        self.data = data
        self.target = target
        self.model = RandomForestRegressor()

    def train(self, test_size: float = 0.2, random_state: int = 42):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)

    def predict(self, new_data: np.ndarray) -> np.ndarray:
        return self.model.predict(new_data)
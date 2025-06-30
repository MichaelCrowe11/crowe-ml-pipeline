from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model, data, target):
        self.model = model
        self.data = data
        self.target = target

    def train(self, test_size=0.2, random_state=None):
        logger.info("Starting model training...")
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.target, test_size=test_size, random_state=random_state
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        logger.info("Model training completed.")

        # Evaluate the model
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        logger.info(f"Model accuracy: {accuracy:.2f}")

        return accuracy

    def save_model(self, filename):
        joblib.dump(self.model, filename)
        logger.info(f"Model saved to {filename}")

    def load_model(self, filename):
        self.model = joblib.load(filename)
        logger.info(f"Model loaded from {filename}")
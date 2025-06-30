from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

class BioactivityModel:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.model = RandomForestClassifier()
        self.features = None
        self.target = None

    def preprocess_data(self):
        # Split the data into features and target
        self.features = self.data.drop(columns=['bioactivity'])
        self.target = self.data['bioactivity']

    def train_model(self):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        return accuracy, report

    def predict(self, new_data: pd.DataFrame):
        return self.model.predict(new_data)
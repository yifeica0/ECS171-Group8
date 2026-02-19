import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class RandomForestPipeline:
    """
    A class to handle Random Forest classification pipeline including:
    - Data preprocessing (scaling, PCA)
    - Model training
    - Prediction and evaluation
    """
    
    def __init__(self, features, test_size=0.2, random_state=42, n_estimators=100):
        """
        Initialize the pipeline
        
        Args:
            features: list of feature names to use
            test_size: proportion of data to use for testing
            random_state: random seed for reproducibility
            n_estimators: number of trees in random forest
        """
        self.features = features
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        
        # Initialize components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components="mle")
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        
        # Data splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        
        # Metrics
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.cm = None
    
    def preprocess(self, X, y, standardization=False):
        """
        Preprocess data: scaling and PCA
        
        Args:
            X: feature data
            y: target data
        """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        if standardization:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test) 
            print("Dataset is standardized.")
        
        # PCA
        self.X_train = self.pca.fit_transform(self.X_train)
        self.X_test = self.pca.transform(self.X_test)
        print(f"PCA complete. Components: {self.pca.n_components_}")
    
    def train(self):
        """Train the Random Forest model"""
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed!")
    
    def predict(self):
        """Make predictions on test set"""
        self.y_pred = self.model.predict(self.X_test)
        print("Predictions completed!")
    
    def evaluate(self):
        """Evaluate model performance"""
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred, average='weighted')
        self.recall = recall_score(self.y_test, self.y_pred, average='weighted')
        self.f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        self.cm = confusion_matrix(self.y_test, self.y_pred)
    
    def print_results(self):
        """Print evaluation metrics"""
        print("=" * 50)
        print("MODEL EVALUATION METRICS")
        print("=" * 50)
        print(f"Accuracy:  {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall:    {self.recall:.4f}")
        print(f"F1-Score:  {self.f1:.4f}")
        print("\n" + "=" * 50)
        print("CLASSIFICATION REPORT")
        print("=" * 50)
        print(classification_report(self.y_test, self.y_pred))
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues', square=True)
        plt.title('Confusion Matrix - Random Forest Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def run_pipeline(self, X, y, standardization = False):
        """
        Run the complete pipeline
        
        Args:
            X: feature data
            y: target data
        """
        self.preprocess(X, y, standardization=standardization)
        self.train()
        self.predict()
        self.evaluate()
        self.print_results()
        self.plot_confusion_matrix()


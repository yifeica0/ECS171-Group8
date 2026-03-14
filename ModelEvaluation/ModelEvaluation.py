import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the pipeline

        """
        self.y_test = None
        self.y_pred = None
        self.y_score = None
        
        # Metrics
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.cm = None
        self.fpr = None
        self.tpr = None
        self.thresholds = None
        self.auc = None

    def evaluate(self):
        """Evaluate model performance"""
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred, average='weighted')
        self.recall = recall_score(self.y_test, self.y_pred, average='weighted')
        self.f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        self.cm = confusion_matrix(self.y_test, self.y_pred)
        self.auc = roc_auc_score(self.y_test, self.y_score, multi_class='ovr', average='weighted')
     
    def print_results(self):
        if self.accuracy is None:
            print("Error: Please run .evaluate() first.")
            return
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
        print("=" * 50)
        print("ROC=AUC REPORT")
        print("=" * 50)
        print(f'AUC score: {self.auc}')
    
    def plot_graphs(self):
        if self.cm is None:
            print("Error: Please run .evaluate() first.")
            return
        plt.figure(figsize=(14, 5))
        """Plot confusion matrix"""
        plt.subplot(1, 2, 1)
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        """Plot ROC curve"""
        plt.subplot(1, 2, 2)
        classes = np.unique(self.y_test)
        y_test_bin = label_binarize(self.y_test, classes=classes)
        
        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], self.y_score[:, i])
            auc_val = roc_auc_score(y_test_bin[:, i], self.y_score[:, i])
            plt.plot(fpr, tpr, label=f'Class {class_label} (AUC = {auc_val:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC (OvR)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def run_pipeline(self, y_test, y_pred, y_score):
        """
        Run the complete pipeline
        """
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_score = y_score
        self.evaluate()
        self.print_results()
        self.plot_graphs()
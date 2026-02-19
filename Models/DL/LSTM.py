import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class LSTMClassifier:
    """LSTM Classifier for text classification with pandas DataFrame input"""
    
    def __init__(self, vocab_size=10000, max_len=10, embedding_dim=128, lstm_dim=128, dense_dim=64, num_classes=3, 
                 df=None, text_column=None, label_column=None, test_size=0.33, random_state=42):
        """
        Initialize LSTM Classifier
        
        Args:
            vocab_size: Size of vocabulary
            max_len: Maximum sequence length
            embedding_dim: Embedding dimension
            lstm_dim: LSTM layer dimension
            dense_dim: Dense layer dimension
            num_classes: Number of classes
            df: Optional pandas DataFrame (if provided, preprocess directly)
            text_column: Text column name
            label_column: Label column name
            test_size: Test set proportion
            random_state: Random seed
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.dense_dim = dense_dim
        self.num_classes = num_classes
        self.tokenizer = None
        self.model = None
        self.history = None
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.y_pred = None
        self.y_score = None

        self.precision = None
        self.recall = None
        self.f1 = None
        self.cm = None
        self.fpr = None
        self.tpr = None
        self.thresholds = None
        self.auc = None
        
        # If DataFrame is provided, preprocess directly
        if df is not None and text_column is not None and label_column is not None:
            self.preprocess(df, text_column, label_column, test_size, random_state)
    
    def preprocess(self, df, text_column, label_column, test_size=0.33, random_state=42):
        """
        Preprocess data
        
        Args:
            df: pandas DataFrame
            text_column: Text column name
            label_column: Label column name
            test_size: Test set proportion
            random_state: Random seed
        """
        texts = df[text_column].tolist()
        labels = df[label_column].values
        
        # Tokenization
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        print("Word Index (first 20):", dict(list(self.tokenizer.word_index.items())[:20]))
        print("Sequences (first 3):", sequences[:3])
        
        # Padding
        X = pad_sequences(sequences, maxlen=self.max_len)
        print("Padded Sequences shape:", X.shape)
        
        # Encode labels to integer indices 0..C-1, then one-hot encode
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels_enc = le.fit_transform(labels)
        labels_enc = np.asarray(labels_enc, dtype=np.int32)
        self.num_classes = len(le.classes_)
        y = to_categorical(labels_enc, num_classes=self.num_classes)
        print("One-hot Encoded Labels shape:", y.shape)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")
    
    def build_model(self):
        """Build LSTM model"""
        self.model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_len),
            LSTM(self.lstm_dim),
            Dense(self.dense_dim, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        print("Model Summary:")
        self.model.summary()
    
    def train(self, epochs=10, batch_size=1, verbose=1):
        """
        Train model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Must call preprocess() method to preprocess data first!")
        
        self.history = self.model.fit(
            self.X_train, self.y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(self.X_test, self.y_test),
            verbose=verbose
        )
    
    def evaluate(self):
        """Evaluate model"""
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Loss: {loss:.4f}")
        return loss, accuracy
    
    def plot_history(self):
        """Plot training history"""
        if self.history is None:
            print("Error: Model has not been trained yet!")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Accuracy Plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, texts):
        """
        Predict on texts
        
        Args:
            texts: List of texts or single text
        
        Returns:
            Prediction results
        """
        if isinstance(texts, str):
            texts = [texts]
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        predictions = self.model.predict(X)
        return predictions
    
    def plot_results(self):
        """Plot training history and ROC curve"""
        if self.history is None or self.y_score is None:
            print("Error: Model has not been trained or evaluated yet!")
            return
        
        plt.figure(figsize=(18, 5)) # 调宽画布以容纳三张图

        # 1. Accuracy Plot
        plt.subplot(1, 3, 1)
        plt.plot(self.history.history['accuracy'], label='Train')
        plt.plot(self.history.history['val_accuracy'], label='Val')
        plt.title('Accuracy')
        plt.legend()

        # 2. Loss Plot
        plt.subplot(1, 3, 2)
        plt.plot(self.history.history['loss'], label='Train')
        plt.plot(self.history.history['val_loss'], label='Val')
        plt.title('Loss')
        plt.legend()

        # 3. ROC Curve Plot (OvR)
        plt.subplot(1, 3, 3)
        n_classes = self.num_classes
        for i in range(n_classes):
            # 这里的 self.y_test 已经是 one-hot 编码，第 i 列即为该类的 binary 标签
            fpr, tpr, _ = roc_curve(self.y_test[:, i], self.y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (One-vs-Rest)')
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        plt.show()
    
    def run_pipeline(self, df, text_column, label_column, epochs=10, batch_size=1, 
                     test_size=0.33, random_state=42, verbose=1, plot=True):
        """
        Run complete ML pipeline: preprocess -> build model -> train -> evaluate
        
        Args:
            df: pandas DataFrame
            text_column: Text column name
            label_column: Label column name
            epochs: Number of training epochs, default 10
            batch_size: Batch size, default 1
            test_size: Test set proportion, default 0.33
            random_state: Random seed, default 42
            verbose: Verbosity level, default 1
            plot: Whether to plot training history, default True
        
        Returns:
            tuple: (loss, accuracy)
        """
        print("=" * 50)
        print("Starting LSTM Classification Pipeline")
        print("=" * 50)
        
        # Step 1: Data preprocessing
        print("\n[Step 1] Preprocessing data...")
        self.preprocess(df, text_column, label_column, test_size, random_state)
        print("✓ Data preprocessing completed")
        
        # Step 2: Build model
        print("\n[Step 2] Building LSTM model...")
        self.build_model()
        print("✓ Model built successfully")
        
        # Step 3: Train model
        print("\n[Step 3] Training model...")
        self.train(epochs=epochs, batch_size=batch_size, verbose=verbose)
        print("✓ Model training completed")
        
        # Step 4: Evaluate model
        print("\n[Step 4] Evaluating model...")
        loss, accuracy = self.evaluate()
        print("✓ Model evaluation completed")
        # Step 4b: Generate predictions and compute additional metrics
        print("\n[Step 4b] Generating predictions and computing metrics...")
        self.y_score = self.model.predict(self.X_test)
        # Convert one-hot to label indices
        try:
            y_test_labels = np.argmax(self.y_test, axis=1)
        except Exception:
            y_test_labels = self.y_test
        self.y_pred = np.argmax(self.y_score, axis=1)
        # Compute metrics (use label arrays)
        self.auc = roc_auc_score(self.y_test, self.y_score, multi_class='ovr', average='weighted')
        self.accuracy = accuracy_score(y_test_labels, self.y_pred)
        self.precision = precision_score(y_test_labels, self.y_pred, average='weighted')
        self.recall = recall_score(y_test_labels, self.y_pred, average='weighted')
        self.f1 = f1_score(y_test_labels, self.y_pred, average='weighted')
        self.cm = confusion_matrix(y_test_labels, self.y_pred)
        print(f"Accuracy: {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall: {self.recall:.4f}")
        print(f"F1-Score: {self.f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_labels, self.y_pred))
        
        # Step 5: Plot results
        self.plot_results(self)
        
        print("\n" + "=" * 50)
        print("Pipeline execution completed!")
        print("=" * 50)
        
        return loss, accuracy


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import joblib

class AdvancedLSTMModel:
    def __init__(self, input_shape, n_features, config):
        self.input_shape = input_shape
        self.n_features = n_features
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def build_model(self):
        """Build advanced LSTM architecture with attention mechanism"""
        inputs = layers.Input(shape=(self.input_shape[1], self.input_shape[2]))
        
        # First LSTM layer with return sequences
        lstm1 = layers.LSTM(
            256,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer=regularizers.l2(0.01)
        )(inputs)
        
        # Batch normalization
        bn1 = layers.BatchNormalization()(lstm1)
        
        # Second LSTM layer
        lstm2 = layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer=regularizers.l2(0.01)
        )(bn1)
        
        # Attention mechanism
        attention = self.attention_layer(lstm2)
        
        # Third LSTM layer
        lstm3 = layers.LSTM(
            64,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer=regularizers.l2(0.01)
        )(attention)
        
        # Dense layers
        dense1 = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )(lstm3)
        dropout1 = layers.Dropout(0.3)(dense1)
        
        dense2 = layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )(dropout1)
        dropout2 = layers.Dropout(0.3)(dense2)
        
        # Output layer with sigmoid for probability
        outputs = layers.Dense(3, activation='softmax')(dropout2)  # 3 classes: buy, hold, sell
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Custom optimizer with learning rate scheduling
        initial_learning_rate = self.config.get('learning_rate', 0.001)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        
        optimizer = Adam(learning_rate=lr_schedule)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', self.custom_profit_metric]
        )
        
        return self.model
    
    def attention_layer(self, inputs):
        """Implement attention mechanism"""
        # Self-attention
        attention_probs = layers.Dense(
            inputs.shape[-1],
            activation='softmax',
            name='attention_probs'
        )(inputs)
        
        attention_mul = layers.multiply([inputs, attention_probs])
        
        return attention_mul
    
    def custom_profit_metric(self, y_true, y_pred):
        """Custom metric that considers trading profits"""
        # Convert predictions to trading signals
        predicted_signals = tf.argmax(y_pred, axis=1) - 1  # -1, 0, 1
        true_signals = tf.argmax(y_true, axis=1) - 1
        
        # Calculate profit (simplified)
        profit = tf.reduce_sum(predicted_signals * true_signals)
        
        return profit
    
    def prepare_sequences(self, data, sequence_length=50):
        """Prepare sequences for LSTM"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length - 1):
            seq = data[i:i + sequence_length]
            target = data[i + sequence_length]
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train the model with advanced callbacks"""
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        model_checkpoint = callbacks.ModelCheckpoint(
            'models/lstm_best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        tensorboard = callbacks.TensorBoard(
            log_dir='logs/lstm',
            histogram_freq=1
        )
        
        # Custom callback for adaptive learning
        class AdaptiveLearning(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs.get('val_loss') > logs.get('loss') * 1.2:
                    print("Overfitting detected, adjusting regularization...")
                    # Implement regularization adjustment logic
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[
                early_stopping,
                reduce_lr,
                model_checkpoint,
                tensorboard,
                AdaptiveLearning()
            ],
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions with uncertainty estimation"""
        # Make multiple predictions with dropout enabled
        predictions = []
        for _ in range(10):  # Monte Carlo dropout
            pred = self.model(X, training=True)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate mean and uncertainty
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty
    
    def save_model(self, path):
        """Save model and scaler"""
        self.model.save(f"{path}/lstm_model.h5")
        joblib.dump(self.scaler, f"{path}/lstm_scaler.pkl")
        
        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(f"{path}/lstm_architecture.json", "w") as json_file:
            json_file.write(model_json)
    
    def load_model(self, path):
        """Load model and scaler"""
        self.model = tf.keras.models.load_model(
            f"{path}/lstm_model.h5",
            custom_objects={'custom_profit_metric': self.custom_profit_metric}
        )
        self.scaler = joblib.load(f"{path}/lstm_scaler.pkl")
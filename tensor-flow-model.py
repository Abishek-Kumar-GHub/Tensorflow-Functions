import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Abalone dataset
data = pd.read_csv("abalone.csv")

# Data preprocessing
# 1. Handle categorical variables (if any)
# In this case, 'Sex' is categorical. We'll use one-hot encoding.
data = pd.get_dummies(data, columns=['Sex']) 

# 2. Split data into features (X) and target (y)
X = data.drop('Rings', axis=1)
y = data['Rings']

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create TensorFlow models and perform functionalities

# 1. Simple Linear Regression
model_linear = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X_train_scaled.shape[1],))
])
model_linear.compile(optimizer='adam', loss='mean_squared_error')
model_linear.fit(X_train_scaled, y_train, epochs=100, verbose=0)

# 2. Multi-layer Perceptron (MLP)
model_mlp = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model_mlp.compile(optimizer='adam', loss='mean_squared_error')
model_mlp.fit(X_train_scaled, y_train, epochs=100, verbose=0)

# 3. Convolutional Neural Network (CNN) - Requires reshaping data
# (Not directly applicable to this dataset, but included for demonstration)
# X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
# X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
# model_cnn = tf.keras.models.Sequential([
#     tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
#     tf.keras.layers.MaxPooling1D(pool_size=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(1)
# ])
# model_cnn.compile(optimizer='adam', loss='mean_squared_error')
# model_cnn.fit(X_train_reshaped, y_train, epochs=100, verbose=0) 

# 4. Recurrent Neural Network (RNN) - Requires time series data
# (Not directly applicable to this dataset, but included for demonstration)
# # Assuming 'Time' feature exists (if applicable)
# X_train_rnn = X_train_scaled[['Time']] 
# X_test_rnn = X_test_scaled[['Time']]
# model_rnn = tf.keras.models.Sequential([
#     tf.keras.layers.SimpleRNN(units=50, return_sequences=True, input_shape=(X_train_rnn.shape[1], 1)),
#     tf.keras.layers.SimpleRNN(units=50),
#     tf.keras.layers.Dense(1)
# ])
# model_rnn.compile(optimizer='adam', loss='mean_squared_error')
# model_rnn.fit(X_train_rnn, y_train, epochs=100, verbose=0)

# 5. Long Short-Term Memory (LSTM) - Similar to RNN
# (Not directly applicable to this dataset, but included for demonstration)
# model_lstm = tf.keras.models.Sequential([
#     tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train_rnn.shape[1], 1)),
#     tf.keras.layers.LSTM(units=50),
#     tf.keras.layers.Dense(1)
# ])
# model_lstm.compile(optimizer='adam', loss='mean_squared_error')
# model_lstm.fit(X_train_rnn, y_train, epochs=100, verbose=0)

# 6. Gated Recurrent Unit (GRU) - Similar to LSTM
# (Not directly applicable to this dataset, but included for demonstration)
# model_gru = tf.keras.models.Sequential([
#     tf.keras.layers.GRU(units=50, return_sequences=True, input_shape=(X_train_rnn.shape[1], 1)),
#     tf.keras.layers.GRU(units=50),
#     tf.keras.layers.Dense(1)
# ])
# model_gru.compile(optimizer='adam', loss='mean_squared_error')
# model_gru.fit(X_train_rnn, y_train, epochs=100, verbose=0)

# 7. Dropout
model_dropout = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])
model_dropout.compile(optimizer='adam', loss='mean_squared_error')
model_dropout.fit(X_train_scaled, y_train, epochs=100, verbose=0)

# 8. Batch Normalization
model_batchnorm = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1)
])
model_batchnorm.compile(optimizer='adam', loss='mean_squared_error')
model_batchnorm.fit(X_train_scaled, y_train, epochs=100, verbose=0)

# 9. Early Stopping
# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model_early_stopping = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model_early_stopping.compile(optimizer='adam', loss='mean_squared_error')
model_early_stopping.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping], verbose=0)

# 10. Model Checkpointing
# Define model checkpointing callback
checkpoint_filepath = 'best_model.hdf5' 
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, 
                                                     save_best_only=True, 
                                                     monitor='val_loss', 
                                                     mode='min')

model_checkpointing = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model_checkpointing.compile(optimizer='adam', loss='mean_squared_error')
model_checkpointing.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, callbacks=[model_checkpoint], verbose=0)

# 11. Learning Rate Scheduling
# Define learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,  # Initial learning rate
    decay_steps=10000,         # Number of steps to decay the learning rate
    decay_rate=0.9,            # Decay rate 
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model_lr_schedule = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model_lr_schedule.compile(optimizer=optimizer, loss='mean_squared_error')
model_lr_schedule.fit(X_train_scaled, y_train, epochs=100, verbose=0)

# 12. Custom Loss Function
def custom_loss(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference)

model_custom_loss = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model_custom_loss.compile(optimizer='adam', loss=custom_loss)
model_custom_loss.fit(X_train_scaled, y_train, epochs=100, verbose=0)

# 13. L1 Regularization
model_l1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01), input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    tf.keras.layers.Dense(1)
])

model_l1.compile(optimizer='adam', loss='mean_squared_error')
model_l1.fit(X_train_scaled, y_train, epochs=100, verbose=0)

# 14. L2 Regularization
model_l2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1)
])

model_l2.compile(optimizer='adam', loss='mean_squared_error')
model_l2.fit(X_train_scaled, y_train, epochs=100, verbose=0)

# 15. Data Augmentation (Not directly applicable to this dataset, but included for demonstration)
# This would typically involve techniques like:
# - Adding noise to the input features
# - Randomly flipping or rotating images (if dealing with image data)
# - Applying random transformations to the data

# Example (hypothetical):
# def data_augmentation(X):
#     noise_factor = 0.1
#     noise = np.random.normal(loc=0.0, scale=noise_factor, size=X.shape)
#     return X + noise

# X_train_augmented = data_augmentation(X_train_scaled) 

# model_augmented = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

# model_augmented.compile(optimizer='adam', loss='mean_squared_error')
# model_augmented.fit(X_train_augmented, y_train, epochs=100, verbose=0)
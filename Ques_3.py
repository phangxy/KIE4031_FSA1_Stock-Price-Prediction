from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Import preprocessed data and variables from the Task 1 module
from Task_1 import X_test, X_val, X_train, y_test, y_val, y_train, scaler, close_prices, train_end, val_end, scaled_data

# ==========================================
# 1. Model Architecture
# ==========================================
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
]) 

model.summary()

# Compile model using Adam optimizer and MSE loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# Configure Early Stopping to prevent regime overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)

# ==========================================
# 2. Model Training
# ==========================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# 3. Evaluation & Metrics
# ==========================================
# Generate predictions on the unseen test set
predictions_scaled = model.predict(X_test)

# Inverse transform predictions and ground-truth targets to original scale (USD)
predictions = scaler.inverse_transform(predictions_scaled)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate regression metrics
mse = mean_squared_error(y_test_actual, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, predictions)
r2 = r2_score(y_test_actual, predictions)
mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# ==========================================
# 4. Visualisations
# ==========================================
# Graph 1: Training and Validation Loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# Map original dates to dataset splits, accounting for the sliding window offset
window_size = 60  # this is the window size used for creating sequences
dates = close_prices.index[window_size:]  # account for the 60-day window offset
train_dates = dates[:train_end]
val_dates   = dates[train_end:val_end]
test_dates  = dates[val_end:]

# Inverse transform the full dataset sequence for timeline visualization
full_actual = scaler.inverse_transform(scaled_data[window_size:])
train_actual = full_actual[:train_end]
val_actual   = full_actual[train_end:val_end]
test_actual  = full_actual[val_end:]

plt.figure(figsize=(16,6))

# Graph 2: Full Historical Timeline
# Plot historical context (train + validation) in grey/blue
plt.plot(train_dates, train_actual, label='Training Data', color='steelblue')
plt.plot(val_dates, val_actual, label='Validation Data', color='orange')

# Plot test actual vs predicted, clearly distinguished
plt.plot(test_dates, test_actual, label='Test Actual Price', color='green')
plt.plot(test_dates, predictions, label='Test Predicted Price', color='red', linestyle='--')

plt.title('MSFT Stock Price: Full Timeline with Test Set Predictions')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Graph 3: Zoomed-in Test Set Comparison
plt.figure(figsize=(14,6))
plt.plot(test_dates, test_actual, label='Actual Price', color='green', marker='o', markersize=3)
plt.plot(test_dates, predictions, label='Predicted Price', color='red', linestyle='--', marker='x', markersize=3)
plt.title('MSFT Stock Price: Test Set — Actual vs Predicted (Zoomed)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
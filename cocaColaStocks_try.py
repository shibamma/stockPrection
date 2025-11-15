import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


CSV_PATH = "Coca-Cola_stock_history.csv"  
DATE_COL = "Date"            
TARGET_COL = "Close"         
TEST_SIZE_DAYS = 365         
RF_SAVE_PATH = "rf_model.pkl"
LSTM_SAVE_PATH = "lstm_model.h5"
SCALER_MM_PATH = "scaler_mm.pkl"
SCALER_STD_PATH = "scaler_std.pkl"
SEED = 42


np.random.seed(SEED)

def load_data(path):
    df = pd.read_csv(path)
    # ensure date parsing
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df

def feature_engineer(df):
    """
    Adds:
    - MA_20, MA_50 (moving averages)
    - Daily_Return
    - Volatility (rolling std of returns over 20 days)
    - Year, Month, Day as possible features
    """
    df = df.copy()
    df['MA_20'] = df[TARGET_COL].rolling(window=20, min_periods=1).mean()
    df['MA_50'] = df[TARGET_COL].rolling(window=50, min_periods=1).mean()
    df['Daily_Return'] = df[TARGET_COL].pct_change().fillna(0)
    df['Volatility'] = df['Daily_Return'].rolling(window=20, min_periods=1).std().fillna(0)
    # Optional additional features
    df['Year'] = df[DATE_COL].dt.year
    df['Month'] = df[DATE_COL].dt.month
    df['Day'] = df[DATE_COL].dt.day
    return df

def clean_data(df):
    df = df.copy()
    # forward-fill numeric columns, then drop any remaining NA
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill()
    df = df.dropna().reset_index(drop=True)
    return df

def train_test_split_time(df, test_days=TEST_SIZE_DAYS):
    """
    Splits the dataframe into train/test by time. Test set is last `test_days` rows.
    """
    if test_days <= 0 or test_days >= len(df):
        raise ValueError("Invalid test_days compared to dataframe length.")
    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    return train, test

# --- RandomForest baseline (tabular features) ---
def rf_baseline(train, test, feature_cols, target_col=TARGET_COL):
    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values

    # scale features for RF (Standard)
    scaler_std = StandardScaler()
    X_train_s = scaler_std.fit_transform(X_train)
    X_test_s = scaler_std.transform(X_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_s, y_train)

    # predictions
    preds = rf.predict(X_test_s)
    mae = mean_absolute_error(y_test, preds)
    # Calculate RMSE manually if 'squared' argument is not supported
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # persist
    joblib.dump(rf, RF_SAVE_PATH)
    joblib.dump(scaler_std, SCALER_STD_PATH)
    print(f"[RF] MAE: {mae:.4f}, RMSE: {rmse:.4f} -- saved to {RF_SAVE_PATH}, {SCALER_STD_PATH}")
    return rf, preds, mae, rmse

# --- LSTM model (sequences on scaled close price + features) ---
def create_sequences(data_array, seq_len):
    X, y = [], []
    for i in range(len(data_array) - seq_len):
        X.append(data_array[i:i+seq_len])
        y.append(data_array[i+seq_len, 0])  # assuming first column is the target (Close)
    return np.array(X), np.array(y)

def lstm_pipeline(train, test, feature_cols, seq_len=60, epochs=30, batch_size=32):
    # For LSTM we use MinMaxScaler on the feature set so values are in (0,1)
    mm_scaler = MinMaxScaler()
    train_vals = train[feature_cols].values
    test_vals = test[feature_cols].values
    all_vals = np.vstack([train_vals, test_vals])
    mm_scaler.fit(all_vals)
    train_scaled = mm_scaler.transform(train_vals)
    test_scaled = mm_scaler.transform(test_vals)

    # Save MinMax scaler
    joblib.dump(mm_scaler, SCALER_MM_PATH)

    # Create sequences (we'll include all features; target is column 0 -> Close must be first in feature_cols)
    train_arr = train_scaled
    test_arr = test_scaled

    X_train, y_train = create_sequences(train_arr, seq_len)
    X_test, y_test = create_sequences(np.vstack([train_arr[-seq_len:], test_arr]), seq_len)

    # reshape for LSTM: (samples, timesteps, features)
    n_features = X_train.shape[2]
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # callbacks
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    mc = ModelCheckpoint(LSTM_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, mc],
        verbose=1
    )

    # Evaluate on X_test
    preds_scaled = model.predict(X_test).flatten()
    # preds_scaled are scaled in terms of the target's scale inside MinMaxScaler.
    # We need to inverse-transform predictions back to original scale.
    # To do that, recreate a "single-step" inverse by building arrays where the first column is predicted values
    # and the rest are taken from the corresponding last step input (works because MinMax was fit on all features).
    def inv_transform_preds(preds, X_reference):
        # X_reference: array of shape (n_samples, seq_len, n_features) -- use last time step as context
        refs = X_reference[:, -1, :].copy()  # shape (n_samples, n_features)
        invs = []
        for p, r in zip(preds, refs):
            arr = r.copy()
            arr[0] = p  # place predicted scaled target into first column
            inv = mm_scaler.inverse_transform(arr.reshape(1, -1))[0, 0]  # inverse transform and take target
            invs.append(inv)
        return np.array(invs)

    preds_inv = inv_transform_preds(preds_scaled, X_test)
    # Build y_test original-scale values
    def inv_transform_targets(y_scaled, X_reference):
        refs = X_reference[:, -1, :].copy()
        invs = []
        for y_s, r in zip(y_scaled, refs):
            arr = r.copy()
            arr[0] = y_s
            inv = mm_scaler.inverse_transform(arr.reshape(1, -1))[0, 0]
            invs.append(inv)
        return np.array(invs)

    y_test_inv = inv_transform_targets(y_test, X_test)

    mae = mean_absolute_error(y_test_inv, preds_inv)
    # Calculate RMSE manually if 'squared' argument is not supported
    rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))

    print(f"[LSTM] MAE: {mae:.4f}, RMSE: {rmse:.4f} -- model saved to {LSTM_SAVE_PATH}")
    # model already saved via ModelCheckpoint
    return model, preds_inv, y_test_inv, mae, rmse

def plot_results(dates, true_vals, preds, title="Model predictions vs True"):
    plt.figure(figsize=(12,6))
    plt.plot(dates, true_vals, label="True")
    plt.plot(dates, preds, label="Predicted")
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(TARGET_COL)
    plt.tight_layout()
    plt.show()

def main():
    #if not os.path.exists(CSV_PATH):
        #raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Please provide the Coca-Cola CSV.")
    df = load_data(CSV_PATH)
    df_fe = feature_engineer(df)
    df_clean = clean_data(df_fe)

    # choose feature list where first column is the target (Close) for LSTM inverse transform convenience
    feature_cols = [TARGET_COL, 'Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility', 'Year', 'Month', 'Day']

    # ensure all feature_cols exist
    missing = [c for c in feature_cols if c not in df_clean.columns]
    if missing:
        raise ValueError("Missing expected columns after feature engineering: " + ", ".join(missing))

    train, test = train_test_split_time(df_clean, test_days=TEST_SIZE_DAYS)

    # RandomForest baseline on tabular features
    rf_features = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility']
    rf, rf_preds, rf_mae, rf_rmse = rf_baseline(train, test, rf_features)

    # Optionally plot RF predictions for the test period
    test_dates = test[DATE_COL].iloc[len(test) - len(rf_preds):]  # ensure alignment
    try:
        plot_results(test_dates, test[TARGET_COL].values, rf_preds, title="RandomForest Predictions vs True (Test)")
    except Exception:
        pass

    # LSTM pipeline
    # For LSTM we pass feature_cols with TARGET_COL first
    lstm_model, lstm_preds, lstm_y_true, lstm_mae, lstm_rmse = lstm_pipeline(train, test, feature_cols, seq_len=60, epochs=25, batch_size=32)

    # Plot LSTM results (use last len(preds) dates from test set)
    try:
        # compute aligned dates for predictions
        # After sequence creation test predictions correspond to some subset; we used test sequences from combined tail+test,
        # so map predictions to test dataframe's dates conservatively:
        aligned_dates = test[DATE_COL].iloc[0:len(lstm_preds)].values
        plot_results(aligned_dates, lstm_y_true, lstm_preds, title="LSTM Predictions vs True (Test)")
    except Exception:
        pass

if __name__ == "__main__":
    main()
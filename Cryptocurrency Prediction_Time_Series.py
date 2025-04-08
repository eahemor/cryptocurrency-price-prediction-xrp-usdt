

# Predictive Modeling and Machine Learning:

# Which machine learning models and statistical techniques have been applied to predict cryptocurrency prices, and what level of success have these models achieved in tracking market trends and behaviours?

#%% 1. SETTING UP THE ENVIRONMENT AND IMPORTING LIBRARIES
#%% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For time series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller # Used to find out if the dataset has staionality or not
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# For machine learning models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# For ensemble methods
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import itertools
import os
#%% Set plot style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
plt.style.use('ggplot')

#Create output directory for visuals
if not os.path.exists('visuals'): # Creating a folder or directory
    os.makedirs('visuals')
    print("Created 'visuals' directory for all plots")
#%% 2. LOADING AND PREPROCESSING THE DATASET
#%%  Load Dataset
df = pd.read_csv('crypto-currency-dataset.csv')
print(df.head())
print(df.date)
#%% Convert date to datetime format
#%% Rename columns to standardized format is needed
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], format='%m-%d-%Y')
    print("Sucessfully converted date column to datetime format")
    df.set_index('date', inplace=True)
else:
    print("Warning: 'date' column not found. Using numeric index") 
#%% Sort by date for proper time series processing
if isinstance(df.index, pd.DatetimeIndex):
    df.sort_index(inplace=True)
print(f"Data sorted by date from {df.index.min()} to {df.index.max()}")
#%% Display basic information
print("\nSummary overview:")
print(df.info)
print("\nData overview:")
print(df.describe())
#%% Analysing Missing Values
print("\nChecking for missing values:")
print(df.isnull().sum())
#%% Handling missing values
if df.isnull().sum().sum() > 0:
    print("Handling missing values...")
    df.fillna(method = 'ffill', inplace=True)
#%% Checking for duplicates
duplicates = df.index.duplicated().sum()
if duplicates > 0:
    print(f"Removing {duplicates} duplicate entries...")
    df = df[~df.index.duplicates()]
print(df)    

#%% 3. EXPLOARATORY DATA ANALYSIS (EDA)
#Defining whic column to visualize
column = 'close'
#(a) Visualize the closing price over time
plt.plot(df.index, df[column],color='blue', linewidth=2)
plt.title(f'Cryptocurrency {column.capitalize()} Price Over Time')
plt.xlabel('Date')
plt.ylabel(f'{column.capitalize()} Price (USDT)')
plt.grid(True)
plt.tight_layout()
plt.savefig('visuals/price_over_time.png', dpi=300)
plt.show()
#%% b. Volume Time Series
#Visualize the trading volume
if 'Volume XRP' in df.columns:
    plt.bar(df.index, df['Volume XRP'], color='red', alpha=0.7)
    plt.title('Trading Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visuals/Trading_Volume.png', dpi=300, bbox_inches='tight')
plt.show()
#%% (c) Visualizinf Daily returns
df['daily_return'] = df[column].pct_change() * 100
plt.plot(df.index, df['daily_return'], color='green', linewidth=1.0)
plt.title('Daily Returns (%)')
plt.xlabel('Date')
plt.ylabel('Return (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig('visuals/daily_return.png', dpi=300)
plt.show()
#%% (d) Distribution of daily returns
sns.histplot(df['daily_return']
, kde=True, bins=100, color='blue')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
plt.title('Distribution of Daily Returns', fontsize=16)
plt.xlabel('Daily Returns (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig('visuals/Histogram_distribution_of_daily_return.png', dpi=300)
plt.show()
#%% (e) Visualize Volume vs Price Correlation
plt.scatter(df[column], df['Volume XRP'], alpha=0.5, color='skyblue')
plt.title('Price vs Volume Correlation')
plt.xlabel('Closing Price (USDT)')
plt.ylabel('Trading Volume')
plt.grid(True, alpha=0.3)
plt.savefig('visuals/xrp_price_volume_correlation.png', dpi=300, bbox_inches='tight')
plt.show()
#%% (f) Visualize Volatility (for rolling standard deviation)
df['volatility'] = df['daily_return'].rolling(window=30).std()
plt.plot(df.index, df['volatility'], color='purple', linewidth=2)
plt.title('30-Day Rolling Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility (std Dev of Return)')
plt.grid(True)
plt.tight_layout()
plt.savefig('visuals/volatility.png', dpi=300)
plt.show()

#%% 4. CHECKING FOR STATIONARITY
def check_stationarity(df, column='close'):
    """
    check if the time series is tationary using ADF test
    """
    print("\nChecking for stationarity...")

#Perform Augmented Dickey-Fuller test
    result = adfuller(df[column].dropna())
    
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
        
# Interprete the results
        if result[1] <0.05:
            print("The time series is stationary (reject HO)")
            return True
        else:
            print("The time series is not stationanry (fail to reject HO")
            return False
is_stationary = check_stationarity(df, 'close')
print(f"Is the time series stationary? {is_stationary}")
 
#%% Make the time series stationary

def make_stationary(df, column='close'):
    """
    Transforms the time series to make it stationary
    """
    print("\nMaking the time series stationary...")
# First difference
    df['diff_1'] = df[column].diff()
    
# Check if first difference is stationary
    is_stationary = check_stationarity(df, 'diff_1')
    
# If not stationary, take second difference option
    if not is_stationary:
        df['diff_2'] = df['diff_1'].diff()
        is_stationary = check_stationarity(df, 'diff_2')
        if is_stationary:
            print("Second difference is stationary")
            return df, 2
        else:
            print("Neither first nor second difference achieved stationarity")
            return df, 0
    else:
            print("First difference is stationary")
            return df, 1
      
# Run stationarity analysis
df, diff_order = make_stationary(df, 'close')
print(f"Achieved stationarity with {diff_order} differencing")

#%% 5. SEASONAL DECOMPOSITION
def decompose_time_series(df, column='close', period=30):
    """
    Decompose time seires into tredns, seasonal, and residual components
    """
    print("\nDecomposing time series...")

# Fill any reamining NaN values
    series = df[column].fillna(method='ffill')
# Perfoming decomposition
    try:
        decomposition = seasonal_decompose(series, model='additive', period=period)
        
# Visualising decomposition
        plt.figure(figsize=(12, 10))
        
        plt.subplot(411)
        plt.plot(decomposition.observed, linewidth=1.5)
        plt.title('Observed')
        plt.grid(True)
        
        plt.subplot(412)
        plt.plot(decomposition.trend, linewidth=1.5, color='skyblue')
        plt.title('Trend')
        plt.grid(True)
        
        plt.subplot(413)
        plt.plot(decomposition.seasonal, linewidth=1.5, color='green')
        plt.title('Seasonal')
        plt.grid(True)
        
        plt.subplot(414)
        plt.plot(decomposition.resid, linewidth=1.5, color='red')
        plt.title('Residual')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('visuals/decomposition.png', dpi=300)
        plt.show()
  
        return decomposition
    except Exception as e:
        print(f"Error in decomposition: {e}")
        return None
# Display results
decomposition = decompose_time_series(df, 'close', period=30)
#%% 6. DETERMINATION OF ARIMA PARAMETERS

def determine_arima_params(df, column='diff_1'):
    """
    Determins the parameters (p,d,q) for ARIMA model
    """
    print("\nDetermining ARIMA parameters")
    
# Fill NaN values in the differenced series
    series = df[column].dropna()
    
# Displayaing ACF and PACF
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    plot_acf(series, ax=plt.gca(), lags=40)
    plt.grid(True)
    
    plt.subplot(212)
    plot_pacf(series, ax=plt.gca(), lags=40)
    plt.grid(True)
       
    plt.tight_layout()
    plt.savefig('visuals/acf_pacf.png', dpi=300)
    plt.show()
    
    print("Examine the ACF and PACF visuals to determine p and q values")
    print("Suggested starting values based on standard practice: p=1, d=1, q=1")
    
    return 1, 1, 1 # Defaault suggestion, should be refined properly based on ACF/PACF visuals

 # Display results
p, d, q = determine_arima_params(df, 'diff_1')
print(f"Suggeted ARIMA parameters: p={p}, d={d}, q={q}")
#%% 6. Building ARIMA Model to predict price
 
def arima_forecast(df, column='close', train_size=0.8, p=1, d=1, q=1, forecast_steps=30):
    """
    Builds and evaluates an ARIMA model for forecasting
    """
    print("\nBuilding ARIMA model...")
    
    series = df[column].dropna()
     
# Split data into train and test sets    
    train_size = int(len(series) * train_size)
    train, test = series[:train_size], series[train_size:]
     
    print(f"Training set size: {len(train)}, Test set size: {len(test)}")
# Fit ARIMA model
    try:
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        print(model_fit.summary())
# in-and- Out Sample prediction
        in_sample_predictions = model_fit.predict(start=0, end=len(train)-1)
        forecast = model_fit.forecast(steps=len(test))
# Error calculations
        in_sample_rmse = np.sqrt(mean_squared_error(train, in_sample_predictions))
        out_sample_rmse = np.sqrt(mean_squared_error(test, forecast))
        out_sample_mae = mean_absolute_error(test, forecast)
# Print out results
        print(f"In-Sample RMSE: {in_sample_rmse:.3f}")
        print(f"Out-of-Sample RMSE: {out_sample_rmse:.3f}")
        print(f"Out-of-Sample MAE: {out_sample_mae:.3f}")
        
# Visualising the future trends     
        plt.figure(figsize=(12, 8))
        
        plt.plot(train.index, train, label='Training Data')
        plt.plot(test.index, test, color='skyblue', alpha=0.7, label='Test Data')
        plt.plot(test.index, forecast, color='blue', alpha=0.9, label='ARIMA Forecast')
# Addinf futre forecast
        future_index = pd.date_range(start=test.index[-1], periods=forecast_steps+1)
        future_forecast = model_fit.forecast(steps=forecast_steps+1)
        plt.plot(future_index, future_forecast, color='red', linestyle='--', label='Future Forecast')
        
        plt.title('ARIMA Model - Actual vs Forecasted Values')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('visuals/arima_forecast.png', dpi=300)
        plt.show()
        
        return model_fit, out_sample_rmse, out_sample_mae, future_forecast
    except Exception as e:
        print(f"Error in ARIMA modeling: {e}")
        return None, None, None, None
    
# Display results
arima_model, arima_rmse, arima_mae, arima_future = arima_forecast(df, 'close', 0.8, p, d, q)

#%% 7. Prepare adta to Building LSTM Model
# Prepare data for LSTM

def prepare_data_for_lstm(df, column='close', look_back=60, train_size=0.8):
    """
    Prepares time series data for LSTM model training
    """
    print("\nPreparing data for LSTM")
    
# Extracting the features
    data = df[column].values.reshape(-1, 1)
    
# Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
# Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:i+look_back, 0])
        y.append(data_scaled[i+look_back, 0])
        
# Convert to numpy arrays
    X, y = np.array(X), np.array(y)
    
# Reshape for LSTM (samples, time steps, and features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
# Splitting data into train and test sets
    train_size = int(len(X) * train_size)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

# Print dataset information
    print(f"LSTM input shape: {X.shape}")
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, scaler

X_train, X_test, y_train, y_test, scaler = prepare_data_for_lstm(df, column='close')
    
#%% 8. Building LSTM Model

def build_lstm_model(X_train, y_train, X_test, y_test, scaler, epochs=50, batch_size=32):
    """
    Builds and trains an LSTM model for time series forecasting
    """
    print("\nBuilding LSTM model...")
    
   # Define model composition
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
# Compile LSTM model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
# Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor= 'val_loss', patience=10, restore_best_weights=True)
    
# Training the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)
    
# Visualize the training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('visuals/lstm_loss.png', dpi=300)
    plt.show()
    
# Making predictions  
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
# Inverse transform to original scale 
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
# Calculate metrics   
    train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict))
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict))
    test_mae = mean_absolute_error(y_test_inv, test_predict)
    
    print(f"Training RMSE: {train_rmse:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    
    return model, test_rmse, test_mae, train_predict, test_predict
    
model, test_rmse, test_mae, train_predict, test_predict = build_lstm_model(
    X_train, y_train, X_test, y_test, scaler
) 
#%% 9. Building ensemble model

def build_ensemble_model(df, column='close', train_size=0.8, look_back=5):
    """
    Builds an ensemble model using Random Forest and Gradient Boosting
    """
    print("\nBuilding ensemble model...")
    
# Prepare features
    data = df.copy()
    
# Create lagged features
    for i in range(1, look_back + 1):
        data[f'lag_{i}'] = data[column].shift(i)
        
# Add some technical indicators   # Simple Moving Averages
    data['sma_7'] = data[column].rolling(window=7).mean()
    data['sma_30'] = data[column].rolling(window=30).mean()
    
# Exponential Moving Averages 
    data['ema_7'] = data[column].ewm(span=7).mean()
    data['ema_30'] = data[column].ewm(span=30).mean()
    
# Relative Strength Index (simplified)
    delta = data[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['rsi_14'] = 100 - (100 / (1 + rs))
    
# Defining volatility
    data['volatility_7'] = data[column].pct_change().rolling(window=7).std()
# Remove NaN values
    data.dropna(inplace=True)
    
# Prepare target variable (next day's price)
    y= data[column].shift(-1)
    data = data[:-1] # Remove the last row where we dont have the target
    y= y[:-1]
    
# Feature modeling 
    feature_columns = [col for col in data.columns if col != column and col not in ['unix', 'symbol', 'date', 'daily_return', 'volatility']]
    X = data[feature_columns]
    
# Splitting data into train and test sets
    train_size = int(len(X) * train_size)    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    print(f"Features used: {feature_columns}")
    
# Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
# Train Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    
# Ensemble (average predictions)
    ensemble_pred = (rf_pred + gb_pred) / 2 
    
# Calculate metrics
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    gb_mae = mean_absolute_error(y_test, gb_pred)
    
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
# Print out results
    print(f"Random Forest RMSE: {rf_rmse:.3f}, MAE: {rf_mae:.3f}")
    print(f"Gradient Boosting RMSE: {gb_rmse:.3f}, MAE: {gb_mae:.3f}")
    print(f"Ensemble RMSE: {ensemble_rmse:.3f}, MAE: {ensemble_mae:.3f}")
    
# Visualize results
    plt.figure(figsize=(12, 8))
    plt.plot(y_test.index, y_test.values, label='Actual', linewidth=1.5)
    plt.plot(y_test.index, ensemble_pred, label='Ensemble Prediction', color='green', alpha=0.7)
    plt.title('Ensemble Model - Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('visuals/ensemble_prediction.png', dpi=300)
    plt.show()
    
# Feature Importance 
    plt.figure(figsize=(12, 8))
    feature_importance = pd.DataFrame({
        'Feature':feature_columns,
        'Random Forest': rf_model.feature_importances_,
        'Gradient Boosting': gb_model.feature_importances_
        })
    feature_importance = feature_importance.sort_values('Random Forest', ascending=False)
    
    plt.barh(feature_importance['Feature'], feature_importance['Random Forest'], alpha=0.7, label='Random Forest')
    plt.barh(feature_importance['Feature'], feature_importance['Gradient Boosting'], alpha=0.7, label='Gradient Boosting')
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.legend()
    plt.tight_layout()
    
    return (rf_model, gb_model), ensemble_rmse, ensemble_mae, ensemble_pred
ensemble_models, ensemble_rmse, ensemble_mae, ensemble_pred = build_ensemble_model(df)
# Further Print out
print("\nEnsemble Model Results:")
print(f"RMSE: {ensemble_rmse:.3f}")
print(f"MAE: {ensemble_mae:.3f}")
#%% 10. COMPARE MODELS

def compare_models(arima_rmse, arima_mae, lstm_rmse, lstm_mae, ensemble_rmse, ensemble_mae):
    """
    Compares the performance of different models
    """
    print("\nComparing model performance...")
    
    
    models = ['ARIMA', 'LSTM', 'Ensemble']
    rmse_values = [arima_rmse, lstm_rmse, ensemble_rmse]
    mae_values = [arima_mae, lstm_mae, ensemble_mae]
    
    comparison_df = pd.DataFrame({
        'Model': models,
        'RMSE': rmse_values,
        'MAE': mae_values
        })
    
    print(comparison_df)
    
# Visualise comparison
    plt.figure(figsize=(12, 8))
    barWidth = 0.3
    r1 = np.arange(len(models))
    r2 = [x + barWidth for x in r1]
    
    plt.bar(r1, rmse_values, width=barWidth, label='RMSE', color='purple', alpha=0.7)
    plt.bar(r2, mae_values, width=barWidth, label='MAE', color='blue', alpha=0.7)
    
    plt.xlabel('Models')
    plt.ylabel('Error')
    plt.title('Model Performance Comparison')
    plt.xticks([r + barWidth/2 for r in range(len(models))], models)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('visuals/model_comaprison.png', dpi=300)
    plt.show()
    
    return comparison_df

compare_models(
    arima_rmse=0.5,
    arima_mae=0.4,
    lstm_rmse=0.3,
    lstm_mae=0.2,
    ensemble_rmse=0.25,
    ensemble_mae=0.15
    )

#%% 11. THE MAIN FUNCTION

def main():
    """
    Main function to run the entore analysis
    """
    print("Starting cryptocurrency price prediction analysis...")
    
    # Load and prepare data
    df = pd.read_csv('crypto-currency-dataset.csv')
    print(df.head())
    print(df.date)
    
  
    
    # Check for staionarity
    is_stationary = check_stationarity(df, column='close')
    
    # Make staionary if needed
    if not is_stationary:
        df, diff_order = make_stationary(df, column='close')
    
    # Determine ARIMA parameters
    p, d, q = determine_arima_params(df, column='diff_1' if 'diff_' in df.columns else 'close')
    
    # ARIMA mODEL
    arima_model, arima_rmse, arima_mae, arima_future = arima_forecast(df, 'close', 0.8, p, d, q)
    
    # LSTM model
    X_trian, X_test, y_train, y_test, scaler = prepare_data_for_lstm(df, column='close')
    lstm_model, lstm_rmse, lstm_mae, lstm_train_pred, lstm_test_pred = build_lstm_model(
        X_train, y_train, X_test, y_test, scaler
    )
    
    # Ensemble model
    ensemble_models, ensemble_rmse, ensemble_mae, ensemble_pred = build_ensemble_model(df, column='close')
     
   # Compare models
    comparison = compare_models(arima_rmse, arima_mae, lstm_rmse, lstm_mae, ensemble_rmse, ensemble_mae)
    
    print("\nAnalysis complete by Ebenezer Ahemor. Please refer to the saved visualizations and results")
    
    return df, comparison
      
# Run the main FUNCTION                                                  
if __name__ == "__main__":
    df, model_comparison = main()


    
    


       
        

     
     
            
     
     
            
    
    
    

    
        
        
    
    


    
    

        
        
        
        
        
        

    
    
    


     
     
     


     
     
        
     


    
        
        
        

    

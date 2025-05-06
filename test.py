import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("near.csv")

df_split = df.iloc[:, 0].str.split(";", expand=True)
df_split.columns = ['id', 'open', 'high', 'low', 'close', 'volume', 'marketCap', 'timestamp']


for col in ['open', 'high', 'low', 'close', 'volume', 'marketCap']:
    df_split[col] = pd.to_numeric(df_split[col], errors='coerce')
df_split['timestamp'] = pd.to_datetime(df_split['timestamp'], errors='coerce')


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def sma(series, window):
    return series.rolling(window=window).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# MACD
macd_line = ema(df_split['close'], span=12) - ema(df_split['close'], span=26)
signal_line = ema(macd_line, span=9)
df_split['macd_diff'] = macd_line - signal_line

# EMA, SMA, RSI
df_split['ema20'] = ema(df_split['close'], span=20)
df_split['sma20'] = sma(df_split['close'], window=20)
df_split['rsi'] = rsi(df_split['close'], period=14)

# OBV
obv = [0]
for i in range(1, len(df_split)):
    if df_split['close'].iloc[i] > df_split['close'].iloc[i - 1]:
        obv.append(obv[-1] + df_split['volume'].iloc[i])
    elif df_split['close'].iloc[i] < df_split['close'].iloc[i - 1]:
        obv.append(obv[-1] - df_split['volume'].iloc[i])
    else:
        obv.append(obv[-1])
df_split['obv'] = obv

sma20 = sma(df_split['close'], 20)
std20 = df_split['close'].rolling(window=20).std()
df_split['bollinger_width'] = 2 * std20

df_split['month'] = df_split['timestamp'].dt.month
df_split['week'] = df_split['timestamp'].dt.isocalendar().week
df_split['dayofweek'] = df_split['timestamp'].dt.dayofweek

df_split['target_close'] = df_split['close'].shift(-1)


df_split.dropna(inplace=True)


last_row = df_split.tail(1)
df_split = df_split.iloc[:-1]


X = df_split[['open', 'high', 'low', 'volume', 'marketCap',
              'rsi', 'ema20', 'macd_diff', 'obv', 'bollinger_width',
              'month', 'week', 'dayofweek']]
y = df_split['target_close']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


params = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
grid = GridSearchCV(estimator=xgb, param_grid=params, cv=3, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

X_last = last_row[['open', 'high', 'low', 'volume', 'marketCap',
                   'rsi', 'ema20', 'macd_diff', 'obv', 'bollinger_width',
                   'month', 'week', 'dayofweek']]
real_next_close = last_row['target_close'].values[0]
prediction = best_model.predict(X_last)

print(f"Прогнозирано: {prediction[0]:.4f}")

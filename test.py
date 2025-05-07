import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Зареждане на CSV файла
df = pd.read_csv("near.csv")

# 2. Разделяне на колоните
df_split = df.iloc[:, 0].str.split(";", expand=True)
df_split.columns = ['id', 'open', 'high', 'low', 'close', 'volume', 'marketCap', 'timestamp']

# 3. Преобразуване в числови стойности
for col in ['open', 'high', 'low', 'close', 'volume', 'marketCap']:
    df_split[col] = pd.to_numeric(df_split[col], errors='coerce')

# 4. Индикатори

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

# OBV (On-Balance Volume)
obv = [0]
for i in range(1, len(df_split)):
    if df_split['close'].iloc[i] > df_split['close'].iloc[i - 1]:
        obv.append(obv[-1] + df_split['volume'].iloc[i])
    elif df_split['close'].iloc[i] < df_split['close'].iloc[i - 1]:
        obv.append(obv[-1] - df_split['volume'].iloc[i])
    else:
        obv.append(obv[-1])
df_split['obv'] = obv

# Bollinger Bands Width
sma20 = sma(df_split['close'], 20)
std20 = df_split['close'].rolling(window=20).std()
upper_band = sma20 + 2 * std20
lower_band = sma20 - 2 * std20
df_split['bollinger_width'] = upper_band - lower_band

# 5. Целева стойност: close за следващата седмица
df_split['target_close'] = df_split['close'].shift(-1)

# 6. Премахване на NaN стойности
df_split.dropna(inplace=True)

# 7. Подготовка на входове и изходи
X = df_split[['open', 'high', 'low', 'volume', 'marketCap',
              'rsi', 'ema20', 'sma20', 'macd_diff', 'obv', 'bollinger_width']]
y = df_split['target_close']

# 8. Разделяне на тренировъчни и тестови данни
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Обучение
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

# 10. Оценка
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

# 11. Прогноза за следващата седмица
last_row = X.tail(1)
prediction = model.predict(last_row)

print(f"Прогнозирано: {prediction[0]:.4f}")

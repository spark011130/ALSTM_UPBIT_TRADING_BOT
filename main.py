import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import os
import pyupbit
import schedule
import time
from datetime import datetime
import pytz
from tqdm import tqdm
from inputs.data_loader import load_config, get_upbit_keys
from trading.indicators import calculate_rsi, calculate_macd, \
    calculate_bollinger_bands, calculate_atr, calculate_stochastic_oscillator
from models.alstm import ALSTM
from utils.logger import get_logger

# Load config and keys
CONFIG = load_config()
upbit = get_upbit_keys()
logger = get_logger("trading_bot", "outputs/trading.log")

# Global configs
SYMBOL = CONFIG["symbol"]
TICKER = CONFIG["ticker"]
INTERVAL = CONFIG["interval"]
INTERVAL_IN_MIN = CONFIG["interval_in_min"]
SEQ_LENGTH = CONFIG["seq_length"]
BATCH_SIZE = CONFIG["batch_size"]
LR = CONFIG["lr"]
EPOCHS = CONFIG["epochs"]
SPLIT_RATIO = CONFIG["split_ratio"]
OVERLAP_GAP = CONFIG["overlap_gap"]
THRESHOLD = CONFIG["threshold"]
RISK_FREE_RATE = CONFIG["risk_free_rate"]
TRANSACTION_FEE = CONFIG["transaction_fee"]
MODEL_PATH = CONFIG["model_path"]
LOSS_PLOT_PATH = CONFIG["loss_plot_path"]
RESULTS_PLOT_PATH = CONFIG["results_plot_path"]
BACKTEST = CONFIG["backtest"]
CRYPTO_DATA_PATH = f"inputs/{SYMBOL}_prices_{INTERVAL}.csv"


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}.")
np.random.seed(0)
logger.info(f"✅ Computation device set to {device}. Successfully connected to Upbit account.")

### DATA PROCESSING

# process_data: for prediction, load_data: for training
# Load stock data
def process_data(df):    
    df['RSI'] = calculate_rsi(df)
    df['MACD'], df['Signal'] = calculate_macd(df)
    df['Middle_Band'], df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(df)
    df['ATR'] = calculate_atr(df)
    df['%K'], df['%D'] = calculate_stochastic_oscillator(df)
    
    scalers = dict()
    columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'ATR', '%K', '%D']
    for col in columns:
        scaler = MinMaxScaler(feature_range=(-1, 1) if col in ['MACD', 'Signal'] else (0, 1))
        df[[col]] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
    df.dropna(inplace=True)
    return df, scalers

def load_data(df, split_ratio):
    # calculating each of the indicators
    df['RSI'] = calculate_rsi(df)
    df['MACD'], df['Signal'] = calculate_macd(df)
    df['Middle_Band'], df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(df)
    df['ATR'] = calculate_atr(df)
    df['%K'], df['%D'] = calculate_stochastic_oscillator(df)
    
    # regularization (scaler for the indicators, differencing for the stock prices)
    train_size = int(len(df) * split_ratio)

    scalers = dict()
    columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'ATR', '%K', '%D']
    for col in columns:
        scaler = MinMaxScaler(feature_range=(-1, 1) if col in ['MACD', 'Signal'] else (0, 1))
        df.loc[:train_size, col] = scaler.fit_transform(df.loc[:train_size, [col]])
        scalers[f'{col}'] = scaler
        df.loc[train_size+1:, col] = scaler.transform(df.loc[train_size+1:, [col]])
    df.dropna(inplace=True)

    return df, scalers

class StockDataset(Dataset):
    def __init__(self, data, seq_length=10):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.seq_length, :]  # Input: Today-seq_length ~ today stock price + strategies + volume
        y = self.data[idx + self.seq_length, 3]  # Output: Next day stock price
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Train the model
def train_model(model, train_loader, epochs=100, lr=0.001, resume_from_checkpoint=None):
    model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    progress_bar = tqdm(range(epochs), desc="Training", ncols=100)

    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        checkpoint = torch.load(resume_from_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"✅ Resuming training from checkpoint at epoch {start_epoch}")
        
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in progress_bar:        
        total_loss=0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred, _ = model(X_batch)
            y_pred = y_pred.squeeze()

            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        progress_bar.set_postfix(loss=avg_loss)

        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ Unstable loss detected.")
            logger.warning("⚠️ Unstable loss detected.")
            continue

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, f"outputs/checkpoint_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), MODEL_PATH)
    print("✅Model saved.")
    logger.info("✅Model saved.")
    
    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_PLOT_PATH)
    print(f"📉 Loss curve saved to {LOSS_PLOT_PATH}.")
    logger.info(f"📉 Loss curve saved to {LOSS_PLOT_PATH}.")
    plt.close()


def evaluate_trading_strategy(y_test, y_pred, interval_in_min, threshold=0.01, risk_free_rate=0.02):
    N = 527040 / interval_in_min
    # Calculate daily returns
    returns = np.diff(y_test) / y_test[:-1]  # Actual returns
    pred_returns = np.diff(y_pred) / y_pred[:-1]  # Predicted returns

    df = pd.DataFrame(returns)
    print("Real Returns Description")
    print(df.describe())
    print("Predicted Returns Description")
    df = pd.DataFrame(pred_returns)
    print(df.describe())

    # Trend-following strategy: If y_pred > y_pred, go long (buy)
    trend_strategy = np.where(pred_returns > 0, 1, 0)  # Buy (1), no Short.
    print(f"trend strategy is traded {np.sum(trend_strategy)} times in {len(y_pred)} intervals.")
    trend_returns = trend_strategy * (returns-TRANSACTION_FEE)  # Returns based on the trading strategy
    
    # Event-driven strategy: If y_pred exceeds the threshold, take the buy position.
    event_strategy = np.where(pred_returns > threshold, 1, 0)
    print(f"event strategy is traded {np.sum(event_strategy)} times in {len(y_pred)} intervals.")
    event_returns = event_strategy * (returns-TRANSACTION_FEE)  # Returns based on the event-driven strategy

    # Calculate cumulative returns (Keep as arrays)
    cumulative_return_trend = np.cumprod(1  + trend_returns) - 1  # Trend-following strategy
    cumulative_return_event = np.cumprod(1 + event_returns) - 1  # Event-driven strategy

    # Annualized Return (AR) - Convert to percentage
    ar_trend = float(np.mean(trend_returns) * N * 100)  # Convert to %
    ar_event = float(np.mean(event_returns) * N * 100)  # Convert to %

    # Annualized Volatility (AV) - Convert to percentage
    av_trend = float(np.std(trend_returns) * np.sqrt(N) * 100)  # Convert to %
    av_event = float(np.std(event_returns) * np.sqrt(N) * 100)  # Convert to %

    # Sharpe Ratio (SR)
    sr_trend = float((ar_trend - risk_free_rate * 100) / av_trend) if av_trend != 0 else np.nan
    sr_event = float((ar_event - risk_free_rate * 100) / av_event) if av_event != 0 else np.nan

    # Maximum Drawdown (MDD) - Convert to percentage
    def calculate_mdd(cumulative_returns):
        if len(cumulative_returns) == 0:  # Check if array is empty
            return 0.0
        peak = np.maximum.accumulate(cumulative_returns)  # Track the highest value reached
        drawdown = (cumulative_returns - peak) / peak  # Drawdown calculation
        return abs(float(np.min(drawdown) * 100))  # Convert to %

    mdd_trend = calculate_mdd(cumulative_return_trend)
    mdd_event = calculate_mdd(cumulative_return_event)

    # Output results
    results = {
        "Trend Following": {
            "Cumulative Return": float(cumulative_return_trend[-1] * 100),  # Convert to %
            "AR (%)": ar_trend,
            "AV (%)": av_trend,
            "SR": sr_trend,
            "MDD (%)": mdd_trend
        },
        "Event-Driven Trading": {
            "Cumulative Return": float(cumulative_return_event[-1] * 100),  # Convert to %
            "AR (%)": ar_event,
            "AV (%)": av_event,
            "SR": sr_event,
            "MDD (%)": mdd_event
        }
    }
    
    return results

# Evaluate model and calculate performance metrics
def evaluate_model(model, test_loader, scalers):
    model.to(device)
    model.eval()
    predictions, actuals = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred, _ = model(X_batch)
            y_pred = y_pred.squeeze().cpu().numpy()

            # 🛠 Fix for scalar output
            if y_pred.ndim == 0:
                predictions.append(y_pred.item())
            else:
                predictions.extend(y_pred)

            actuals.extend(y_batch.numpy())

    # Convert to numpy arrays
    actuals = np.array(actuals).reshape(-1, 1)
    predictions = np.array(predictions).reshape(-1, 1)

    # Inverse transform only 'Close' price
    min_close = scalers['Close'].data_min_  # 'Close' price min
    max_close = scalers['Close'].data_max_  # 'Close' price max
    actuals_original = actuals * (max_close - min_close) + min_close
    predictions_original = predictions * (max_close - min_close) + min_close

    y_test = actuals_original.flatten()
    y_pred = predictions_original.flatten()

    # Calculate returns for IC
    returns_actual = np.diff(y_test) / y_test[:-1]
    returns_pred = np.diff(y_pred) / y_pred[:-1]

    # Performance Metrics
    mse = np.mean((y_test - y_pred)**2)
    mae = np.mean(np.abs(y_test - y_pred))
    ic, _ = spearmanr(returns_pred, returns_actual)

    # ICIR (Information Coefficient Information Ratio)
    ic_series = pd.Series(returns_actual).rolling(window=10).apply(
        lambda x: spearmanr(x, pd.Series(returns_pred).iloc[x.index])[0], raw=False
    )
    icir = ic_series.mean() / ic_series.std() if ic_series.std() != 0 else np.nan

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual (Original Scale)')
    plt.plot(y_pred, label='Predicted (Original Scale)')
    plt.legend()
    plt.title('LSTM Stock Prediction')
    plt.savefig(RESULTS_PLOT_PATH, dpi=300)
    plt.close()

    metrices = [mse, mae, ic, icir]
    return y_pred, y_test, metrices

def print_and_save_evaluation_results(results, metrics, filename="outputs/ALSTM_trading_strategy_evaluation.csv"):
    # Convert results dictionary to DataFrame
    df = pd.DataFrame(results).T  # Transpose for a structured tabular format

    # Round numerical values for better readability
    df = df.round(2)

    # Append additional evaluation metrics to the DataFrame
    additional_metrics = pd.DataFrame({
        "MSE": [metrics[0]],
        "MAE": [metrics[1]],
        "IC": [metrics[2]],
        "ICIR": [metrics[3]]
    }, index=["Evaluation Metrics"])

    # Concatenate strategy results and evaluation metrics
    df = pd.concat([df, additional_metrics])

    # Print formatted DataFrame
    print("\n📊 Trading Strategy Evaluation Results 📊\n")
    print(df.to_string())
    print("\n🔹 Cumulative Return: Total return (%)")
    print("🔹 AR (Annualized Return): Yearly return (%)")
    print("🔹 AV (Annualized Volatility): Yearly volatility (%)")
    print("🔹 SR (Sharpe Ratio): Risk-adjusted return (Higher is better)")
    print("🔹 MDD (Maximum Drawdown): Maximum loss from peak (%)")
    print("\n📌 Additional Evaluation Metrics:")
    print("🔹 MSE (Mean Squared Error): Measures prediction error (Lower is better)")
    print("🔹 MAE (Mean Absolute Error): Measures average absolute error (Lower is better)")
    print("🔹 IC (Information Coefficient): Measures predictive power (Higher is better)")
    print("🔹 ICIR (Information Coefficient Information Ratio): Measures consistency of IC")

    # Save results to a CSV file
    df.to_csv(filename, index=True)
    print(f"\n✅ Results saved successfully to '{filename}'.")
    logger.info(f"\n✅ Results saved successfully to '{filename}'.")


    return df

def main(train_mode):
    if train_mode == "backtest":
        if not os.path.exists(CRYPTO_DATA_PATH):
            df = pyupbit.get_ohlcv_from(ticker=TICKER, interval=INTERVAL, fromDatetime="2020-01-01")
            df.reset_index(inplace=True)
            df.columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume", "Value"] 
            df = df.drop(columns=["Value"])
            df.to_csv(CRYPTO_DATA_PATH, index=False, encoding="utf-8-sig", float_format="%.8f")
        else:
            df = pd.read_csv(CRYPTO_DATA_PATH)
        print(f"\n✅ Dataframe initiated successfully to '{CRYPTO_DATA_PATH}'.")
        logger.info(f"\n✅ Dataframe initiated successfully to '{CRYPTO_DATA_PATH}'.")

        ## TEST THE STRATEGIES
        df, scalers = load_data(df, SPLIT_RATIO)
        data_np = df.iloc[:, 1:].values
        split = int(len(data_np) * SPLIT_RATIO)
        # Adding 90, to prevent overlapping
        train_data, test_data = data_np[:split], data_np[split+90:]


        train_dataset = StockDataset(train_data, SEQ_LENGTH)
        test_dataset = StockDataset(test_data, SEQ_LENGTH)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        input_dim = data_np.shape[1]
        model = ALSTM(input_dim=input_dim).to(device)
        
        train_model(model=model, train_loader=train_loader, epochs=EPOCHS, lr=LR)
        
        model.load_state_dict(torch.load(MODEL_PATH))
        y_pred, y_test, metrices = evaluate_model(model, test_loader, scalers)

        results = evaluate_trading_strategy(y_test, y_pred, INTERVAL_IN_MIN, THRESHOLD, RISK_FREE_RATE)
        print_and_save_evaluation_results(results, metrices)

    elif train_mode == "train":
        df = pd.read_csv(CRYPTO_DATA_PATH)
        last_time = df.iloc[-1, 0]
        df_new = pyupbit.get_ohlcv_from(ticker=TICKER, interval=INTERVAL, fromDatetime=last_time) # this step is meant to make the scraping step faster using previous data.
        df_new.reset_index(inplace=True)
        df_new.columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume", "Value"]
        df_new = df_new.drop(columns=["Value"])
        if not df_new.iloc[1:, :].empty:
            df = pd.concat([df, df_new.iloc[1:, :]], ignore_index=True)
        df.to_csv(CRYPTO_DATA_PATH, index=False, encoding="utf-8-sig", float_format="%.8f")
        N = df_new.shape[0] - 1
        data_np = df.iloc[:, 1:].values
        train_data = data_np[:-11]
        train_dataset = StockDataset(train_data, SEQ_LENGTH)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

        input_dim = data_np.shape[1]
        model = ALSTM(input_dim=input_dim).to(device)

        train_model(model=model, train_loader=train_loader, epochs=EPOCHS, lr=LR)

    ## 여기 부분을 한시간마다 돌려서, 매수나 매도를 결정하면 됨.
    elif train_mode == "invest":    
        df = pd.read_csv(CRYPTO_DATA_PATH)
        last_time = df.iloc[-1, 0]

        ## EVERY RUN, CHECK THE UPDATES
        df_new = pyupbit.get_ohlcv_from(ticker=TICKER, interval=INTERVAL, fromDatetime=last_time)
        df_new.reset_index(inplace=True)
        df_new.columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume", "Value"] 
        df_new = df_new.drop(columns=["Value"])
        if not df_new.iloc[1:, :].empty:
            df = pd.concat([df, df_new.iloc[1:, :]], ignore_index=True)
        df.to_csv(CRYPTO_DATA_PATH, index=False, encoding="utf-8-sig", float_format="%.8f")        
        N = df_new.shape[0] - 1
        df, scalers = process_data(df)
        if N >= 1:
            print(f"\n✅ Dataframe updated successfully to '{CRYPTO_DATA_PATH}'.")
            logger.info(f"\n✅ Dataframe updated successfully to '{CRYPTO_DATA_PATH}'.")
        else:
            logger.warning("⚠️ Dataframe has nothing to update.")

        data_np = df.iloc[:, 1:].values
        test_data = data_np[-11:]

        input_dim = data_np.shape[1]
        model = ALSTM(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(MODEL_PATH))
        
        X_input_past = torch.tensor(test_data[-11:-1], dtype=torch.float32).unsqueeze(0).to(device)
        X_input = torch.tensor(test_data[-10:], dtype=torch.float32).unsqueeze(0).to(device)

        ## PREDiCTION PHASE
        model.eval()
        with torch.no_grad():
            output, _ = model(X_input)
            prediction = output.squeeze().cpu().numpy()
            output_past, _ = model(X_input_past)
            prediction_past = output_past.squeeze().cpu().numpy()    

        ## UPBIT 매수 매도
        wallet = upbit.get_balances()
        krw_balance = [item['balance'] for item in wallet if item['currency'] == 'KRW']
        if krw_balance:
            krw_balance = float(krw_balance[0])
        crypto_balance = [item['balance'] for item in wallet if item['currency'] == f'{TICKER.replace("KRW-", "")}']
        isOrdered = False
        if crypto_balance:
            crypto_balance = float(crypto_balance[0])
            print("매수 중인 물량이 있습니다.")
            isOrdered = True
        print(f"예측 가격 (과거): {prediction_past}, 예측 가격 (현재): {prediction}, 종가: {df.iloc[-1, 4]}")
        logger.info(f"Predicted Price (Previous): {prediction_past}, Predicted Price (Current): {prediction}, Closing Price: {df.iloc[-1, 4]}")
        if prediction > prediction_past and (prediction / prediction_past) > (1+THRESHOLD): #  prediction > df.iloc[-1, 4]
            # 매수
            if isOrdered:
                with open("outputs/trade_log.txt", "a") as f:
                    f.write(f"{datetime.now()} - 매수 포지션 유지")
                print("매수 포지션을 유지합니다.")
                logger.info("✅ Retaining current buy position.")
            else:
                with open("outputs/trade_log.txt", "a") as f:
                    f.write(f"{datetime.now()} - 시장가 매수")
                upbit.buy_market_order(TICKER, krw_balance*0.95)
                print("매수 완료!")
                logger.info("✅ Buy order executed successfully.")
        else:
            # 매도
            if isOrdered:
                with open("outputs/trade_log.txt", "a") as f:
                    f.write(f"{datetime.now()} - 시장가 매도")
                print(upbit.sell_market_order(TICKER, crypto_balance)) # type: ignore
                print("매도 완료!")
                logger.info("✅ Sell order executed successfully.")

            else:
                with open("outputs/trade_log.txt", "a") as f:
                    f.write(f"{datetime.now()} - 매수 건너뛰기")
                print("매수하지 않습니다.")
                logger.info("✅ Buy order not executed.")
def job():
    now = datetime.now(pytz.timezone('Asia/Seoul'))
    if now.minute == 0:
        print(f"⏳ Running script at {now.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("⏳ Running trading job.")
        try:
            main(train_mode="invest")
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"❌ Error: {e}")

def job2():
    now = datetime.now(pytz.timezone('Asia/Seoul'))
    if now.minute == 30:
        print(f"⏳ Retraining the model at {now.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("⏳ Retraining the model.")
        try:
            file_path = f"inputs/{SYMBOL}_prices_minute60.csv"
            if os.path.exists(file_path):
                os.remove(file_path)
            print("✅ File successfully deleted.")
            logger.info("✅ File successfully deleted.")
            main(train_mode="train")
            print("✅ Model successfully trained.")
            logger.info("✅ Model successfully trained.")

        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"❌ Error: {e}")

if not BACKTEST:
    schedule.every().minute.at(":00").do(job)
    schedule.every().minute.at(":30").do(job2)
    while True:
        schedule.run_pending()
        time.sleep(1)

elif BACKTEST:
    main(train_mode="backtest")
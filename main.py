import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def get_stock_data(ticker, start_date, end_date, period='1d', pre_days=300):
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        extended_start_dt = start_dt - timedelta(days=pre_days)
        extended_start = extended_start_dt.strftime("%Y-%m-%d")

        extended_full_data = yf.download(ticker, start=extended_start, end=end_date, interval=period)
        extended_full_data.columns = extended_full_data.columns.droplevel('Ticker')
        close_prices = extended_full_data.loc[start_date:end_date]['Close'].squeeze()
        extended_prices = extended_full_data.loc[extended_start:end_date]['Close'].squeeze()
        volume = extended_full_data.loc[start_date:end_date]['Volume'].squeeze()
        extended_volume = extended_full_data.loc[extended_start:end_date]['Volume'].squeeze()
        full_data = extended_full_data.loc[start_date:end_date]

        if extended_full_data.empty:
            raise ValueError("未獲取到數據，請檢查股票代碼和日期範圍")

        return full_data, extended_full_data, close_prices, extended_prices, volume, extended_volume

    except Exception as e:
        print(f"數據獲取失敗: {str(e)}")
        return None, None, None, None, None, None


def VIX(start_date, end_date):
    VIX = yf.download("^VIX", start_date, end_date)
    VIX = VIX['Close']
    return VIX


def calculate_correlation_and_leverage(tickers, start_date, end_date, window=30, benchmark='QQQ'):
    """
    計算多個標的與基準標的（預設為 QQQ）的相關係數和槓桿係數。

    參數：
    - tickers: 股票代碼列表（如 ['QQQ', 'TQQQ', 'SQQQ', 'NVDA', 'TSLA', 'AAPL']）
    - start_date: 開始日期（如 '2023-01-01'）
    - end_date: 結束日期（如 '2025-08-06'）
    - window: 滾動相關係數的窗口大小（預設 60 天）
    - benchmark: 基準標的（預設為 QQQ）

    輸出：
    - 相關係數矩陣
    - 每個標的與基準標的槓桿係數（β）和 R²
    - 滾動相關係數圖表
    """
    try:
        # 獲取所有標的的收盤價
        data = yf.download(tickers, start=start_date, end=end_date, interval='1d')['Close']

        # 計算日收益
        returns = data.pct_change().dropna()

        # 1. 計算皮爾森相關係數
        correlation_matrix = returns.corr()
        print("\n相關係數矩陣：")
        print(correlation_matrix)

        # 2. 計算槓桿係數（回歸分析）
        print(f"\n槓桿係數（相對於 {benchmark}）：")
        leverage_results = {}
        for ticker in tickers:
            if ticker != benchmark:
                X = returns[benchmark].values.reshape(-1, 1)
                y = returns[ticker].values
                model = LinearRegression()
                model.fit(X, y)
                beta = model.coef_[0]
                r2 = model.score(X, y)
                leverage_results[ticker] = {'Beta': beta, 'R²': r2}
                print(f"{ticker} vs {benchmark}: Beta = {beta:.2f}, R² = {r2:.2f}")

        # 3. 計算滾動相關係數
        rolling_corrs = {}
        for ticker in tickers:
            if ticker != benchmark:
                rolling_corr = returns[benchmark].rolling(window=window).corr(returns[ticker])
                rolling_corrs[ticker] = rolling_corr

        # 4. 可視化滾動相關係數
        plt.figure(figsize=(12, 6))
        for ticker, corr in rolling_corrs.items():
            plt.plot(corr.index, corr, label=f'{ticker} vs {benchmark}')
        plt.title(f'Rolling Correlation with {benchmark} ({window}-day Window)')
        plt.xlabel('Date')
        plt.ylabel('Correlation Coefficient')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 5. 獲取 VIX 數據（作為市場波動性參考）
        vix_data = VIX(start_date, end_date)
        print("\nVIX 數據摘要：")
        print(vix_data.describe())

        return correlation_matrix, leverage_results, rolling_corrs, vix_data

    except Exception as e:
        print(f"計算失敗: {str(e)}")
        return None, None, None, None


# 示例用法
if __name__ == "__main__":
    tickers = ['QQQ','NVDA']
    start_date = '2023-01-01'
    end_date = '2025-9-8'

    # 計算相關係數和槓桿係數
    corr_matrix, leverage_results, rolling_corrs, vix_data = calculate_correlation_and_leverage(
        tickers, start_date, end_date, window=30
    )
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def download_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty or data['Adj Close'].isna().all():
            return None
        return data['Adj Close']
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None

def cluster_stocks_and_plot(tickers, start_date, end_date, num_clusters):
    data = pd.DataFrame()
    for ticker in tickers:
        ticker_data = download_stock_data(ticker, start_date, end_date)
        if ticker_data is not None:
            data[ticker] = ticker_data

    data_filled = data.ffill().bfill()

    daily_returns = data_filled.pct_change()
    correlation_matrix = daily_returns.corr()

    distance_matrix = 1 - np.abs(correlation_matrix)

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(distance_matrix)

    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(reduced_data)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', marker='o')
    valid_tickers = data_filled.columns
    for i, ticker in enumerate(valid_tickers):
        plt.text(reduced_data[i, 0], reduced_data[i, 1], ticker)

    plt.title('Figure 5.2.1 Stock Clusters based on Correlations')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.show()
    stock_clusters = {valid_tickers[i]: clusters[i] for i in range(len(valid_tickers))}

    cluster_dict = {}
    for stock, cluster in stock_clusters.items():
        if cluster not in cluster_dict:
            cluster_dict[cluster] = [stock]
        else:
            cluster_dict[cluster].append(stock)

    return cluster_dict

tickers = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "NVDA", "JPM", "JNJ",
    "V", "PG", "UNH", "DIS", "HD", "MA", "BAC", "PYPL", "CMCSA", "XOM", "LLY",
    "TSM", "AVGO", "NVO", "WMT", "ASML", "MRK", "COST", "TM", "ABBV", "ORCL",
    "CVX", "AMD", "CRM", "KO", "NFLX", "ADBE", "ACN", "PEP", "TMO", "LIN", "MCD",
    "NVS", "ABT", "CSCO", "DHR", "INTC", "WFC", "BABA", "SAP"
]
start_date = '2012-01-01'
end_date = '2015-12-31'
num_clusters = 3
cluster_dict = cluster_stocks_and_plot(tickers, start_date, end_date, num_clusters)

data = yf.download(['V', 'MA'], start='2016-01-01', end='2020-01-01')

data = data['Adj Close']

data['V_return'] = data['V'].pct_change()
data['MA_return'] = data['MA'].pct_change()

V_mean_return = data['V_return'].mean()
MA_mean_return = data['MA_return'].mean()

correlation = data['V_return'].corr(data['MA_return'])

print(f"Mean return for V from Jan 2016 to Jan 2020: {V_mean_return}")
print(f"Mean return for MA from Jan 2016 to Jan 2020: {MA_mean_return}")
print(f"Correlation between V and MA returns: {correlation}")

plt.figure(figsize=(14,7))
plt.plot(data.index, data['V_return'], label='Visa')
plt.plot(data.index, data['MA_return'], label='Mastercard')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('Figure 5.2.4 Visa and Mastercard Returns from Jan 2016 to Jan 2020')
plt.legend()
plt.grid(True)
plt.show()

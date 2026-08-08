import os
import csv
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "01_DATA")
MARKET_DIR = os.path.join(DATA_DIR, "market")

os.makedirs(MARKET_DIR, exist_ok=True)

# Scanner 2에서 사용할 기본 Universe
DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "GOOG", "TSLA", "AMD", "AVGO",
    "NFLX", "PLTR", "MSTR", "COIN", "HOOD",
    "CRWD", "ORCL", "CRM", "ADBE", "NOW",
    "INTC", "MU", "SNDK", "LRCX", "AMAT",
    "QCOM", "MRVL", "ARM", "SMCI",
    "SOXL", "SOXX", "SMH",
    "XLF", "XLE", "XLK", "XLV",
    "XLI", "XLY", "XLP", "XLU",
    "GLD", "SLV", "TLT", "HYG",
    "VST", "CEG", "NRG",
    "RKLB", "LUNR", "SNOW", "SOFI",
    "UBER", "ABNB", "SHOP", "CRCL"
]


def create_universe():
    today = datetime.now().strftime("%Y-%m-%d")

    filename = f"AUTO_UNIVERSE_{today}.csv"
    output_file = os.path.join(MARKET_DIR, filename)

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(["Ticker"])

        for ticker in DEFAULT_TICKERS:
            writer.writerow([ticker])

    print("=" * 60)
    print("SCANNER 2 AUTO UNIVERSE")
    print("=" * 60)
    print(f"Created: {output_file}")
    print(f"Ticker count: {len(DEFAULT_TICKERS)}")

    return output_file


if __name__ == "__main__":
    create_universe()

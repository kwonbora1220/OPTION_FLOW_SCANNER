import sys
import pandas as pd
import yfinance as yf


MIN_DTE = 0
MAX_DTE = 180


def get_price(ticker):

    stock = yf.Ticker(ticker)

    hist = stock.history(
        period="5d",
        interval="1d",
        auto_adjust=False
    )

    if hist.empty:
        return None

    return float(
        hist["Close"].dropna().iloc[-1]
    )


def get_options(ticker):

    stock = yf.Ticker(ticker)

    expirations = stock.options

    if not expirations:
        return pd.DataFrame()

    today = pd.Timestamp.now().normalize()

    rows = []

    for expiration in expirations:

        exp = pd.Timestamp(expiration)

        dte = (
            exp - today
        ).days

        if not (
            MIN_DTE
            <= dte
            <= MAX_DTE
        ):
            continue

        try:

            chain = stock.option_chain(
                expiration
            )

            calls = chain.calls.copy()
            puts = chain.puts.copy()

            calls["option_type"] = "CALL"
            puts["option_type"] = "PUT"

            calls["expiration"] = expiration
            puts["expiration"] = expiration

            calls["DTE"] = dte
            puts["DTE"] = dte

            rows.append(calls)
            rows.append(puts)

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True
    )


def main():

    if len(sys.argv) < 2:

        print(
            "사용법: python option_search.py AAPL"
        )

        return

    ticker = (
        sys.argv[1]
        .upper()
        .strip()
    )

    print("")
    print("🔥 OPTION SEARCH")
    print("")
    print(
        f"Ticker: {ticker}"
    )

    price = get_price(ticker)

    if price is None:

        print(
            "현재가 조회 실패"
        )

        return

    print(
        f"현재가: ${price:.2f}"
    )

    df = get_options(ticker)

    if df.empty:

        print(
            "옵션 데이터 없음"
        )

        return

    print("")
    print(
        "📅 DTE 0~180 전체 만기 분석"
    )

    expirations = (
        df[
            ["expiration", "DTE"]
        ]
        .drop_duplicates()
        .sort_values("DTE")
    )

    print("")

    for _, row in expirations.iterrows():

        print(
            f"{row['expiration']} "
            f"| DTE {int(row['DTE'])}"
        )

    print("")
    print(
        f"옵션 행 수: {len(df):,}"
    )


if __name__ == "__main__":
    main()

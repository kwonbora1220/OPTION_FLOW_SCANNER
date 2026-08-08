import os
from datetime import datetime

import pandas as pd
import yfinance as yf


# ============================================================
# AUTO UNIVERSE
# 미국 주식 전체 후보군 → 유동성/거래량/시총 필터
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MARKET_DIR = os.path.join(BASE_DIR, "01_DATA", "market")
RESULT_DIR = os.path.join(BASE_DIR, "03_RESULTS", "daily")

os.makedirs(MARKET_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 1. 미국 주식 Universe 확보
# ------------------------------------------------------------

print("=" * 70)
print("AUTO UNIVERSE START")
print("=" * 70)

print("\n[1] 미국 주식 Universe 확보 중...")


# Nasdaq Trader의 상장종목 파일 사용
nasdaq_url = (
    "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
)

other_url = (
    "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
)

try:
    nasdaq = pd.read_csv(
        nasdaq_url,
        sep="|",
        dtype=str
    )

    other = pd.read_csv(
        other_url,
        sep="|",
        dtype=str
    )

    nasdaq = nasdaq[
        nasdaq["Symbol"].notna()
        & ~nasdaq["Symbol"].str.contains("File Creation Time", na=False)
    ]

    other = other[
        other["ACT Symbol"].notna()
        & ~other["ACT Symbol"].str.contains("File Creation Time", na=False)
    ]

    symbols = set()

    for symbol in nasdaq["Symbol"]:
        symbols.add(symbol)

    for symbol in other["ACT Symbol"]:
        symbols.add(symbol)

    symbols = sorted(symbols)

    print(f"Universe 후보: {len(symbols):,}개")

except Exception as e:
    print("\nUniverse 다운로드 실패:")
    print(e)
    raise SystemExit


# ------------------------------------------------------------
# 2. 기본 심볼 정리
# ------------------------------------------------------------

clean_symbols = []

for symbol in symbols:

    symbol = str(symbol).strip().upper()

    if not symbol:
        continue

    # ADR/펀드/워런트 등 일부 특수 심볼 제외
    if any(x in symbol for x in ["$", "^", "/", "."]):
        continue

    if len(symbol) > 6:
        continue

    clean_symbols.append(symbol)


clean_symbols = sorted(set(clean_symbols))

print(f"정리 후 후보: {len(clean_symbols):,}개")


# ------------------------------------------------------------
# 3. 빠른 1차 필터
# ------------------------------------------------------------

print("\n[2] 거래량 / 가격 / 시총 후보 압축 중...")

# 너무 작은 종목을 우선 제거하기 위한 최소 기준
MIN_PRICE = 5
MIN_AVG_VOLUME = 1_000_000

# 최종적으로 옵션을 검사할 주식 후보
PRESELECT = 120


# yfinance에서 한 번에 가져올 수 있도록 batch 처리
batch_size = 100

records = []

for start in range(0, len(clean_symbols), batch_size):

    batch = clean_symbols[start:start + batch_size]

    print(
        f"\r시장 데이터 수집: "
        f"{min(start + batch_size, len(clean_symbols)):,}/"
        f"{len(clean_symbols):,}",
        end=""
    )

    try:

        data = yf.download(
            tickers=batch,
            period="5d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )

        for symbol in batch:

            try:

                if len(batch) == 1:
                    close = data["Close"]
                    volume = data["Volume"]

                else:
                    close = data[symbol]["Close"]
                    volume = data[symbol]["Volume"]

                close = pd.to_numeric(close, errors="coerce")
                volume = pd.to_numeric(volume, errors="coerce")

                if close.dropna().empty:
                    continue

                last_price = float(close.dropna().iloc[-1])
                avg_volume = float(volume.dropna().mean())

                if last_price < MIN_PRICE:
                    continue

                if avg_volume < MIN_AVG_VOLUME:
                    continue

                dollar_volume = last_price * avg_volume

                records.append({
                    "Ticker": symbol,
                    "Price": last_price,
                    "AvgVolume": avg_volume,
                    "DollarVolume": dollar_volume
                })

            except Exception:
                continue

    except Exception:
        continue


print()


# ------------------------------------------------------------
# 4. 유동성 순위
# ------------------------------------------------------------

df = pd.DataFrame(records)

if df.empty:
    print("❌ 시장 데이터가 없습니다.")
    raise SystemExit


df = df.sort_values(
    "DollarVolume",
    ascending=False
).reset_index(drop=True)


# 너무 많은 종목은 이후 옵션 검사 부담이 크므로 압축
df = df.head(PRESELECT)


# ------------------------------------------------------------
# 5. 결과 저장
# ------------------------------------------------------------

today = datetime.now().strftime("%Y-%m-%d")

market_file = os.path.join(
    MARKET_DIR,
    f"AUTO_UNIVERSE_{today}.csv"
)

daily_file = os.path.join(
    RESULT_DIR,
    f"AUTO_UNIVERSE_{today}.csv"
)

df.to_csv(
    market_file,
    index=False,
    encoding="utf-8-sig"
)

df.to_csv(
    daily_file,
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 6. 화면 출력
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AUTO UNIVERSE 완료")
print("=" * 70)

print(f"최종 후보: {len(df)}개")

print("\nTOP 30")

print(
    df[
        [
            "Ticker",
            "Price",
            "AvgVolume",
            "DollarVolume"
        ]
    ].head(30).to_string(index=False)
)

print("\n저장:")
print(market_file)
print(daily_file)

print("\n다음 단계:")
print("→ OPTION LIQUIDITY FILTER")
print("=" * 70)
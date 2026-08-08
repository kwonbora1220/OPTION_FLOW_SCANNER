import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, date


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MARKET_DIR = os.path.join(
    BASE_DIR,
    "01_DATA",
    "market"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "03_RESULTS",
    "daily"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

# Universe에서 가져올 최대 종목 수
MAX_UNIVERSE = 120

# 최종적으로 정밀 분석에 넘길 종목 수
TOP_FLOW = 30

# 옵션 최소 거래량
MIN_OPTION_VOLUME = 100

# 옵션 최소 OI
MIN_OPTION_OI = 100

# 현재가 기준 너무 먼 옵션 제외
MAX_DISTANCE = 0.30

# 종목 사이 대기
SLEEP_SECONDS = 2


# ============================================================
# MONEY FORMAT
# ============================================================

def fmt_money(x):

    try:

        x = float(x)

        if x >= 1_000_000_000:
            return f"${x / 1_000_000_000:.2f}B"

        if x >= 1_000_000:
            return f"${x / 1_000_000:.2f}M"

        if x >= 1_000:
            return f"${x / 1_000:.1f}K"

        return f"${x:.0f}"

    except Exception:

        return "$0"


# ============================================================
# FIND LATEST UNIVERSE
# ============================================================

def find_latest_universe():

    files = []

    if not os.path.exists(MARKET_DIR):

        return None

    for file in os.listdir(MARKET_DIR):

        if (
            file.startswith("AUTO_UNIVERSE_")
            and file.endswith(".csv")
        ):

            files.append(
                os.path.join(
                    MARKET_DIR,
                    file
                )
            )

    if not files:

        return None

    files.sort(
        key=os.path.getmtime,
        reverse=True
    )

    return files[0]


# ============================================================
# LOAD UNIVERSE
# ============================================================

def load_universe():

    universe_file = find_latest_universe()

    if universe_file is None:

        raise Exception(
            "AUTO_UNIVERSE CSV를 찾을 수 없습니다."
        )

    print("")
    print(
        f"📂 Universe: {universe_file}"
    )

    df = pd.read_csv(
        universe_file
    )

    if "Ticker" not in df.columns:

        raise Exception(
            "Universe CSV에 Ticker 컬럼이 없습니다."
        )

    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df = df[
        df["Ticker"].notna()
    ]

    df = df.head(
        MAX_UNIVERSE
    )

    return df


# ============================================================
# GET CURRENT PRICE
# ============================================================

def get_current_price(ticker):

    try:

        t = yf.Ticker(
            ticker
        )

        hist = t.history(
            period="5d",
            interval="1d",
            auto_adjust=False,
            timeout=20
        )

        if hist.empty:

            return None

        close = (
            hist["Close"]
            .dropna()
        )

        if close.empty:

            return None

        return float(
            close.iloc[-1]
        )

    except Exception:

        return None


# ============================================================
# OPTION EXPIRATIONS
# ============================================================

def get_valid_expirations(ticker):

    try:

        t = yf.Ticker(
            ticker
        )

        expirations = t.options

        if not expirations:

            return []

        today = date.today()

        result = []

        for exp in expirations:

            try:

                exp_date = datetime.strptime(
                    exp,
                    "%Y-%m-%d"
                ).date()

                dte = (
                    exp_date - today
                ).days

                if 0 <= dte <= 180:

                    result.append(
                        (
                            exp,
                            dte
                        )
                    )

            except Exception:

                continue

        return result

    except Exception:

        return []


# ============================================================
# OPTION FLOW ANALYSIS
# ============================================================

def analyze_option_flow(
    ticker,
    current_price
):

    try:

        t = yf.Ticker(
            ticker
        )

        expirations = (
            get_valid_expirations(
                ticker
            )
        )

        if not expirations:

            return None

        # ----------------------------------------------------
        # 1차 필터에서는 가까운 만기 중심
        # ----------------------------------------------------

        selected = []

        for exp, dte in expirations:

            if dte <= 7:

                selected.append(
                    (exp, dte)
                )

            elif dte <= 30:

                selected.append(
                    (exp, dte)
                )

        # 만기가 너무 많으면 앞쪽 위주
        selected = selected[:8]

        if not selected:

            selected = expirations[:5]

        call_volume = 0
        put_volume = 0

        call_oi = 0
        put_oi = 0

        call_premium = 0
        put_premium = 0

        total_rows = 0

        liquid_calls = 0
        liquid_puts = 0

        dte_buckets = {
            "0_7": 0,
            "8_30": 0,
            "31_60": 0,
            "61_180": 0
        }

        for exp, dte in selected:

            try:

                chain = t.option_chain(
                    exp
                )

                calls = chain.calls.copy()
                puts = chain.puts.copy()

                for df, option_type in [
                    (calls, "CALL"),
                    (puts, "PUT")
                ]:

                    if df.empty:

                        continue

                    df["volume"] = pd.to_numeric(
                        df["volume"],
                        errors="coerce"
                    ).fillna(0)

                    df["openInterest"] = pd.to_numeric(
                        df["openInterest"],
                        errors="coerce"
                    ).fillna(0)

                    df["lastPrice"] = pd.to_numeric(
                        df["lastPrice"],
                        errors="coerce"
                    ).fillna(0)

                    df["strike"] = pd.to_numeric(
                        df["strike"],
                        errors="coerce"
                    ).fillna(0)

                    # ----------------------------------------
                    # 현재가 ±30%
                    # ----------------------------------------

                    df = df[
                        abs(
                            df["strike"]
                            - current_price
                        )
                        / current_price
                        <= MAX_DISTANCE
                    ]

                    if df.empty:

                        continue

                    df = df[
                        (
                            df["volume"]
                            >= MIN_OPTION_VOLUME
                        )
                        |
                        (
                            df["openInterest"]
                            >= MIN_OPTION_OI
                        )
                    ]

                    if df.empty:

                        continue

                    volume = (
                        df["volume"].sum()
                    )

                    oi = (
                        df["openInterest"].sum()
                    )

                    premium = (
                        df["lastPrice"]
                        * df["volume"]
                        * 100
                    ).sum()

                    if option_type == "CALL":

                        call_volume += volume
                        call_oi += oi
                        call_premium += premium

                        liquid_calls += len(df)

                    else:

                        put_volume += volume
                        put_oi += oi
                        put_premium += premium

                        liquid_puts += len(df)

                    total_rows += len(df)

                    if dte <= 7:

                        dte_buckets["0_7"] += len(df)

                    elif dte <= 30:

                        dte_buckets["8_30"] += len(df)

                    elif dte <= 60:

                        dte_buckets["31_60"] += len(df)

                    else:

                        dte_buckets["61_180"] += len(df)

            except Exception:

                continue

        if total_rows == 0:

            return None

        total_volume = (
            call_volume
            + put_volume
        )

        total_oi = (
            call_oi
            + put_oi
        )

        total_premium = (
            call_premium
            + put_premium
        )

        if total_volume <= 0:

            call_volume_ratio = 0.5

        else:

            call_volume_ratio = (
                call_volume
                / total_volume
            )

        if total_oi <= 0:

            call_oi_ratio = 0.5

        else:

            call_oi_ratio = (
                call_oi
                / total_oi
            )

        if total_premium <= 0:

            call_premium_ratio = 0.5

        else:

            call_premium_ratio = (
                call_premium
                / total_premium
            )

        # ====================================================
        # FLOW SCORE
        # ====================================================

        score = 50.0

        reasons = []

        # ----------------------------------------------------
        # Premium
        # ----------------------------------------------------

        if call_premium_ratio >= 0.60:

            score += 15

            reasons.append(
                "Call Premium 강세"
            )

        elif call_premium_ratio >= 0.55:

            score += 8

            reasons.append(
                "Call Premium 우세"
            )

        elif call_premium_ratio <= 0.40:

            score -= 15

            reasons.append(
                "Put Premium 강세"
            )

        elif call_premium_ratio <= 0.45:

            score -= 8

            reasons.append(
                "Put Premium 우세"
            )

        # ----------------------------------------------------
        # Volume
        # ----------------------------------------------------

        if call_volume_ratio >= 0.60:

            score += 12

            reasons.append(
                "Call 거래량 우세"
            )

        elif call_volume_ratio >= 0.55:

            score += 6

            reasons.append(
                "Call 거래량 우세"
            )

        elif call_volume_ratio <= 0.40:

            score -= 12

            reasons.append(
                "Put 거래량 우세"
            )

        elif call_volume_ratio <= 0.45:

            score -= 6

            reasons.append(
                "Put 거래량 우세"
            )

        # ----------------------------------------------------
        # OI
        # ----------------------------------------------------

        if call_oi_ratio >= 0.60:

            score += 10

            reasons.append(
                "Call OI 우세"
            )

        elif call_oi_ratio >= 0.55:

            score += 5

            reasons.append(
                "Call OI 우세"
            )

        elif call_oi_ratio <= 0.40:

            score -= 10

            reasons.append(
                "Put OI 우세"
            )

        elif call_oi_ratio <= 0.45:

            score -= 5

            reasons.append(
                "Put OI 우세"
            )

        # ----------------------------------------------------
        # DTE 구조
        # ----------------------------------------------------

        if dte_buckets["8_30"] > 0:

            score += 3

        if dte_buckets["31_60"] > 0:

            score += 3

        if dte_buckets["61_180"] > 0:

            score += 3

        score = max(
            0,
            min(
                100,
                score
            )
        )

        if score >= 70:

            direction = "BULLISH"

        elif score >= 55:

            direction = "SLIGHT BULLISH"

        elif score >= 45:

            direction = "NEUTRAL"

        elif score >= 30:

            direction = "SLIGHT BEARISH"

        else:

            direction = "BEARISH"

        return {
            "ticker": ticker,
            "price": current_price,
            "score": score,
            "direction": direction,
            "call_volume": call_volume,
            "put_volume": put_volume,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "call_premium": call_premium,
            "put_premium": put_premium,
            "call_volume_ratio": call_volume_ratio,
            "call_oi_ratio": call_oi_ratio,
            "call_premium_ratio": call_premium_ratio,
            "liquid_calls": liquid_calls,
            "liquid_puts": liquid_puts,
            "dte_0_7": dte_buckets["0_7"],
            "dte_8_30": dte_buckets["8_30"],
            "dte_31_60": dte_buckets["31_60"],
            "dte_61_180": dte_buckets["61_180"],
            "reasons": reasons
        }

    except Exception:

        return None


# ============================================================
# MAIN
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "🔥 OPTION FLOW FILTER V2"
)

print(
    "=" * 70
)

try:

    universe = load_universe()

except Exception as e:

    print("")
    print(
        f"❌ Universe 오류: {e}"
    )

    raise SystemExit


print("")
print(
    f"📊 검사 대상: "
    f"{len(universe)}개"
)

print("")

results = []


# ============================================================
# SEARCH
# ============================================================

for i, row in enumerate(
    universe.itertuples(index=False),
    1
):

    ticker = str(
        row.Ticker
    ).upper().strip()

    print(
        "=" * 70
    )

    print(
        f"🔥 {i}/{len(universe)} : {ticker}"
    )

    try:

        price = get_current_price(
            ticker
        )

        if price is None:

            print(
                "⏭️ 현재가 없음"
            )

            continue

        print(
            f"💰 ${price:.2f}"
        )

        result = analyze_option_flow(
            ticker,
            price
        )

        if result is None:

            print(
                "⏭️ 옵션 데이터 없음"
            )

            continue

        results.append(
            result
        )

        print(
            f"📊 Flow Score: "
            f"{result['score']:.1f}"
        )

        print(
            f"📈 Call Premium: "
            f"{fmt_money(result['call_premium'])}"
        )

        print(
            f"📉 Put Premium: "
            f"{fmt_money(result['put_premium'])}"
        )

        print(
            f"📊 Call Volume Ratio: "
            f"{result['call_volume_ratio'] * 100:.1f}%"
        )

        print(
            f"📊 Call OI Ratio: "
            f"{result['call_oi_ratio'] * 100:.1f}%"
        )

        if result["reasons"]:

            print(
                "→ "
                + ", ".join(
                    result["reasons"]
                )
            )

    except Exception as e:

        print(
            f"❌ {ticker} 오류: {e}"
        )

    if i < len(universe):

        print(
            "⏳ 다음 종목..."
        )

        time.sleep(
            SLEEP_SECONDS
        )


# ============================================================
# FINAL
# ============================================================

if not results:

    print("")
    print(
        "❌ 옵션 수급 분석 결과가 없습니다."
    )

    raise SystemExit


results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)

top_results = results[
    :TOP_FLOW
]


# ============================================================
# SAVE
# ============================================================

today = datetime.now().strftime(
    "%Y-%m-%d"
)

output_file = os.path.join(
    RESULT_DIR,
    f"OPTION_FLOW_FILTER_{today}.csv"
)

save_rows = []

for r in top_results:

    save_rows.append(
        {
            "ticker": r["ticker"],
            "price": r["price"],
            "score": r["score"],
            "direction": r["direction"],
            "call_premium": r["call_premium"],
            "put_premium": r["put_premium"],
            "call_volume": r["call_volume"],
            "put_volume": r["put_volume"],
            "call_oi": r["call_oi"],
            "put_oi": r["put_oi"],
            "call_volume_ratio": r["call_volume_ratio"],
            "call_oi_ratio": r["call_oi_ratio"],
            "call_premium_ratio": r["call_premium_ratio"],
            "dte_0_7": r["dte_0_7"],
            "dte_8_30": r["dte_8_30"],
            "dte_31_60": r["dte_31_60"],
            "dte_61_180": r["dte_61_180"],
            "reasons": " | ".join(
                r["reasons"]
            )
        }
    )

pd.DataFrame(
    save_rows
).to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# DISPLAY
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "🧠 OPTION FLOW FILTER RESULT"
)

print(
    "=" * 70
)

print("")

for i, r in enumerate(
    top_results,
    1
):

    print(
        f"{i:>2}. "
        f"{r['ticker']:<6} "
        f"{r['score']:>5.1f} "
        f"{r['direction']}"
    )

    print(
        f"    Call Premium "
        f"{fmt_money(r['call_premium'])}"
        f" | Put Premium "
        f"{fmt_money(r['put_premium'])}"
    )

    print(
        f"    Call Vol "
        f"{r['call_volume']:,.0f}"
        f" | Put Vol "
        f"{r['put_volume']:,.0f}"
    )

    print(
        f"    Call OI "
        f"{r['call_oi']:,.0f}"
        f" | Put OI "
        f"{r['put_oi']:,.0f}"
    )

    if r["reasons"]:

        print(
            "    → "
            + ", ".join(
                r["reasons"]
            )
        )

    print("")


print(
    "=" * 70
)

print(
    f"💾 저장 완료: {output_file}"
)

print(
    "=" * 70
)

print(
    "🔥 OPTION FLOW FILTER 완료"
)

print(
    "다음 단계 → TOP 30 정밀 OPTION SEARCH"
)

print(
    "=" * 70
)
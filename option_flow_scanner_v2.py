import os
import time
import math
import urllib.parse
import urllib.request
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# OPTION FLOW SCANNER V2
# GitHub Actions / Local Windows 공용
# ============================================================


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "01_DATA"
)

MARKET_DIR = os.path.join(
    DATA_DIR,
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


# ============================================================
# TELEGRAM
# ============================================================

# GitHub Actions:
#   TELEGRAM_BOT_TOKEN
#   TELEGRAM_CHAT_ID
#
# Windows CMD:
#   set TELEGRAM_BOT_TOKEN=...
#   set TELEGRAM_CHAT_ID=...

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    ""
).strip()

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    ""
).strip()

TELEGRAM_API_URL = (
    "https://api.telegram.org/bot{}/sendMessage"
)

TELEGRAM_MAX_LENGTH = 4000


# ============================================================
# SCANNER SETTINGS
# ============================================================

MAX_WORKERS = 8

MAX_UNIVERSE = 120

FLOW_PRESELECT = 20

TOP_ENTRY = 5

ENTRY_SCORE = 70

WATCH_SCORE = 40

MIN_DTE = 0

MAX_DTE = 180

MIN_OPTION_VOLUME = 100

MIN_OPEN_INTEREST = 100


# ============================================================
# FORMAT
# ============================================================

def fmt_money(value):

    try:

        x = float(value)

        sign = "-" if x < 0 else ""

        x = abs(x)

        if x >= 1_000_000_000:
            return f"{sign}${x / 1_000_000_000:.2f}B"

        if x >= 1_000_000:
            return f"{sign}${x / 1_000_000:.2f}M"

        if x >= 1_000:
            return f"{sign}${x / 1_000:.1f}K"

        return f"{sign}${x:.0f}"

    except Exception:

        return "$0"


# ============================================================
# LOAD UNIVERSE
# ============================================================

def load_latest_universe():

    if not os.path.exists(MARKET_DIR):

        print("ERROR: MARKET directory not found")
        print(MARKET_DIR)

        return pd.DataFrame()

    files = []

    for filename in os.listdir(MARKET_DIR):

        if (
            filename.startswith("AUTO_UNIVERSE_")
            and filename.endswith(".csv")
        ):

            files.append(
                os.path.join(
                    MARKET_DIR,
                    filename
                )
            )

    if not files:

        print("ERROR: AUTO_UNIVERSE csv not found")

        return pd.DataFrame()

    files.sort(
        key=os.path.getmtime,
        reverse=True
    )

    latest = files[0]

    print()
    print("Latest Universe:")
    print(latest)

    try:

        df = pd.read_csv(
            latest,
            encoding="utf-8-sig"
        )

    except Exception:

        df = pd.read_csv(
            latest
        )

    if "Ticker" not in df.columns:

        print("ERROR: Ticker column not found")

        return pd.DataFrame()

    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df = df[
        df["Ticker"].notna()
    ]

    df = df.drop_duplicates(
        subset=["Ticker"]
    )

    df = df.head(
        MAX_UNIVERSE
    )

    return df


# ============================================================
# CURRENT PRICE
# ============================================================

def get_current_price(ticker):

    try:

        stock = yf.Ticker(
            ticker
        )

        hist = stock.history(
            period="5d",
            interval="1d",
            auto_adjust=False
        )

        if hist.empty:

            return None

        close = hist["Close"].dropna()

        if close.empty:

            return None

        return float(
            close.iloc[-1]
        )

    except Exception as e:

        print(
            f"\nPRICE ERROR {ticker}: {e}"
        )

        return None


# ============================================================
# OPTION DATA
# ============================================================

def get_option_data(ticker):

    try:

        stock = yf.Ticker(
            ticker
        )

        expirations = stock.options

        if not expirations:

            return pd.DataFrame()

        today = pd.Timestamp.now().normalize()

        selected_expirations = []

        for expiration in expirations:

            try:

                exp_date = pd.Timestamp(
                    expiration
                )

                dte = (
                    exp_date - today
                ).days

                if (
                    MIN_DTE
                    <= dte
                    <= MAX_DTE
                ):

                    selected_expirations.append(
                        expiration
                    )

            except Exception:

                continue

        if not selected_expirations:

            return pd.DataFrame()

        rows = []

        for expiration in selected_expirations:

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

                rows.append(
                    calls
                )

                rows.append(
                    puts
                )

            except Exception as e:

                print(
                    f"\nOPTION ERROR "
                    f"{ticker} {expiration}: {e}"
                )

                continue

        if not rows:

            return pd.DataFrame()

        return pd.concat(
            rows,
            ignore_index=True
        )

    except Exception as e:

        print(
            f"\nOPTION DATA ERROR {ticker}: {e}"
        )

        return pd.DataFrame()


# ============================================================
# PREMIUM FLOW
# ============================================================

def calculate_premium_flow(df):

    df = df.copy()

    numeric_columns = [
        "volume",
        "openInterest",
        "lastPrice",
        "bid",
        "ask",
        "strike",
        "impliedVolatility"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    df["mid_price"] = (
        df["bid"] + df["ask"]
    ) / 2

    df["mid_price"] = df[
        "mid_price"
    ].where(
        df["mid_price"] > 0,
        df["lastPrice"]
    )

    df["premium_flow"] = (
        df["volume"]
        * df["mid_price"]
        * 100
    )

    return df


# ============================================================
# DTE
# ============================================================

def calculate_dte(df):

    df = df.copy()

    today = pd.Timestamp.now().normalize()

    df["expiration_date"] = pd.to_datetime(
        df["expiration"],
        errors="coerce"
    )

    df["DTE"] = (
        df["expiration_date"]
        - today
    ).dt.days

    df = df[
        (df["DTE"] >= MIN_DTE)
        & (df["DTE"] <= MAX_DTE)
    ]

    return df


# ============================================================
# BASIC FLOW
# ============================================================

def calculate_basic_flow(df):

    calls = df[
        df["option_type"] == "CALL"
    ]

    puts = df[
        df["option_type"] == "PUT"
    ]

    if calls.empty or puts.empty:

        return {
            "flow_score": 0,
            "call_premium": 0,
            "put_premium": 0,
            "call_volume": 0,
            "put_volume": 0,
            "call_oi": 0,
            "put_oi": 0,
            "call_premium_ratio": 0,
            "call_volume_ratio": 0,
            "call_oi_ratio": 0
        }

    call_premium = float(
        calls["premium_flow"].sum()
    )

    put_premium = float(
        puts["premium_flow"].sum()
    )

    call_volume = float(
        calls["volume"].sum()
    )

    put_volume = float(
        puts["volume"].sum()
    )

    call_oi = float(
        calls["openInterest"].sum()
    )

    put_oi = float(
        puts["openInterest"].sum()
    )

    total_premium = (
        call_premium + put_premium
    )

    total_volume = (
        call_volume + put_volume
    )

    total_oi = (
        call_oi + put_oi
    )

    call_premium_ratio = (
        call_premium / total_premium
        if total_premium > 0
        else 0
    )

    call_volume_ratio = (
        call_volume / total_volume
        if total_volume > 0
        else 0
    )

    call_oi_ratio = (
        call_oi / total_oi
        if total_oi > 0
        else 0
    )

    score = 50.0

    if call_premium_ratio >= 0.60:

        score += 12

    elif call_premium_ratio >= 0.55:

        score += 7

    elif call_premium_ratio <= 0.40:

        score -= 12

    elif call_premium_ratio <= 0.45:

        score -= 7

    if call_volume_ratio >= 0.60:

        score += 10

    elif call_volume_ratio <= 0.40:

        score -= 10

    if call_oi_ratio >= 0.60:

        score += 8

    elif call_oi_ratio <= 0.40:

        score -= 8

    score = max(
        0,
        min(100, score)
    )

    return {
        "flow_score": score,
        "call_premium": call_premium,
        "put_premium": put_premium,
        "call_volume": call_volume,
        "put_volume": put_volume,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "call_premium_ratio": call_premium_ratio,
        "call_volume_ratio": call_volume_ratio,
        "call_oi_ratio": call_oi_ratio
    }


# ============================================================
# FAST SCAN
# ============================================================

def fast_scan(ticker):

    try:

        price = get_current_price(
            ticker
        )

        if price is None:

            return None

        df = get_option_data(
            ticker
        )

        if df.empty:

            return None

        df = calculate_premium_flow(
            df
        )

        df = calculate_dte(
            df
        )

        if df.empty:

            return None

        flow = calculate_basic_flow(
            df
        )

        return {
            "ticker": ticker,
            "price": price,
            "df": df,
            **flow
        }

    except Exception as e:

        print(
            f"\nSCAN ERROR {ticker}: {e}"
        )

        return None


# ============================================================
# DTE QUALITY
# ============================================================

def dte_quality(df):

    score = 0

    reasons = []

    ranges = [
        (
            0,
            7,
            2,
            "0-7DTE"
        ),
        (
            8,
            30,
            5,
            "8-30DTE"
        ),
        (
            31,
            60,
            5,
            "31-60DTE"
        ),
        (
            61,
            180,
            5,
            "61-180DTE"
        )
    ]

    for low, high, points, label in ranges:

        subset = df[
            (df["DTE"] >= low)
            & (df["DTE"] <= high)
        ]

        if not subset.empty:

            score += points

            reasons.append(
                label
            )

    return score, reasons


# ============================================================
# WALL
# ============================================================

def calculate_walls(
    df,
    current_price
):

    calls = df[
        df["option_type"] == "CALL"
    ].copy()

    puts = df[
        df["option_type"] == "PUT"
    ].copy()

    call_wall = None
    put_wall = None

    calls_above = calls[
        calls["strike"] > current_price
    ].copy()

    if not calls_above.empty:

        calls_above["wall_strength"] = (
            calls_above["openInterest"]
            * calls_above["volume"].clip(
                lower=1
            )
            * calls_above["mid_price"].clip(
                lower=0.01
            )
        )

        grouped = (
            calls_above
            .groupby("strike")[
                "wall_strength"
            ]
            .sum()
        )

        if not grouped.empty:

            call_wall = float(
                grouped.idxmax()
            )

    puts_below = puts[
        puts["strike"] < current_price
    ].copy()

    if not puts_below.empty:

        puts_below["wall_strength"] = (
            puts_below["openInterest"]
            * puts_below["volume"].clip(
                lower=1
            )
            * puts_below["mid_price"].clip(
                lower=0.01
            )
        )

        grouped = (
            puts_below
            .groupby("strike")[
                "wall_strength"
            ]
            .sum()
        )

        if not grouped.empty:

            put_wall = float(
                grouped.idxmax()
            )

    call_distance = None
    put_distance = None

    if call_wall is not None:

        call_distance = (
            (call_wall - current_price)
            / current_price
            * 100
        )

    if put_wall is not None:

        put_distance = (
            (put_wall - current_price)
            / current_price
            * 100
        )

    return {
        "call_wall": call_wall,
        "put_wall": put_wall,
        "call_distance": call_distance,
        "put_distance": put_distance
    }


# ============================================================
# WALL SCORE
# ============================================================

def calculate_wall_score(walls):

    score = 0

    reasons = []

    call_distance = walls.get(
        "call_distance"
    )

    put_distance = walls.get(
        "put_distance"
    )

    if call_distance is not None:

        if call_distance >= 10:

            score += 5

            reasons.append(
                f"Call Wall +{call_distance:.1f}%"
            )

        elif call_distance >= 5:

            score += 3

            reasons.append(
                f"Call Wall +{call_distance:.1f}%"
            )

        elif call_distance >= 2:

            score += 1

        else:

            score -= 5

            reasons.append(
                "Call Wall Close"
            )

    if put_distance is not None:

        if abs(put_distance) >= 8:

            score += 3

        elif abs(put_distance) >= 4:

            score += 2

        else:

            score -= 2

            reasons.append(
                "Put Wall Close"
            )

    if (
        call_distance is not None
        and put_distance is not None
    ):

        space = (
            call_distance
            - put_distance
        )

        if space >= 12:

            score += 4

            reasons.append(
                "Wide Wall Space"
            )

        elif space >= 7:

            score += 2

        else:

            score -= 4

            reasons.append(
                "Narrow Wall Space"
            )

    return score, reasons


# ============================================================
# IV
# ============================================================

def calculate_iv_quality(df):

    if "impliedVolatility" not in df.columns:

        return 0, "IV unavailable", None

    iv = pd.to_numeric(
        df["impliedVolatility"],
        errors="coerce"
    )

    iv = iv.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if iv.empty:

        return (
            0,
            "IV unavailable",
            None
        )

    median_iv = float(
        iv.median()
    ) * 100

    if median_iv < 40:

        return (
            5,
            "Low IV",
            median_iv
        )

    if median_iv < 80:

        return (
            3,
            "Normal IV",
            median_iv
        )

    if median_iv < 120:

        return (
            -3,
            "High IV",
            median_iv
        )

    return (
        -7,
        "Extreme IV",
        median_iv
    )


# ============================================================
# SIGNAL CONFLICT
# ============================================================

def signal_conflict(
    delta,
    gex,
    hiro,
    vanna
):

    bullish = 0
    bearish = 0

    values = [
        delta,
        gex,
        hiro,
        vanna
    ]

    for value in values:

        if value > 0:

            bullish += 1

        elif value < 0:

            bearish += 1

    difference = abs(
        bullish - bearish
    )

    if (
        bullish >= 2
        and bearish >= 1
    ) or (
        bearish >= 2
        and bullish >= 1
    ):

        if difference <= 1:

            return (
                -10,
                "Signal Conflict"
            )

        return (
            -5,
            "Signal Conflict"
        )

    return (
        0,
        None
    )


# ============================================================
# APPROX GREEKS
# ============================================================

def calculate_greeks(
    df,
    current_price
):

    calls = df[
        df["option_type"] == "CALL"
    ]

    puts = df[
        df["option_type"] == "PUT"
    ]

    delta = 0.0
    gex = 0.0
    hiro = 0.0
    vanna = 0.0

    for _, row in calls.iterrows():

        oi = float(
            row.get(
                "openInterest",
                0
            ) or 0
        )

        volume = float(
            row.get(
                "volume",
                0
            ) or 0
        )

        strike = float(
            row.get(
                "strike",
                current_price
            )
        )

        moneyness = (
            strike / current_price
        )

        if moneyness < 0.95:

            d = 0.80

        elif moneyness < 1.02:

            d = 0.50

        else:

            d = 0.20

        delta += (
            oi
            * d
            * 100
            * current_price
        )

        hiro += (
            volume
            * d
            * 100
            * current_price
        )

    for _, row in puts.iterrows():

        oi = float(
            row.get(
                "openInterest",
                0
            ) or 0
        )

        volume = float(
            row.get(
                "volume",
                0
            ) or 0
        )

        strike = float(
            row.get(
                "strike",
                current_price
            )
        )

        moneyness = (
            strike / current_price
        )

        if moneyness > 1.05:

            d = -0.80

        elif moneyness > 0.98:

            d = -0.50

        else:

            d = -0.20

        delta += (
            oi
            * d
            * 100
            * current_price
        )

        hiro += (
            volume
            * d
            * 100
            * current_price
        )

    for _, row in df.iterrows():

        oi = float(
            row.get(
                "openInterest",
                0
            ) or 0
        )

        iv = float(
            row.get(
                "impliedVolatility",
                0
            ) or 0
        )

        strike = float(
            row.get(
                "strike",
                current_price
            )
        )

        if iv <= 0:

            iv = 0.50

        distance = (
            abs(
                strike - current_price
            )
            / current_price
        )

        gamma = (
            math.exp(
                -distance * 10
            )
            / (
                current_price
                * iv
                * 2
            )
        )

        sign = 1

        if row["option_type"] == "PUT":

            sign = -1

        gex += (
            oi
            * gamma
            * 100
            * current_price
            * sign
        )

        v = (
            oi
            * (
                strike
                - current_price
            )
            / current_price
            * iv
        )

        if row["option_type"] == "CALL":

            vanna += v

        else:

            vanna -= v

    return {
        "Delta": delta,
        "GEX": gex,
        "HIRO": hiro,
        "Vanna": vanna
    }


# ============================================================
# FINAL SCORE
# ============================================================

def calculate_final_score(
    df,
    current_price
):

    reasons = []

    score = 50.0

    calls = df[
        df["option_type"] == "CALL"
    ]

    puts = df[
        df["option_type"] == "PUT"
    ]

    if calls.empty or puts.empty:

        return (
            score,
            "NEUTRAL",
            reasons,
            {},
            {}
        )

    call_premium = calls[
        "premium_flow"
    ].sum()

    put_premium = puts[
        "premium_flow"
    ].sum()

    total_premium = (
        call_premium
        + put_premium
    )

    if total_premium > 0:

        ratio = (
            call_premium
            / total_premium
        )

        if ratio >= 0.60:

            score += 12
            reasons.append(
                "Call Premium Strong"
            )

        elif ratio >= 0.55:

            score += 7
            reasons.append(
                "Call Premium Bias"
            )

        elif ratio <= 0.40:

            score -= 12
            reasons.append(
                "Put Premium Strong"
            )

        elif ratio <= 0.45:

            score -= 7
            reasons.append(
                "Put Premium Bias"
            )

    call_volume = calls[
        "volume"
    ].sum()

    put_volume = puts[
        "volume"
    ].sum()

    total_volume = (
        call_volume
        + put_volume
    )

    if total_volume > 0:

        ratio = (
            call_volume
            / total_volume
        )

        if ratio >= 0.60:

            score += 10
            reasons.append(
                "Call Volume Strong"
            )

        elif ratio <= 0.40:

            score -= 10
            reasons.append(
                "Put Volume Strong"
            )

    call_oi = calls[
        "openInterest"
    ].sum()

    put_oi = puts[
        "openInterest"
    ].sum()

    total_oi = (
        call_oi
        + put_oi
    )

    if total_oi > 0:

        ratio = (
            call_oi
            / total_oi
        )

        if ratio >= 0.60:

            score += 8
            reasons.append(
                "Call OI Strong"
            )

        elif ratio <= 0.40:

            score -= 8
            reasons.append(
                "Put OI Strong"
            )

    dte_score, dte_reasons = (
        dte_quality(df)
    )

    score += dte_score

    reasons.extend(
        dte_reasons
    )

    long_calls = calls[
        calls["DTE"] >= 30
    ]

    long_puts = puts[
        puts["DTE"] >= 30
    ]

    if not long_calls.empty:

        score += 5

        reasons.append(
            "30D+ Call Structure"
        )

    if not long_puts.empty:

        score -= 2

        reasons.append(
            "30D+ Put Structure"
        )

    greeks = calculate_greeks(
        df,
        current_price
    )

    delta = greeks["Delta"]
    gex = greeks["GEX"]
    hiro = greeks["HIRO"]
    vanna = greeks["Vanna"]

    if abs(delta) > 5_000_000:

        if delta > 0:

            score += 12

            reasons.append(
                "Strong Positive Delta"
            )

        else:

            score -= 12

            reasons.append(
                "Strong Negative Delta"
            )

    elif delta > 0:

        score += 7

        reasons.append(
            "Positive Delta"
        )

    elif delta < 0:

        score -= 7

        reasons.append(
            "Negative Delta"
        )

    if gex > 0:

        score += 3

        reasons.append(
            "Positive GEX"
        )

    elif gex < 0:

        score -= 3

        reasons.append(
            "Negative GEX"
        )

    if hiro > 0:

        score += 5

        reasons.append(
            "Positive HIRO"
        )

    elif hiro < 0:

        score -= 5

        reasons.append(
            "Negative HIRO"
        )

    if vanna > 0:

        score += 3

        reasons.append(
            "Positive Vanna"
        )

    elif vanna < 0:

        score -= 3

        reasons.append(
            "Negative Vanna"
        )

    walls = calculate_walls(
        df,
        current_price
    )

    wall_score, wall_reasons = (
        calculate_wall_score(
            walls
        )
    )

    score += wall_score

    reasons.extend(
        wall_reasons
    )

    iv_score, iv_reason, median_iv = (
        calculate_iv_quality(
            df
        )
    )

    score += iv_score

    if iv_reason:

        reasons.append(
            iv_reason
        )

    conflict_score, conflict_reason = (
        signal_conflict(
            delta,
            gex,
            hiro,
            vanna
        )
    )

    score += conflict_score

    if conflict_reason:

        reasons.append(
            conflict_reason
        )

    score = max(
        0,
        min(100, score)
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

    details = {
        "delta": delta,
        "gex": gex,
        "hiro": hiro,
        "vanna": vanna,
        "call_wall": walls.get(
            "call_wall"
        ),
        "put_wall": walls.get(
            "put_wall"
        ),
        "call_distance": walls.get(
            "call_distance"
        ),
        "put_distance": walls.get(
            "put_distance"
        ),
        "iv": median_iv
    }

    return (
        score,
        direction,
        reasons,
        details,
        greeks
    )


# ============================================================
# CLASSIFY
# ============================================================

def classify(score):

    if score >= ENTRY_SCORE:

        return "TODAY ENTRY"

    if score >= WATCH_SCORE:

        return "WATCH"

    return "AVOID"


# ============================================================
# ANALYZE ONE
# ============================================================

def analyze_one(item):

    ticker = item["ticker"]

    try:

        score, direction, reasons, details, greeks = (
            calculate_final_score(
                item["df"],
                item["price"]
            )
        )

        return {
            "ticker": ticker,
            "price": item["price"],
            "score": score,
            "direction": direction,
            "category": classify(
                score
            ),
            "reasons": reasons,
            **details
        }

    except Exception as e:

        print(
            f"\nANALYSIS ERROR "
            f"{ticker}: {e}"
        )

        return None


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if not TELEGRAM_BOT_TOKEN:

        raise ValueError(
            "TELEGRAM_BOT_TOKEN is not set."
        )

    if not TELEGRAM_CHAT_ID:

        raise ValueError(
            "TELEGRAM_CHAT_ID is not set."
        )

    url = TELEGRAM_API_URL.format(
        TELEGRAM_BOT_TOKEN
    )

    chunks = []

    current = ""

    for line in message.split("\n"):

        if (
            len(current)
            + len(line)
            + 1
            > TELEGRAM_MAX_LENGTH
        ):

            if current:

                chunks.append(
                    current
                )

                current = ""

            while len(line) > TELEGRAM_MAX_LENGTH:

                chunks.append(
                    line[:TELEGRAM_MAX_LENGTH]
                )

                line = line[
                    TELEGRAM_MAX_LENGTH:
                ]

        if current:

            current += "\n" + line

        else:

            current = line

    if current:

        chunks.append(
            current
        )

    for chunk in chunks:

        payload = urllib.parse.urlencode(
            {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
                "disable_web_page_preview": "true"
            }
        ).encode(
            "utf-8"
        )

        request = urllib.request.Request(
            url,
            data=payload,
            method="POST"
        )

        with urllib.request.urlopen(
            request,
            timeout=30
        ) as response:

            result = (
                response
                .read()
                .decode("utf-8")
            )

        if '"ok":true' not in result.replace(
            " ",
            ""
        ):

            raise RuntimeError(
                f"Telegram API error: {result}"
            )


# ============================================================
# BUILD MESSAGE
# ============================================================

def build_message(results):

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    lines = []

    lines.append(
        "🔥 OPTION FLOW SCANNER V2"
    )

    lines.append(
        f"📅 {today}"
    )

    lines.append("")

    entry = [
        x for x in results
        if x["score"] >= ENTRY_SCORE
    ][:TOP_ENTRY]

    watch = [
        x for x in results
        if WATCH_SCORE
        <= x["score"]
        < ENTRY_SCORE
    ]

    avoid = [
        x for x in results
        if x["score"] < WATCH_SCORE
    ]

    lines.append(
        "🟢 TODAY ENTRY TOP 5"
    )

    lines.append("")

    if entry:

        for i, r in enumerate(
            entry,
            1
        ):

            lines.append(
                f"{i}. {r['ticker']} "
                f"${r['price']:.2f} "
                f"| {r['score']:.1f} "
                f"| {r['direction']}"
            )

            if r["reasons"]:

                lines.append(
                    "   "
                    + " | ".join(
                        r["reasons"][:5]
                    )
                )

    else:

        lines.append(
            "No entry candidates."
        )

    lines.append("")

    lines.append(
        "🟡 WATCH"
    )

    lines.append("")

    if watch:

        for r in watch:

            lines.append(
                f"• {r['ticker']} "
                f"${r['price']:.2f} "
                f"| {r['score']:.1f} "
                f"| {r['direction']}"
            )

    else:

        lines.append(
            "No watch candidates."
        )

    lines.append("")

    lines.append(
        "🔴 AVOID"
    )

    lines.append("")

    if avoid:

        for r in avoid:

            lines.append(
                f"• {r['ticker']} "
                f"${r['price']:.2f} "
                f"| {r['score']:.1f} "
                f"| {r['direction']}"
            )

    else:

        lines.append(
            "No avoid candidates."
        )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📊 ALL FINAL RANKING"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    for i, r in enumerate(
        results,
        1
    ):

        lines.append(
            f"{i}. {r['ticker']:<6} "
            f"{r['score']:>5.1f} "
            f"| {r['direction']}"
        )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📌 TOP STRUCTURE"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    for r in results[:5]:

        lines.append(
            f"{r['ticker']} "
            f"${r['price']:.2f}"
        )

        if r["call_wall"] is not None:

            distance = r[
                "call_distance"
            ]

            if distance is not None:

                lines.append(
                    f"Call Wall "
                    f"${r['call_wall']:.2f} "
                    f"({distance:+.1f}%)"
                )

        if r["put_wall"] is not None:

            distance = r[
                "put_distance"
            ]

            if distance is not None:

                lines.append(
                    f"Put Wall "
                    f"${r['put_wall']:.2f} "
                    f"({distance:+.1f}%)"
                )

        if r["iv"] is not None:

            lines.append(
                f"IV {r['iv']:.1f}%"
            )

        lines.append("")

    lines.append(
        "⚠️ Greeks / GEX / HIRO / "
        "Vanna are proxy calculations "
        "based on yfinance option data."
    )

    return "\n".join(
        lines
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(results):

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    filename = (
        f"OPTION_FINAL_RANKING_V2_"
        f"{today}.csv"
    )

    path = os.path.join(
        RESULT_DIR,
        filename
    )

    rows = []

    for r in results:

        rows.append(
            {
                "ticker": r["ticker"],
                "price": r["price"],
                "score": r["score"],
                "direction": r["direction"],
                "category": r["category"],
                "reasons": " | ".join(
                    r["reasons"]
                ),
                "delta": r["delta"],
                "gex": r["gex"],
                "hiro": r["hiro"],
                "vanna": r["vanna"],
                "call_wall": r["call_wall"],
                "put_wall": r["put_wall"],
                "call_distance": r[
                    "call_distance"
                ],
                "put_distance": r[
                    "put_distance"
                ],
                "iv": r["iv"]
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "🔥 OPTION FLOW SCANNER V2"
    )
    print("=" * 70)

    universe = load_latest_universe()

    if universe.empty:

        print(
            "ERROR: Universe is empty."
        )

        return 1

    tickers = (
        universe["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
        .drop_duplicates()
        .head(MAX_UNIVERSE)
        .tolist()
    )

    print()
    print(
        f"Universe: {len(tickers)} tickers"
    )

    print(
        f"Workers: {MAX_WORKERS}"
    )

    print()

    fast_results = []

    completed = 0

    start_time = time.time()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                fast_scan,
                ticker
            ): ticker
            for ticker in tickers
        }

        for future in as_completed(
            futures
        ):

            ticker = futures[
                future
            ]

            completed += 1

            try:

                result = future.result()

                if result is not None:

                    fast_results.append(
                        result
                    )

                    print(
                        f"\rFLOW "
                        f"{completed}/"
                        f"{len(tickers)} "
                        f"| {ticker:<6} "
                        f"| {result['flow_score']:>5.1f}",
                        end=""
                    )

                else:

                    print(
                        f"\rFLOW "
                        f"{completed}/"
                        f"{len(tickers)} "
                        f"| {ticker:<6} "
                        f"| SKIP",
                        end=""
                    )

            except Exception as e:

                print(
                    f"\rFLOW "
                    f"{completed}/"
                    f"{len(tickers)} "
                    f"| {ticker:<6} "
                    f"| ERROR {e}",
                    end=""
                )

    print()

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Fast scan completed "
        f"in {elapsed:.1f}s"
    )

    if not fast_results:

        print(
            "ERROR: No option data."
        )

        return 1

    fast_results.sort(
        key=lambda x: x[
            "flow_score"
        ],
        reverse=True
    )

    preselected = fast_results[
        :FLOW_PRESELECT
    ]

    print()

    print(
        "FLOW PRESELECT"
    )

    for i, r in enumerate(
        preselected,
        1
    ):

        print(
            f"{i:>2}. "
            f"{r['ticker']:<6} "
            f"| Flow "
            f"{r['flow_score']:>5.1f} "
            f"| Call "
            f"{fmt_money(r['call_premium'])} "
            f"| Put "
            f"{fmt_money(r['put_premium'])}"
        )

    print()

    final_results = []

    for i, item in enumerate(
        preselected,
        1
    ):

        print(
            f"FINAL "
            f"{i}/{len(preselected)} "
            f"{item['ticker']}"
        )

        result = analyze_one(
            item
        )

        if result is not None:

            final_results.append(
                result
            )

    if not final_results:

        print(
            "ERROR: Final analysis empty."
        )

        return 1

    final_results.sort(
        key=lambda x: x[
            "score"
        ],
        reverse=True
    )

    message = build_message(
        final_results
    )

    print()
    print("=" * 70)
    print(message)
    print("=" * 70)

    csv_path = save_csv(
        final_results
    )

    print()
    print(
        f"CSV saved: {csv_path}"
    )

    try:

        send_telegram(
            message
        )

        print(
            "✅ Telegram sent successfully."
        )

    except Exception as e:

        print(
            f"❌ Telegram error: {e}"
        )

        # 분석 자체는 성공했으므로
        # Telegram 실패 때문에 전체 작업을
        # 실패 처리하지 않음.

    print()
    print(
        "🔥 OPTION FLOW SCANNER V2 DONE"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

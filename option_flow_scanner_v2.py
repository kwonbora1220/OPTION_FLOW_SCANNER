

import os
import time
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import yfinance as yf
import urllib.parse
import urllib.request


# ============================================================
# OPTION FLOW SCANNER V2
# ============================================================
#
# 목적
# ------------------------------------------------------------
# 1. AUTO_UNIVERSE_YYYY-MM-DD.csv 자동 탐색
# 2. Universe 종목을 읽음
# 3. 옵션 유동성/수급 1차 필터
# 4. 좋은 종목만 정밀 분석
# 5. 기존 OPTION FINAL RANKING 구조 유지
#
# V2 핵심
# ------------------------------------------------------------
# 기존:
#   120개 종목 → 하나씩 순차 옵션 조회 → 매우 느림
#
# V2:
#   120개 종목 → 병렬 처리
#   → 1차 Flow Filter
#   → 상위 종목만 정밀 분석
#
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


# ------------------------------------------------------------
# V2 속도 설정
# ------------------------------------------------------------

# 동시에 검사할 종목 수
MAX_WORKERS = 8

# Universe에서 가져올 최대 종목
MAX_UNIVERSE = 120

# 1차 옵션 Flow 통과 종목 수
FLOW_PRESELECT = 20

# 최종 TOP
TOP_ENTRY = 5

# ------------------------------------------------------------
# NEW TELEGRAM BOT
# ------------------------------------------------------------
# 새 Telegram BotFather 토큰과 Chat ID를 여기에 입력하세요.
# 토큰은 외부에 공개하지 마세요.
TELEGRAM_BOT_TOKEN = "여기에_새봇_TOKEN_입력"
TELEGRAM_CHAT_ID = "여기에_CHAT_ID_입력"
TELEGRAM_API_URL = "https://api.telegram.org/bot{}/sendMessage"
TELEGRAM_MAX_LENGTH = 4000


# ------------------------------------------------------------
# 최종 점수 기준
# ------------------------------------------------------------

ENTRY_SCORE = 70
WATCH_SCORE = 40


# ------------------------------------------------------------
# 옵션 조건
# ------------------------------------------------------------

MIN_DTE = 0
MAX_DTE = 180

MIN_OPTION_VOLUME = 100
MIN_OPEN_INTEREST = 100


# ============================================================
# FORMAT
# ============================================================

def fmt_money(x):

    try:

        x = float(x)

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

    files = []

    if not os.path.exists(MARKET_DIR):
        return pd.DataFrame()

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

        return pd.DataFrame()

    files.sort(
        key=os.path.getmtime,
        reverse=True
    )

    latest = files[0]

    print("")
    print("📂 Universe 파일:")
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

        print("")
        print("❌ Universe에 Ticker 컬럼이 없습니다.")

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

    except Exception:

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

        for exp in expirations:

            try:

                exp_date = pd.Timestamp(
                    exp
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
                        exp
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

            except Exception:

                continue

        if not rows:

            return pd.DataFrame()

        df = pd.concat(
            rows,
            ignore_index=True
        )

        return df

    except Exception:

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
        "ask"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    # --------------------------------------------------------
    # Mid Price
    # --------------------------------------------------------

    df["mid_price"] = (
        (
            df["bid"]
            + df["ask"]
        )
        / 2
    )

    df["mid_price"] = df[
        "mid_price"
    ].where(
        df["mid_price"] > 0,
        df["lastPrice"]
    )

    # --------------------------------------------------------
    # Premium
    # --------------------------------------------------------

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
    ].copy()

    puts = df[
        df["option_type"] == "PUT"
    ].copy()

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

    call_premium = (
        calls["premium_flow"].sum()
    )

    put_premium = (
        puts["premium_flow"].sum()
    )

    call_volume = (
        calls["volume"].sum()
    )

    put_volume = (
        puts["volume"].sum()
    )

    call_oi = (
        calls["openInterest"].sum()
    )

    put_oi = (
        puts["openInterest"].sum()
    )

    total_premium = (
        call_premium
        + put_premium
    )

    total_volume = (
        call_volume
        + put_volume
    )

    total_oi = (
        call_oi
        + put_oi
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

    # Premium

    if call_premium_ratio >= 0.60:
        score += 12

    elif call_premium_ratio >= 0.55:
        score += 7

    elif call_premium_ratio <= 0.40:
        score -= 12

    elif call_premium_ratio <= 0.45:
        score -= 7

    # Volume

    if call_volume_ratio >= 0.60:
        score += 10

    elif call_volume_ratio <= 0.40:
        score -= 10

    # OI

    if call_oi_ratio >= 0.60:
        score += 8

    elif call_oi_ratio <= 0.40:
        score -= 8

    score = max(
        0,
        min(
            100,
            score
        )
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
# FAST FLOW SCAN
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

    except Exception:

        return None


# ============================================================
# DTE QUALITY
# ============================================================

def dte_quality(df):

    reasons = []
    score = 0

    dte_0_7 = df[
        (df["DTE"] >= 0)
        & (df["DTE"] <= 7)
    ]

    dte_8_30 = df[
        (df["DTE"] >= 8)
        & (df["DTE"] <= 30)
    ]

    dte_31_60 = df[
        (df["DTE"] >= 31)
        & (df["DTE"] <= 60)
    ]

    dte_61_180 = df[
        (df["DTE"] >= 61)
        & (df["DTE"] <= 180)
    ]

    if not dte_0_7.empty:

        score += 2

        reasons.append(
            "0~7DTE 구조"
        )

    if not dte_8_30.empty:

        score += 5

        reasons.append(
            "8~30DTE 구조"
        )

    if not dte_31_60.empty:

        score += 5

        reasons.append(
            "31~60DTE 구조"
        )

    if not dte_61_180.empty:

        score += 5

        reasons.append(
            "61~180DTE 구조"
        )

    return score, reasons


# ============================================================
# WALL
# ============================================================

def calculate_walls(df, current_price):

    calls = df[
        df["option_type"] == "CALL"
    ].copy()

    puts = df[
        df["option_type"] == "PUT"
    ].copy()

    call_wall = None
    put_wall = None

    # --------------------------------------------------------
    # CALL WALL
    # 현재가 위쪽만 검사
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PUT WALL
    # 현재가 아래쪽만 검사
    # --------------------------------------------------------

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

    if call_wall:

        call_distance = (
            (call_wall - current_price)
            / current_price
            * 100
        )

    if put_wall:

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
# IV QUALITY
# ============================================================

def calculate_iv_quality(df):

    iv = pd.to_numeric(
        df.get(
            "impliedVolatility",
            pd.Series(dtype=float)
        ),
        errors="coerce"
    )

    iv = iv.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if iv.empty:

        return 0, "IV 데이터 없음", None

    median_iv = float(
        iv.median()
    ) * 100

    if median_iv < 40:

        return (
            5,
            "IV 낮음",
            median_iv
        )

    elif median_iv < 80:

        return (
            3,
            "IV 적정",
            median_iv
        )

    elif median_iv < 120:

        return (
            -3,
            "IV 과열",
            median_iv
        )

    else:

        return (
            -7,
            "IV 극단적",
            median_iv
        )


# ============================================================
# WALL DISTANCE SCORE
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
                f"Call Wall 여유 +{call_distance:.1f}%"
            )

        elif call_distance >= 5:

            score += 3

            reasons.append(
                f"Call Wall +{call_distance:.1f}%"
            )

        elif call_distance >= 2:

            score += 1

            reasons.append(
                f"Call Wall 근접"
            )

        else:

            score -= 5

            reasons.append(
                f"Call Wall 매우 근접"
            )

    if put_distance is not None:

        if abs(put_distance) >= 8:

            score += 3

            reasons.append(
                f"Put Wall 지지 {put_distance:+.1f}%"
            )

        elif abs(put_distance) >= 4:

            score += 2

        else:

            score -= 2

            reasons.append(
                f"Put Wall 근접"
            )

    # --------------------------------------------------------
    # Wall Space
    # --------------------------------------------------------

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
                "Wall 공간 넓음"
            )

        elif space >= 7:

            score += 2

            reasons.append(
                "Wall 공간 양호"
            )

        else:

            score -= 4

            reasons.append(
                "Wall 사이 협소"
            )

    return score, reasons


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

    if delta > 0:
        bullish += 1

    elif delta < 0:
        bearish += 1

    if gex > 0:
        bullish += 1

    elif gex < 0:
        bearish += 1

    if hiro > 0:
        bullish += 1

    elif hiro < 0:
        bearish += 1

    if vanna > 0:
        bullish += 1

    elif vanna < 0:
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
                "⚠️ Signal Conflict"
            )

        return (
            -5,
            "⚠️ Signal Conflict"
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
    ].copy()

    puts = df[
        df["option_type"] == "PUT"
    ].copy()

    delta = 0.0
    gex = 0.0
    vanna = 0.0
    hiro = 0.0

    # --------------------------------------------------------
    # Delta Proxy
    # --------------------------------------------------------

    for _, row in calls.iterrows():

        oi = float(
            row.get(
                "openInterest",
                0
            )
            or 0
        )

        volume = float(
            row.get(
                "volume",
                0
            )
            or 0
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
            )
            or 0
        )

        volume = float(
            row.get(
                "volume",
                0
            )
            or 0
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

    # --------------------------------------------------------
    # GEX Proxy
    # --------------------------------------------------------

    for _, row in df.iterrows():

        oi = float(
            row.get(
                "openInterest",
                0
            )
            or 0
        )

        iv = float(
            row.get(
                "impliedVolatility",
                0
            )
            or 0
        )

        strike = float(
            row.get(
                "strike",
                current_price
            )
        )

        if iv <= 0:
            iv = 0.50

        distance = abs(
            strike - current_price
        ) / current_price

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

    # --------------------------------------------------------
    # Vanna Proxy
    # --------------------------------------------------------

    for _, row in df.iterrows():

        oi = float(
            row.get(
                "openInterest",
                0
            )
            or 0
        )

        iv = float(
            row.get(
                "impliedVolatility",
                0
            )
            or 0
        )

        if iv <= 0:
            continue

        strike = float(
            row.get(
                "strike",
                current_price
            )
        )

        distance = (
            strike
            - current_price
        ) / current_price

        v = (
            oi
            * distance
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
            None
        )

    # --------------------------------------------------------
    # Premium
    # --------------------------------------------------------

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
                "Call Premium 강세"
            )

        elif ratio >= 0.55:

            score += 7
            reasons.append(
                "Call Premium 우세"
            )

        elif ratio <= 0.40:

            score -= 12
            reasons.append(
                "Put Premium 강세"
            )

        elif ratio <= 0.45:

            score -= 7
            reasons.append(
                "Put Premium 우세"
            )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

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
                "Call 거래량 우세"
            )

        elif ratio <= 0.40:

            score -= 10
            reasons.append(
                "Put 거래량 우세"
            )

    # --------------------------------------------------------
    # OI
    # --------------------------------------------------------

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
                "Call OI 우세"
            )

        elif ratio <= 0.40:

            score -= 8
            reasons.append(
                "Put OI 우세"
            )

    # --------------------------------------------------------
    # DTE QUALITY
    # --------------------------------------------------------

    dte_score, dte_reasons = (
        dte_quality(df)
    )

    score += dte_score

    reasons.extend(
        dte_reasons
    )

    # --------------------------------------------------------
    # 30D+
    # --------------------------------------------------------

    long_calls = calls[
        calls["DTE"] >= 30
    ]

    long_puts = puts[
        puts["DTE"] >= 30
    ]

    if not long_calls.empty:

        score += 5

        reasons.append(
            "30D+ Call 구조 존재"
        )

    if not long_puts.empty:

        score -= 2

        reasons.append(
            "30D+ Put 구조 존재"
        )

    # --------------------------------------------------------
    # Greeks
    # --------------------------------------------------------

    greeks = calculate_greeks(
        df,
        current_price
    )

    delta = greeks["Delta"]
    gex = greeks["GEX"]
    hiro = greeks["HIRO"]
    vanna = greeks["Vanna"]

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    if abs(delta) > 5_000_000:

        if delta > 0:

            score += 12

            reasons.append(
                "Delta Exposure 강한 상방"
            )

        else:

            score -= 12

            reasons.append(
                "Delta Exposure 강한 하방"
            )

    elif delta > 0:

        score += 7

        reasons.append(
            "Delta Exposure 상방"
        )

    elif delta < 0:

        score -= 7

        reasons.append(
            "Delta Exposure 하방"
        )

    # --------------------------------------------------------
    # GEX
    # --------------------------------------------------------

    if gex > 0:

        score += 3

        reasons.append(
            "GEX Positive"
        )

    elif gex < 0:

        score -= 3

        reasons.append(
            "GEX Negative"
        )

    # --------------------------------------------------------
    # HIRO
    # --------------------------------------------------------

    if hiro > 0:

        score += 5

        reasons.append(
            "HIRO Proxy Positive"
        )

    elif hiro < 0:

        score -= 5

        reasons.append(
            "HIRO Proxy Negative"
        )

    # --------------------------------------------------------
    # Vanna
    # --------------------------------------------------------

    if vanna > 0:

        score += 3

        reasons.append(
            "Vanna 상방"
        )

    elif vanna < 0:

        score -= 3

        reasons.append(
            "Vanna 하방"
        )

    # --------------------------------------------------------
    # WALL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # IV
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SIGNAL CONFLICT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LIMIT
    # --------------------------------------------------------

    score = max(
        0,
        min(
            100,
            score
        )
    )

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

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
# CALL BUY + PUT SELL STRUCTURE
# ============================================================
#
# IMPORTANT
# ------------------------------------------------------------
# yfinance 옵션 체인만으로 실제 체결 방향(BUY/SELL)을 확정할 수 없음.
# 따라서 아래는 "체결 방향 확정"이 아니라
# CALL BUY + PUT SELL "구조 추정 후보"를 찾는 휴리스틱이다.
#
# CALL BUY 추정:
#   - 현재가 위/근처의 Call
#   - 높은 거래량
#   - 높은 Premium
#   - 충분한 OI
#
# PUT SELL 추정:
#   - 현재가 아래의 Put
#   - 충분한 OI
#   - 거래량이 존재
#   - 상대적으로 낮은 Put Premium
#
# 동일 만기 안에서 가장 좋은 Call/Put 조합을 찾는다.
# ============================================================

def find_call_buy_put_sell_candidates(
    df,
    current_price,
    max_candidates=5
):
    if df.empty:
        return []

    work = df.copy()

    numeric_cols = [
        "strike",
        "volume",
        "openInterest",
        "mid_price",
        "premium_flow",
        "impliedVolatility",
        "DTE"
    ]

    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(
                work[col],
                errors="coerce"
            ).fillna(0)

    calls = work[
        work["option_type"] == "CALL"
    ].copy()

    puts = work[
        work["option_type"] == "PUT"
    ].copy()

    if calls.empty or puts.empty:
        return []

    # --------------------------------------------------------
    # Call BUY 추정 후보
    # 현재가 근처~상방 12%까지
    # --------------------------------------------------------
    call_candidates = calls[
        (calls["strike"] >= current_price * 0.98)
        & (calls["strike"] <= current_price * 1.12)
        & (calls["volume"] >= MIN_OPTION_VOLUME)
    ].copy()

    # --------------------------------------------------------
    # Put SELL 추정 후보
    # 현재가 아래 12%까지
    # --------------------------------------------------------
    put_candidates = puts[
        (puts["strike"] <= current_price * 1.00)
        & (puts["strike"] >= current_price * 0.88)
        & (puts["volume"] >= MIN_OPTION_VOLUME)
        & (puts["openInterest"] >= MIN_OPEN_INTEREST)
    ].copy()

    if call_candidates.empty or put_candidates.empty:
        return []

    # --------------------------------------------------------
    # 정규화 함수
    # --------------------------------------------------------
    def safe_ratio(series):
        max_value = float(series.max()) if not series.empty else 0
        if max_value <= 0:
            return pd.Series(
                0.0,
                index=series.index
            )
        return series / max_value

    call_candidates["volume_score"] = safe_ratio(
        call_candidates["volume"]
    )
    call_candidates["premium_score"] = safe_ratio(
        call_candidates["premium_flow"]
    )
    call_candidates["oi_score"] = safe_ratio(
        call_candidates["openInterest"]
    )

    put_candidates["volume_score"] = safe_ratio(
        put_candidates["volume"]
    )
    put_candidates["oi_score"] = safe_ratio(
        put_candidates["openInterest"]
    )

    # Call BUY 점수
    call_candidates["buy_score"] = (
        call_candidates["premium_score"] * 0.45
        + call_candidates["volume_score"] * 0.35
        + call_candidates["oi_score"] * 0.20
    )

    # Put SELL 구조 점수
    # 실제 SELL 확정값이 아니라 "지지/매도 프리미엄 구조" 추정
    put_candidates["sell_score"] = (
        put_candidates["oi_score"] * 0.55
        + put_candidates["volume_score"] * 0.25
        + (1 - safe_ratio(
            put_candidates["premium_flow"]
        )) * 0.20
    )

    results = []

    # --------------------------------------------------------
    # 같은 만기끼리 조합
    # --------------------------------------------------------
    for expiration in sorted(
        set(call_candidates["expiration"])
        & set(put_candidates["expiration"])
    ):
        c = call_candidates[
            call_candidates["expiration"] == expiration
        ].copy()

        p = put_candidates[
            put_candidates["expiration"] == expiration
        ].copy()

        if c.empty or p.empty:
            continue

        best_call = c.sort_values(
            "buy_score",
            ascending=False
        ).iloc[0]

        best_put = p.sort_values(
            "sell_score",
            ascending=False
        ).iloc[0]

        dte = int(
            min(
                best_call.get("DTE", 0),
                best_put.get("DTE", 0)
            )
        )

        call_score = float(
            best_call["buy_score"]
        )
        put_score = float(
            best_put["sell_score"]
        )

        structure_score = (
            call_score * 60
            + put_score * 40
        )

        # 너무 약한 조합은 제외
        if structure_score < 35:
            continue

        # 현재가 대비 strike 위치
        call_distance = (
            (float(best_call["strike"]) - current_price)
            / current_price
            * 100
        )

        put_distance = (
            (float(best_put["strike"]) - current_price)
            / current_price
            * 100
        )

        results.append({
            "expiration": str(expiration),
            "DTE": dte,
            "call_strike": float(best_call["strike"]),
            "call_volume": float(best_call["volume"]),
            "call_oi": float(best_call["openInterest"]),
            "call_premium": float(best_call["premium_flow"]),
            "call_iv": float(
                best_call.get(
                    "impliedVolatility",
                    0
                )
            ) * 100,
            "call_distance": call_distance,
            "put_strike": float(best_put["strike"]),
            "put_volume": float(best_put["volume"]),
            "put_oi": float(best_put["openInterest"]),
            "put_premium": float(best_put["premium_flow"]),
            "put_iv": float(
                best_put.get(
                    "impliedVolatility",
                    0
                )
            ) * 100,
            "put_distance": put_distance,
            "call_score": call_score,
            "put_score": put_score,
            "structure_score": structure_score
        })

    results.sort(
        key=lambda x: x["structure_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # 동일 종목에서 중복 만기 과다 노출 방지
    # --------------------------------------------------------
    return results[:max_candidates]


def build_call_buy_put_sell_recommendations(
    final_results,
    max_tickers=5
):
    recommendations = []

    for result in final_results:
        df = result.get("_df")

        if df is None or df.empty:
            continue

        candidates = find_call_buy_put_sell_candidates(
            df,
            result["price"],
            max_candidates=3
        )

        if not candidates:
            continue

        best = candidates[0]

        # 기존 최종 점수와 구조 점수를 함께 반영
        recommendation_score = (
            result["score"] * 0.65
            + best["structure_score"] * 0.35
        )

        recommendations.append({
            "ticker": result["ticker"],
            "price": result["price"],
            "score": result["score"],
            "recommendation_score": recommendation_score,
            "direction": result["direction"],
            "candidate": best
        })

    recommendations.sort(
        key=lambda x: x["recommendation_score"],
        reverse=True
    )

    return recommendations[:max_tickers]


# ============================================================
# CLASSIFY
# ============================================================

def classify(score):

    if score >= ENTRY_SCORE:

        return "🟢 오늘 진입 후보"

    if score >= WATCH_SCORE:

        return "🟡 관망"

    return "🔴 회피"


# ============================================================
# ANALYZE ONE
# ============================================================

def analyze_one(item):

    ticker = item["ticker"]

    try:

        df = item["df"]

        price = item["price"]

        score, direction, reasons, details, greeks = (
            calculate_final_score(
                df,
                price
            )
        )

        return {
            "ticker": ticker,
            "price": price,
            "score": score,
            "direction": direction,
            "category": classify(
                score
            ),
            "reasons": reasons,
            "_df": df,
            **details
        }

    except Exception as e:

        return None


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("🔥 OPTION FLOW SCANNER V2")
print("=" * 70)

print("")
print("🚀 AUTO UNIVERSE 기반 전체 종목 옵션 수급 검색")
print("")

universe = load_latest_universe()

if universe.empty:

    print("")
    print("❌ Universe 파일을 찾을 수 없습니다.")
    print("")
    print("먼저 auto_universe.py를 실행하세요.")
    print("")

    raise SystemExit


tickers = (
    universe["Ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
    .drop_duplicates()
    .head(MAX_UNIVERSE)
    .tolist()
)

print("")
print(
    f"📊 검사 대상: {len(tickers)}개"
)

print("")
print(
    f"⚡ 병렬 검사: {MAX_WORKERS}개 동시"
)

print("")
print("=" * 70)


# ============================================================
# FAST SCAN
# ============================================================

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
                    f"\r⚡ 1차 옵션 수급 검사 "
                    f"{completed}/{len(tickers)} "
                    f"| {ticker:<6} "
                    f"| Flow {result['flow_score']:>5.1f}",
                    end=""
                )

            else:

                print(
                    f"\r⚡ 1차 옵션 수급 검사 "
                    f"{completed}/{len(tickers)} "
                    f"| {ticker:<6} "
                    f"| 제외",
                    end=""
                )

        except Exception:

            print(
                f"\r⚡ 1차 옵션 수급 검사 "
                f"{completed}/{len(tickers)} "
                f"| {ticker:<6} "
                f"| 오류",
                end=""
            )


print("")

elapsed = (
    time.time()
    - start_time
)

print("")
print(
    f"⏱️ 1차 검사 완료: "
    f"{elapsed:.1f}초"
)

print(
    f"📊 옵션 데이터 확보: "
    f"{len(fast_results)}개"
)


# ============================================================
# FLOW PRESELECT
# ============================================================

fast_results = sorted(
    fast_results,
    key=lambda x: x[
        "flow_score"
    ],
    reverse=True
)

preselected = fast_results[
    :FLOW_PRESELECT
]


print("")
print("=" * 70)
print("🎯 1차 OPTION FLOW TOP")
print("=" * 70)

for i, r in enumerate(
    preselected,
    1
):

    print(
        f"{i:>2}. "
        f"{r['ticker']:<6} "
        f"| Flow {r['flow_score']:>5.1f} "
        f"| Call Premium "
        f"{fmt_money(r['call_premium'])} "
        f"| Put Premium "
        f"{fmt_money(r['put_premium'])}"
    )


# ============================================================
# FINAL ANALYSIS
# ============================================================

print("")
print("=" * 70)
print(
    f"🔥 정밀 분석 대상: "
    f"{len(preselected)}개"
)
print("=" * 70)

final_results = []

for i, item in enumerate(
    preselected,
    1
):

    ticker = item["ticker"]

    print("")
    print(
        f"🔥 FINAL {i}/{len(preselected)} : "
        f"{ticker}"
    )

    result = analyze_one(
        item
    )

    if result is None:

        print(
            "❌ 정밀 분석 실패"
        )

        continue

    final_results.append(
        result
    )

    print(
        f"💰 ${result['price']:.2f}"
    )

    print(
        f"📊 Score: "
        f"{result['score']:.1f}"
    )

    print(
        f"📈 Direction: "
        f"{result['direction']}"
    )

    if result["call_wall"] is not None:

        print(
            f"📈 Call Wall: "
            f"${result['call_wall']:.2f}"
        )

    if result["put_wall"] is not None:

        print(
            f"📉 Put Wall: "
            f"${result['put_wall']:.2f}"
        )

    if result["iv"] is not None:

        print(
            f"IV: "
            f"{result['iv']:.1f}%"
        )


# ============================================================
# SORT
# ============================================================

if not final_results:

    print("")
    print(
        "❌ 최종 분석 결과가 없습니다."
    )

    raise SystemExit


final_results = sorted(
    final_results,
    key=lambda x: x["score"],
    reverse=True
)


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

lines = []

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append(
    "🧠 오늘의 OPTION FINAL RANKING"
)

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append("")

# ------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------

entry = [
    x
    for x in final_results
    if x["score"] >= ENTRY_SCORE
][:TOP_ENTRY]

lines.append(
    "🟢 오늘 진입 후보 TOP 5"
)

lines.append("")

if entry:

    for i, r in enumerate(
        entry,
        1
    ):

        lines.append(
            f"{i}. {r['ticker']} | "
            f"{r['score']:.1f}점 | "
            f"{r['direction']}"
        )

        if r["reasons"]:

            lines.append(
                "   → "
                + ", ".join(
                    r["reasons"]
                )
            )

else:

    lines.append(
        "오늘 진입 후보 없음"
    )


lines.append("")

# ------------------------------------------------------------
# WATCH
# ------------------------------------------------------------

watch = [
    x
    for x in final_results
    if WATCH_SCORE
    <= x["score"]
    < ENTRY_SCORE
]

lines.append(
    "🟡 관망"
)

lines.append("")

if watch:

    for r in watch:

        lines.append(
            f"• {r['ticker']} | "
            f"{r['score']:.1f}점 | "
            f"{r['direction']}"
        )

        if r["reasons"]:

            lines.append(
                "→ "
                + ", ".join(
                    r["reasons"]
                )
            )

else:

    lines.append(
        "관망 종목 없음"
    )


lines.append("")

# ------------------------------------------------------------
# AVOID
# ------------------------------------------------------------

avoid = [
    x
    for x in final_results
    if x["score"] < WATCH_SCORE
]

lines.append(
    "🔴 회피"
)

lines.append("")

if avoid:

    for r in avoid:

        lines.append(
            f"• {r['ticker']} | "
            f"{r['score']:.1f}점 | "
            f"{r['direction']}"
        )

        if r["reasons"]:

            lines.append(
                "→ "
                + ", ".join(
                    r["reasons"]
                )
            )

else:

    lines.append(
        "회피 종목 없음"
    )


# ============================================================
# ALL SCORES
# ============================================================

lines.append("")

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append(
    "📊 전체 종목 점수"
)

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append("")

for i, r in enumerate(
    final_results,
    1
):

    lines.append(
        f"{i}. {r['ticker']:<6} "
        f"{r['score']:>5.1f}점 "
        f"{r['category']}"
    )


# ============================================================
# TOP 5 STRUCTURE
# ============================================================

lines.append("")

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append(
    "🎯 TOP 5 구조 상세"
)

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append("")

for r in final_results[:5]:

    lines.append(
        f"📌 {r['ticker']} "
        f"${r['price']:.2f}"
    )

    if r["call_wall"] is not None:

        if r["call_distance"] is not None:

            lines.append(
                f"📈 Call Wall "
                f"${r['call_wall']:.2f} "
                f"({r['call_distance']:+.1f}%)"
            )

        else:

            lines.append(
                f"📈 Call Wall "
                f"${r['call_wall']:.2f}"
            )

    if r["put_wall"] is not None:

        if r["put_distance"] is not None:

            lines.append(
                f"📉 Put Wall "
                f"${r['put_wall']:.2f} "
                f"({r['put_distance']:+.1f}%)"
            )

        else:

            lines.append(
                f"📉 Put Wall "
                f"${r['put_wall']:.2f}"
            )

    if r["iv"] is not None:

        lines.append(
            f"IV {r['iv']:.1f}%"
        )

    lines.append("")


# ============================================================
# CALL BUY + PUT SELL RECOMMENDATION
# ============================================================

call_put_recommendations = (
    build_call_buy_put_sell_recommendations(
        final_results,
        max_tickers=5
    )
)

lines.append("")
lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)
lines.append(
    "🎯 CALL BUY + PUT SELL 후보"
)
lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)
lines.append("")

if call_put_recommendations:

    for i, rec in enumerate(
        call_put_recommendations,
        1
    ):
        c = rec["candidate"]

        lines.append(
            f"{i}. {rec['ticker']} "
            f"${rec['price']:.2f} | "
            f"구조점수 {rec['recommendation_score']:.1f}"
        )

        lines.append(
            f"📅 {c['expiration']} | "
            f"DTE {c['DTE']}"
        )

        lines.append(
            f"🟢 CALL BUY 추정 "
            f"${c['call_strike']:.2f} "
            f"({c['call_distance']:+.1f}%)"
        )

        lines.append(
            f"   Premium {fmt_money(c['call_premium'])} | "
            f"Vol {c['call_volume']:,.0f} | "
            f"OI {c['call_oi']:,.0f}"
        )

        lines.append(
            f"🔴 PUT SELL 추정 "
            f"${c['put_strike']:.2f} "
            f"({c['put_distance']:+.1f}%)"
        )

        lines.append(
            f"   Premium {fmt_money(c['put_premium'])} | "
            f"Vol {c['put_volume']:,.0f} | "
            f"OI {c['put_oi']:,.0f}"
        )

        lines.append(
            f"📊 Call 구조 {c['call_score']*100:.0f} "
            f"| Put 구조 {c['put_score']*100:.0f}"
        )

        lines.append("")

else:

    lines.append(
        "현재 조건에 맞는 후보 없음"
    )

lines.append(
    "⚠️ CALL BUY / PUT SELL은 "
    "yfinance 옵션 체인 기반 구조 추정입니다."
)
lines.append(
    "⚠️ 실제 체결 방향 BUY/SELL을 "
    "확정하는 데이터는 아닙니다."
)
lines.append("")


# ============================================================
# SCORE COMPONENTS
# ============================================================

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append(
    "📊 최종 스코어 구성"
)

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append("")

lines.append(
    "Volume / OI / Premium"
)

lines.append(
    "- DTE Quality"
)

lines.append(
    "- Delta Exposure"
)

lines.append(
    "- GEX"
)

lines.append(
    "- HIRO Proxy"
)

lines.append(
    "- Vanna Exposure"
)

lines.append(
    "- Call Wall / Put Wall Position"
)

lines.append(
    "- Wall Distance"
)

lines.append(
    "- Wall Space"
)

lines.append(
    "- IV Quality"
)

lines.append(
    "- Signal Conflict Penalty"
)

lines.append("")

lines.append(
    "⚠️ Delta/Gamma/Vanna은 "
    "옵션 데이터 기반 Proxy 계산값입니다."
)

lines.append(
    "⚠️ GEX는 OI × Gamma 기반 "
    "근사 계산값입니다."
)

lines.append(
    "⚠️ HIRO는 실제 체결 방향 데이터가 "
    "없는 yfinance 환경의 Proxy입니다."
)

lines.append(
    "⚠️ 옵션 거래량만으로 실제 BUY/SELL을 "
    "확정할 수 없습니다."
)


final_message = "\n".join(
    lines
)


# ============================================================
# PRINT FINAL
# ============================================================

print("")
print("=" * 70)
print(
    final_message
)
print("=" * 70)


# ============================================================
# NEW TELEGRAM BOT
# ============================================================

def send_telegram_new_bot(message):

    if (
        not TELEGRAM_BOT_TOKEN
        or "여기에_새봇" in TELEGRAM_BOT_TOKEN
        or not TELEGRAM_CHAT_ID
        or "여기에_CHAT_ID" in str(TELEGRAM_CHAT_ID)
    ):
        raise ValueError(
            "Telegram Bot Token / Chat ID가 설정되지 않았습니다."
        )

    url = TELEGRAM_API_URL.format(
        TELEGRAM_BOT_TOKEN
    )

    # Telegram 메시지 최대 길이를 고려해 자동 분할
    chunks = []
    current = ""

    for line in message.split("\n"):

        if len(current) + len(line) + 1 > TELEGRAM_MAX_LENGTH:

            if current:
                chunks.append(current)
                current = ""

            # 한 줄 자체가 너무 길 경우 강제 분할
            while len(line) > TELEGRAM_MAX_LENGTH:
                chunks.append(
                    line[:TELEGRAM_MAX_LENGTH]
                )
                line = line[TELEGRAM_MAX_LENGTH:]

        if current:
            current += "\n" + line
        else:
            current = line

    if current:
        chunks.append(current)

    for chunk in chunks:

        payload = urllib.parse.urlencode(
            {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
                "disable_web_page_preview": "true"
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=payload,
            method="POST"
        )

        with urllib.request.urlopen(
            request,
            timeout=20
        ) as response:

            result = response.read().decode(
                "utf-8"
            )

        if '"ok":true' not in result.replace(" ", ""):
            raise RuntimeError(
                f"Telegram API 응답 오류: {result}"
            )


try:

    send_telegram_new_bot(
        final_message
    )

    print("")
    print(
        "📨 새 Telegram Bot 전송 완료"
    )

except Exception as e:

    print("")
    print(
        f"❌ 새 Telegram 전송 오류: {e}"
    )


# _df는 Telegram 추천 계산에만 사용하고 CSV에는 저장하지 않는다.
for r in final_results:
    r.pop("_df", None)

# ============================================================
# CSV
# ============================================================

today = datetime.now().strftime(
    "%Y-%m-%d"
)

ranking_file = os.path.join(
    RESULT_DIR,
    f"OPTION_FINAL_RANKING_V2_{today}.csv"
)

ranking_rows = []

for r in final_results:

    ranking_rows.append(
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
    ranking_rows
).to_csv(
    ranking_file,
    index=False,
    encoding="utf-8-sig"
)


print("")
print(
    f"💾 최종 순위 저장:"
)
print(
    ranking_file
)

print("")
print("=" * 70)
print(
    "🔥 OPTION FLOW SCANNER V2 완료"
)
print("=" * 70)
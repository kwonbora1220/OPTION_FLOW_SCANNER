#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNIVERSAL DRIVER ENGINE
=======================

종목별 과거 OHLCV 구조에서 반복되는 Driver(운전수)를 자동 탐색한다.

핵심 원칙
1. 종목별로 1~3개의 Driver를 자동 결정한다.
2. Driver 개수를 강제로 3개로 만들지 않는다.
3. 신규 상장 종목은 짧은 역사에서도 분석하되, 장기 사이클을 억지로 만들지 않는다.
4. 현재 패턴과 과거 Driver의 유사도를 계산한다.
5. 과거 Driver 발생 후 미래 수익/상승폭/하락폭/반복 간격을 기록한다.
6. 기존 옵션 스캐너와 독립적으로 실행된다.

실행:
    python driver_engine.py
    python driver_engine.py RKLB UBER CBRS

환경변수:
    DRIVER_SYMBOLS="RKLB,UBER,CBRS"
    DRIVER_LOOKBACK="10y"
    DRIVER_WINDOW=40
    DRIVER_FORWARD=40
    DRIVER_MIN_HISTORY=80
    DRIVER_MAX_DRIVERS=3
    DRIVER_ACTIVE_SIM=0.82
    DRIVER_WATCH_SIM=0.72
    DRIVER_CLUSTER_SIM=0.90
    DRIVER_EVENT_SPACING=10

출력:
    03_RESULTS/daily/driver_profile.csv
    03_RESULTS/daily/driver_current.csv
    03_RESULTS/daily/driver_history.csv
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("03_RESULTS/daily")

PROFILE_FILE = OUTPUT_DIR / "driver_profile.csv"
CURRENT_FILE = OUTPUT_DIR / "driver_current.csv"
HISTORY_FILE = OUTPUT_DIR / "driver_history.csv"

LOOKBACK_PERIOD = os.getenv("DRIVER_LOOKBACK", "10y")

WINDOW = int(os.getenv("DRIVER_WINDOW", "40"))
FORWARD = int(os.getenv("DRIVER_FORWARD", "40"))

# 신규 상장 종목 대응
MIN_HISTORY = int(os.getenv("DRIVER_MIN_HISTORY", "80"))

MAX_DRIVERS = int(os.getenv("DRIVER_MAX_DRIVERS", "3"))

CURRENT_ACTIVE_SIM = float(
    os.getenv("DRIVER_ACTIVE_SIM", "0.82")
)

CURRENT_WATCH_SIM = float(
    os.getenv("DRIVER_WATCH_SIM", "0.72")
)

CLUSTER_SIM = float(
    os.getenv("DRIVER_CLUSTER_SIM", "0.90")
)

EVENT_SPACING = int(
    os.getenv("DRIVER_EVENT_SPACING", "10")
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_download(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance single/multi-index 결과를 OHLCV DataFrame으로 정리.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        flattened = []

        for col in out.columns:
            found = None

            for item in col:
                item_str = str(item)

                if item_str in {
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume",
                }:
                    found = item_str
                    break

            flattened.append(
                found if found else str(col[0])
            )

        out.columns = flattened

    rename = {}

    for col in out.columns:
        name = str(col).strip().lower()

        if name == "open":
            rename[col] = "Open"
        elif name == "high":
            rename[col] = "High"
        elif name == "low":
            rename[col] = "Low"
        elif name == "close":
            rename[col] = "Close"
        elif name == "adj close":
            rename[col] = "Adj Close"
        elif name == "volume":
            rename[col] = "Volume"

    out = out.rename(columns=rename)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        col for col in required
        if col not in out.columns
    ]

    if missing:
        return pd.DataFrame()

    for col in required:
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

    out = out.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    out = out[
        ~out.index.duplicated(
            keep="last"
        )
    ]

    return out.sort_index()


def download_price(symbol: str) -> pd.DataFrame:

    try:

        df = yf.download(
            symbol,
            period=LOOKBACK_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        return clean_download(df)

    except Exception as exc:

        print(
            f"[WARN] {symbol}: "
            f"download failed: {exc}"
        )

        return pd.DataFrame()


def pct(value):
    """
    안전한 퍼센트 변환.
    NaN이면 NaN 반환.
    """

    try:

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value * 100.0

    except Exception:
        return np.nan


def rnd(value, digits=2):
    """
    안전한 round.
    기존 코드의 round(value, condition, digits) 문제 방지.
    """

    try:

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return round(value, digits)

    except Exception:
        return np.nan


def finite_values(values):
    """
    유한한 숫자만 Series로 반환.
    """

    result = []

    for value in values:

        try:

            value = float(value)

            if np.isfinite(value):
                result.append(value)

        except Exception:
            pass

    return pd.Series(result, dtype=float)


# ============================================================
# FEATURES
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:

    x = df.copy()

    close = x["Close"]
    high = x["High"]
    low = x["Low"]
    volume = x["Volume"]

    x["ret1"] = close.pct_change()

    x["ma20"] = close.rolling(20).mean()
    x["ma50"] = close.rolling(50).mean()
    x["ma100"] = close.rolling(100).mean()
    x["ma200"] = close.rolling(200).mean()

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr14"] = true_range.rolling(14).mean()

    x["atr_pct"] = (
        x["atr14"]
        / close.replace(0, np.nan)
    )

    x["vol20"] = (
        x["ret1"].rolling(20).std()
        * np.sqrt(252)
    )

    median_volume = (
        volume
        .rolling(20)
        .median()
        .replace(0, np.nan)
    )

    x["vol_ratio"] = (
        volume / median_volume
    )

    x["ma20_gap"] = (
        close / x["ma20"].replace(0, np.nan)
        - 1
    )

    x["ma50_gap"] = (
        close / x["ma50"].replace(0, np.nan)
        - 1
    )

    x["ma100_gap"] = (
        close / x["ma100"].replace(0, np.nan)
        - 1
    )

    x["ma200_gap"] = (
        close / x["ma200"].replace(0, np.nan)
        - 1
    )

    x["hh20"] = (
        close
        / close.rolling(20).max()
        - 1
    )

    x["ll20"] = (
        close
        / close.rolling(20).min()
        - 1
    )

    x["hh60"] = (
        close
        / close.rolling(60).max()
        - 1
    )

    x["ll60"] = (
        close
        / close.rolling(60).min()
        - 1
    )

    x["slope20"] = (
        x["ma20"].pct_change(10)
    )

    x["slope50"] = (
        x["ma50"].pct_change(10)
    )

    return x


# ============================================================
# SIGNATURE
# ============================================================

def robust_zscore(arr):
    """
    Robust standardization.
    """

    a = np.asarray(
        arr,
        dtype=float,
    )

    if a.size == 0:
        return a

    finite = a[np.isfinite(a)]

    if finite.size == 0:
        return np.zeros_like(a)

    median = np.median(finite)

    mad = np.median(
        np.abs(finite - median)
    )

    if not np.isfinite(mad) or mad < 1e-9:

        std = np.std(finite)

        if not np.isfinite(std) or std < 1e-9:
            return np.zeros_like(a)

        return (
            np.nan_to_num(a, nan=median)
            - np.mean(finite)
        ) / std

    return (
        np.nan_to_num(a, nan=median)
        - median
    ) / (1.4826 * mad)


def safe_interpolate(values, n):

    a = np.asarray(
        values,
        dtype=float,
    )

    if len(a) != n:
        a = np.resize(a, n)

    a = (
        pd.Series(a)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .interpolate(
            limit_direction="both"
        )
        .fillna(0)
        .to_numpy()
    )

    return a


def make_signature(window):

    n = len(window)

    if n < 10:
        return np.zeros(1)

    close = window["Close"].to_numpy(
        dtype=float
    )

    base = (
        close[0]
        if close[0] != 0
        else 1.0
    )

    price_path = (
        close / base
        - 1.0
    )

    # 가격 경로 16개
    price_idx = np.linspace(
        0,
        n - 1,
        16,
    )

    price_part = np.interp(
        price_idx,
        np.arange(n),
        price_path,
    )

    # 거래량 8개
    volume_ratio = (
        window["vol_ratio"]
        .fillna(1)
        .to_numpy(dtype=float)
    )

    volume_ratio = np.clip(
        volume_ratio,
        0,
        100,
    )

    volume_part_raw = np.log1p(
        volume_ratio
    )

    volume_idx = np.linspace(
        0,
        n - 1,
        8,
    )

    volume_part = np.interp(
        volume_idx,
        np.arange(n),
        volume_part_raw,
    )

    # MA 관계
    ma_parts = []

    for col in [
        "ma20_gap",
        "ma50_gap",
        "ma100_gap",
    ]:

        arr = safe_interpolate(
            window[col],
            n,
        )

        idx = np.linspace(
            0,
            n - 1,
            4,
        )

        ma_parts.extend(
            np.interp(
                idx,
                np.arange(n),
                arr,
            )
        )

    # 변동성
    volatility = safe_interpolate(
        window["vol20"],
        n,
    )

    volatility_idx = np.linspace(
        0,
        n - 1,
        4,
    )

    volatility_part = np.interp(
        volatility_idx,
        np.arange(n),
        volatility,
    )

    last = window.iloc[-1]

    scalar = np.array(
        [
            last.get("hh20", 0),
            last.get("ll20", 0),
            last.get("hh60", 0),
            last.get("ll60", 0),
            last.get("slope20", 0),
            last.get("slope50", 0),
            last.get("atr_pct", 0),
        ],
        dtype=float,
    )

    signature = np.concatenate(
        [
            price_part,
            volume_part,
            np.asarray(ma_parts),
            volatility_part,
            scalar,
        ]
    )

    signature = np.nan_to_num(
        signature,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    signature = robust_zscore(
        signature
    )

    norm = np.linalg.norm(
        signature
    )

    if norm < 1e-12:
        return np.zeros_like(
            signature
        )

    return signature / norm


def cosine_similarity(a, b):

    if len(a) != len(b):
        return 0.0

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-12 or nb < 1e-12:
        return 0.0

    result = np.dot(a, b) / (
        na * nb
    )

    return float(
        np.clip(result, -1.0, 1.0)
    )


# ============================================================
# PATTERN LABEL
# ============================================================

def choose_pattern_label(window):

    close = window["Close"]

    start = float(
        close.iloc[0]
    )

    end = float(
        close.iloc[-1]
    )

    if start <= 0:
        return "구조불명"

    total_return = (
        end / start - 1
    )

    half = max(
        len(close) // 2,
        1,
    )

    first_half = close.iloc[:half]
    second_half = close.iloc[half:]

    first_move = (
        first_half.iloc[-1]
        / first_half.iloc[0]
        - 1
    )

    second_move = (
        second_half.iloc[-1]
        / second_half.iloc[0]
        - 1
    )

    peak = float(close.max())
    trough = float(close.min())

    peak_pos = int(
        close.argmax()
    )

    trough_pos = int(
        close.argmin()
    )

    peak_return = (
        peak / start - 1
    )

    trough_return = (
        trough / start - 1
    )

    last = window.iloc[-1]

    ma20_gap = float(
        last.get("ma20_gap", 0)
    )

    ma50_gap = float(
        last.get("ma50_gap", 0)
    )

    slope20 = float(
        last.get("slope20", 0)
    )

    # 급락 → 저점 → 회복
    if (
        trough_return <= -0.18
        and trough_pos < len(close) * 0.80
        and total_return > -0.10
    ):
        return "급락→저점형성→회복"

    # 급등 → 조정 → 재상승
    if (
        peak_return >= 0.20
        and trough_pos > peak_pos
        and second_move > 0.05
    ):
        return "급등→조정→재상승"

    current_vol = float(
        window["vol20"].iloc[-1]
    )

    median_vol = float(
        window["vol20"].median()
    )

    # 압축 → 돌파
    if (
        np.isfinite(current_vol)
        and np.isfinite(median_vol)
        and median_vol > 0
        and current_vol > median_vol * 1.15
        and total_return > 0.10
    ):
        return "압축→돌파→확장"

    # 추세 상승
    if (
        total_return > 0.12
        and ma20_gap > 0
        and ma50_gap > -0.02
        and slope20 > 0
    ):
        return "추세상승→확장"

    # 추세 하락 → 반등
    if (
        total_return < -0.12
        and ma20_gap < 0
        and ma50_gap < 0
        and slope20 < 0
    ):
        return "추세하락→반등시도"

    # 박스
    if abs(total_return) < 0.10:
        return "박스→반복회귀"

    return "추세전환형"


# ============================================================
# FORWARD METRICS
# ============================================================

def pattern_metrics(
    df,
    end,
    forward,
):

    entry = float(
        df["Close"].iloc[end - 1]
    )

    if entry <= 0:
        return {
            "forward_return": np.nan,
            "max_up": np.nan,
            "max_down": np.nan,
            "duration_to_high": np.nan,
        }

    future = df.iloc[
        end:
        min(
            end + forward,
            len(df),
        )
    ]

    if future.empty:
        return {
            "forward_return": np.nan,
            "max_up": np.nan,
            "max_down": np.nan,
            "duration_to_high": np.nan,
        }

    high_return = (
        future["High"]
        / entry
        - 1
    )

    low_return = (
        future["Low"]
        / entry
        - 1
    )

    close_return = (
        future["Close"]
        / entry
        - 1
    )

    max_up = float(
        high_return.max()
    )

    max_down = float(
        low_return.min()
    )

    forward_return = float(
        close_return.iloc[-1]
    )

    high_position = (
        int(high_return.argmax())
        + 1
    )

    return {
        "forward_return": forward_return,
        "max_up": max_up,
        "max_down": max_down,
        "duration_to_high": high_position,
    }


# ============================================================
# EVENT EXTRACTION
# ============================================================

def extract_events(df):

    events = []

    if len(df) < (
        MIN_HISTORY
    ):
        return events

    # 신규 상장 종목은 WINDOW + 약간의 여유만 있으면 허용
    first_end = max(
        WINDOW,
        MIN_HISTORY,
    )

    last_end = (
        len(df) - 2
    )

    if last_end <= first_end:
        return events

    for end in range(
        first_end,
        last_end + 1,
    ):

        window = df.iloc[
            end - WINDOW:
            end
        ].copy()

        if len(window) < WINDOW:
            continue

        if window["Close"].isna().any():
            continue

        signature = make_signature(
            window
        )

        if (
            signature.size == 0
            or not np.isfinite(
                signature
            ).all()
        ):
            continue

        metrics = pattern_metrics(
            df,
            end,
            FORWARD,
        )

        events.append(
            {
                "end_idx": end,
                "date": df.index[end - 1],
                "signature": signature,
                "label": choose_pattern_label(
                    window
                ),
                **metrics,
            }
        )

    # 가까운 이벤트를 줄여 과도한 중복 방지
    selected = []

    last_idx = -10**9

    for event in events:

        if (
            event["end_idx"]
            - last_idx
            >= EVENT_SPACING
        ):
            selected.append(event)
            last_idx = event["end_idx"]

    return selected


# ============================================================
# CLUSTERING
# ============================================================

def cluster_patterns(
    events,
    threshold,
):

    clusters = []

    for event in events:

        signature = event[
            "signature"
        ]

        best_idx = None
        best_similarity = -1.0

        for idx, cluster in enumerate(
            clusters
        ):

            similarity = cosine_similarity(
                signature,
                cluster["centroid"],
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx

        if (
            best_idx is not None
            and best_similarity >= threshold
        ):

            cluster = clusters[
                best_idx
            ]

            cluster["events"].append(
                event
            )

            centroid = np.mean(
                [
                    x["signature"]
                    for x in cluster["events"]
                ],
                axis=0,
            )

            norm = np.linalg.norm(
                centroid
            )

            if norm > 1e-12:
                centroid = (
                    centroid / norm
                )

            cluster[
                "centroid"
            ] = centroid

        else:

            clusters.append(
                {
                    "centroid": signature.copy(),
                    "events": [event],
                }
            )

    return clusters


# ============================================================
# CYCLE
# ============================================================

def cycle_statistics(events):

    if len(events) < 2:
        return np.nan, np.nan, np.nan

    dates = sorted(
        pd.Timestamp(
            event["date"]
        )
        for event in events
    )

    gaps = []

    for previous, current in zip(
        dates[:-1],
        dates[1:],
    ):

        gap = (
            current - previous
        ).days

        if gap > 0:
            gaps.append(gap)

    if not gaps:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    return (
        float(np.median(gaps)),
        float(np.mean(gaps)),
        float(np.std(gaps)),
    )


# ============================================================
# CURRENT PHASE
# ============================================================

def current_phase(df):

    if len(df) < 60:
        return "INSUFFICIENT_HISTORY"

    last = df.iloc[-1]

    close = float(
        last["Close"]
    )

    ma20 = float(
        last["ma20"]
    ) if np.isfinite(
        last["ma20"]
    ) else np.nan

    ma50 = float(
        last["ma50"]
    ) if np.isfinite(
        last["ma50"]
    ) else np.nan

    slope20 = float(
        last.get(
            "slope20",
            0,
        )
    )

    volume_ratio = float(
        last.get(
            "vol_ratio",
            1,
        )
    )

    recent_high = float(
        df["High"].tail(20).max()
    )

    recent_low = float(
        df["Low"].tail(20).min()
    )

    if (
        close >= recent_high * 0.995
        and volume_ratio >= 1.25
    ):
        return "BREAKOUT"

    if (
        close <= recent_low * 1.005
    ):
        return "BREAKDOWN"

    if (
        np.isfinite(ma20)
        and np.isfinite(ma50)
        and close > ma20 > ma50
        and slope20 > 0
    ):
        return "UPTREND"

    if (
        np.isfinite(ma20)
        and np.isfinite(ma50)
        and close < ma20 < ma50
        and slope20 < 0
    ):
        return "DOWNTREND"

    if (
        recent_high > 0
        and close > 0
        and (
            recent_high
            - recent_low
        ) / close < 0.12
    ):
        return "COMPRESSION"

    return "TRANSITION"


# ============================================================
# TARGET / INVALIDATION
# ============================================================

def historical_levels(
    cluster_events,
    current_price,
):

    ups = finite_values(
        event["max_up"]
        for event in cluster_events
    )

    downs = finite_values(
        event["max_down"]
        for event in cluster_events
    )

    if ups.empty:

        target1 = (
            current_price * 1.05
        )

        target2 = (
            current_price * 1.08
        )

    else:

        up50 = float(
            ups.quantile(0.50)
        )

        up75 = float(
            ups.quantile(0.75)
        )

        target1 = (
            current_price
            * (
                1
                + max(
                    0.03,
                    up50 * 0.60,
                )
            )
        )

        target2 = (
            current_price
            * (
                1
                + max(
                    0.05,
                    up75 * 0.70,
                )
            )
        )

    if downs.empty:

        invalidation = (
            current_price * 0.95
        )

    else:

        down25 = float(
            downs.quantile(0.25)
        )

        invalidation = (
            current_price
            * (
                1
                + min(
                    -0.03,
                    down25 * 0.75,
                )
            )
        )

    return (
        target1,
        target2,
        invalidation,
    )


# ============================================================
# DRIVER COUNT
# ============================================================

def allowed_driver_count(
    history_length,
):

    if history_length < 120:
        return 1

    if history_length < 300:
        return min(
            2,
            MAX_DRIVERS,
        )

    return MAX_DRIVERS


# ============================================================
# SYMBOL ANALYSIS
# ============================================================

def analyze_symbol(symbol):

    symbol = (
        str(symbol)
        .strip()
        .upper()
    )

    print()
    print(
        f"===== DRIVER: {symbol} ====="
    )

    raw = download_price(
        symbol
    )

    if raw.empty:

        print(
            f"[SKIP] {symbol}: "
            "no price data"
        )

        return [], [], []

    df = add_features(
        raw
    )

    usable = df.dropna(
        subset=["Close"]
    ).copy()

    history_length = len(
        usable
    )

    if history_length < MIN_HISTORY:

        print(
            f"[SKIP] {symbol}: "
            f"history {history_length} "
            f"< minimum {MIN_HISTORY}"
        )

        return [], [], []

    events = extract_events(
        usable
    )

    if len(events) < 2:

        print(
            f"[SKIP] {symbol}: "
            f"not enough pattern events "
            f"({len(events)})"
        )

        return [], [], []

    clusters = cluster_patterns(
        events,
        CLUSTER_SIM,
    )

    valid_clusters = [
        cluster
        for cluster in clusters
        if len(cluster["events"]) >= 2
    ]

    if not valid_clusters:

        print(
            f"[SKIP] {symbol}: "
            "no recurring Driver found"
        )

        return [], [], []

    # 반복 횟수 우선.
    # 같은 횟수라면 평균 미래수익을 보조 기준으로 사용.
    def cluster_sort_key(cluster):

        returns = finite_values(
            event["forward_return"]
            for event in cluster["events"]
        )

        mean_return = (
            float(returns.mean())
            if not returns.empty
            else -999.0
        )

        return (
            len(cluster["events"]),
            mean_return,
        )

    valid_clusters.sort(
        key=cluster_sort_key,
        reverse=True,
    )

    max_allowed = min(
        allowed_driver_count(
            history_length
        ),
        MAX_DRIVERS,
    )

    selected = valid_clusters[
        :max_allowed
    ]

    current_window = usable.iloc[
        -WINDOW:
    ].copy()

    if len(current_window) < WINDOW:

        print(
            f"[SKIP] {symbol}: "
            "current window unavailable"
        )

        return [], [], []

    current_signature = make_signature(
        current_window
    )

    current_price = float(
        usable["Close"].iloc[-1]
    )

    phase = current_phase(
        usable
    )

    profiles = []
    current_rows = []
    history_rows = []

    # --------------------------------------------------------
    # DRIVER PROFILE
    # --------------------------------------------------------

    for cluster in selected:

        cluster_events = (
            cluster["events"]
        )

        similarities = [
            cosine_similarity(
                current_signature,
                event["signature"],
            )
            for event in cluster_events
        ]

        current_similarity = (
            max(similarities)
            if similarities
            else 0.0
        )

        forward_returns = finite_values(
            event["forward_return"]
            for event in cluster_events
        )

        max_ups = finite_values(
            event["max_up"]
            for event in cluster_events
        )

        max_downs = finite_values(
            event["max_down"]
            for event in cluster_events
        )

        # 성공:
        # Driver 발생 이후 40거래일 안에
        # 고가 기준 +5% 이상 도달.
        success_flags = []

        for event in cluster_events:

            max_up = event["max_up"]

            success_flags.append(
                bool(
                    np.isfinite(max_up)
                    and max_up >= 0.05
                )
            )

        occurrence_count = len(
            cluster_events
        )

        success_count = int(
            sum(success_flags)
        )

        success_rate = (
            success_count
            / occurrence_count
            if occurrence_count
            else np.nan
        )

        median_cycle_days, mean_cycle_days, cycle_std_days = (
            cycle_statistics(
                cluster_events
            )
        )

        target1, target2, invalidation = (
            historical_levels(
                cluster_events,
                current_price,
            )
        )

        if occurrence_count < 2:

            state = "INACTIVE"

        elif (
            current_similarity
            >= CURRENT_ACTIVE_SIM
        ):

            state = "ACTIVE"

        elif (
            current_similarity
            >= CURRENT_WATCH_SIM
        ):

            state = "WATCH"

        else:

            state = "INACTIVE"

        labels = [
            event["label"]
            for event in cluster_events
        ]

        label_mode = (
            pd.Series(labels)
            .mode()
        )

        pattern = (
            str(label_mode.iloc[0])
            if not label_mode.empty
            else "구조불명"
        )

        dates = sorted(
            pd.Timestamp(
                event["date"]
            )
            for event in cluster_events
        )

        nearest_event = max(
            cluster_events,
            key=lambda event:
                cosine_similarity(
                    current_signature,
                    event["signature"],
                ),
        )

        nearest_similarity = cosine_similarity(
            current_signature,
            nearest_event["signature"],
        )

        # 발생시점의 평균/중앙값 구조
        event_returns = finite_values(
            (
                event["forward_return"]
                for event in cluster_events
            )
        )

        profile = {
            "symbol": symbol,
            "driver_name": None,
            "pattern": pattern,
            "state": state,
            "phase": phase,
            "history_days": history_length,
            "current_price": rnd(
                current_price,
                4,
            ),
            "current_similarity_pct": rnd(
                current_similarity * 100,
                2,
            ),
            "occurrences": occurrence_count,
            "success_count": success_count,
            "success_rate_pct": rnd(
                success_rate * 100,
                2,
            ),
            "avg_forward_return_pct": rnd(
                event_returns.mean() * 100
                if not event_returns.empty
                else np.nan,
                2,
            ),
            "median_forward_return_pct": rnd(
                event_returns.median() * 100
                if not event_returns.empty
                else np.nan,
                2,
            ),
            "median_max_up_pct": rnd(
                max_ups.median() * 100
                if not max_ups.empty
                else np.nan,
                2,
            ),
            "median_max_down_pct": rnd(
                max_downs.median() * 100
                if not max_downs.empty
                else np.nan,
                2,
            ),
            "median_cycle_days": rnd(
                median_cycle_days,
                1,
            ),
            "mean_cycle_days": rnd(
                mean_cycle_days,
                1,
            ),
            "cycle_std_days": rnd(
                cycle_std_days,
                1,
            ),
            "target1": rnd(
                target1,
                4,
            ),
            "target2": rnd(
                target2,
                4,
            ),
            "invalidation": rnd(
                invalidation,
                4,
            ),
            "first_seen": (
                dates[0]
                .date()
                .isoformat()
            ),
            "last_seen": (
                dates[-1]
                .date()
                .isoformat()
            ),
            "nearest_historical_similarity_pct": rnd(
                nearest_similarity * 100,
                2,
            ),
        }

        profiles.append(
            profile
        )

        for event in cluster_events:

            history_rows.append(
                {
                    "symbol": symbol,
                    "pattern": pattern,
                    "event_date": (
                        pd.Timestamp(
                            event["date"]
                        )
                        .date()
                        .isoformat()
                    ),
                    "forward_return_pct": rnd(
                        pct(
                            event[
                                "forward_return"
                            ]
                        ),
                        2,
                    ),
                    "max_up_pct": rnd(
                        pct(
                            event["max_up"]
                        ),
                        2,
                    ),
                    "max_down_pct": rnd(
                        pct(
                            event["max_down"]
                        ),
                        2,
                    ),
                    "duration_to_high_days": event[
                        "duration_to_high"
                    ],
                    "label_at_event": event[
                        "label"
                    ],
                }
            )

    # --------------------------------------------------------
    # 현재 유사도가 높은 순으로 정렬
    # --------------------------------------------------------

    profiles.sort(
        key=lambda item: (
            {
                "ACTIVE": 0,
                "WATCH": 1,
                "INACTIVE": 2,
            }.get(
                item["state"],
                9,
            ),
            -float(
                item[
                    "current_similarity_pct"
                ]
            ),
            -int(
                item["occurrences"]
            ),
        )
    )

    # 최종 Driver 번호 부여
    for rank, profile in enumerate(
        profiles,
        start=1,
    ):

        profile["driver_rank"] = rank
        profile["driver_name"] = (
            f"DRIVER_{rank}"
        )

    # history에 최종 Driver 연결
    for history in history_rows:

        matching = [
            profile
            for profile in profiles
            if profile["pattern"]
            == history["pattern"]
        ]

        if matching:

            selected_profile = max(
                matching,
                key=lambda item:
                    item["occurrences"],
            )

            history["driver_rank"] = (
                selected_profile[
                    "driver_rank"
                ]
            )

            history["driver_name"] = (
                selected_profile[
                    "driver_name"
                ]
            )

    # 현재 최상위 Driver
    if profiles:

        top = profiles[0]

        current_rows.append(
            {
                "symbol": symbol,
                "as_of": (
                    usable.index[-1]
                    .date()
                    .isoformat()
                ),
                "current_price": top[
                    "current_price"
                ],
                "active_driver": top[
                    "driver_name"
                ],
                "pattern": top[
                    "pattern"
                ],
                "state": top[
                    "state"
                ],
                "phase": top[
                    "phase"
                ],
                "similarity_pct": top[
                    "current_similarity_pct"
                ],
                "success_rate_pct": top[
                    "success_rate_pct"
                ],
                "target1": top[
                    "target1"
                ],
                "target2": top[
                    "target2"
                ],
                "invalidation": top[
                    "invalidation"
                ],
                "occurrences": top[
                    "occurrences"
                ],
                "median_cycle_days": top[
                    "median_cycle_days"
                ],
            }
        )

    print(
        f"[OK] {symbol}: "
        f"history={history_length} "
        f"events={len(events)} "
        f"drivers={len(profiles)} "
        f"phase={phase} "
        f"price={current_price:.2f}"
    )

    for profile in profiles:

        print(
            "  "
            f"{profile['driver_name']} | "
            f"{profile['pattern']} | "
            f"{profile['state']} | "
            f"sim={profile['current_similarity_pct']:.1f}% | "
            f"success={profile['success_rate_pct']:.1f}% | "
            f"cycle={profile['median_cycle_days']}d | "
            f"T1={profile['target1']:.2f} | "
            f"T2={profile['target2']:.2f} | "
            f"INV={profile['invalidation']:.2f}"
        )

    return (
        profiles,
        current_rows,
        history_rows,
    )


# ============================================================
# SYMBOL LOADING
# ============================================================

def load_symbols_from_project():

    # CLI
    cli_symbols = [
        str(value)
        .strip()
        .upper()
        for value in sys.argv[1:]
        if str(value).strip()
    ]

    if cli_symbols:

        return list(
            dict.fromkeys(
                cli_symbols
            )
        )

    # 환경변수
    env_symbols = os.getenv(
        "DRIVER_SYMBOLS",
        "",
    )

    if env_symbols.strip():

        return list(
            dict.fromkeys(
                value.strip().upper()
                for value in env_symbols.split(",")
                if value.strip()
            )
        )

    # 기존 selected_symbols.py
    try:

        from selected_symbols import (
            SELECTED_SYMBOLS
        )

        symbols = [
            str(value)
            .strip()
            .upper()
            for value in SELECTED_SYMBOLS
            if str(value).strip()
        ]

        if symbols:

            return list(
                dict.fromkeys(
                    symbols
                )
            )

    except Exception:
        pass

    # 프로젝트에 symbol 파일이 없을 경우
    return [
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        "NVDA",
        "TSLA",
        "META",
        "AMZN",
        "GOOGL",
        "MSFT",
        "AAPL",
        "AVGO",
        "ORCL",
        "AMD",
        "RKLB",
        "UBER",
        "DKNG",
        "OKLO",
        "IREN",
        "NVTS",
        "CBRS",
    ]


# ============================================================
# OUTPUT
# ============================================================

def save_outputs(
    profiles,
    current_rows,
    history_rows,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_df = pd.DataFrame(
        profiles
    )

    current_df = pd.DataFrame(
        current_rows
    )

    history_df = pd.DataFrame(
        history_rows
    )

    profile_df.to_csv(
        PROFILE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    current_df.to_csv(
        CURRENT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    history_df.to_csv(
        HISTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "===== OUTPUT ====="
    )

    print(
        f"profile : {PROFILE_FILE}"
    )

    print(
        f"current : {CURRENT_FILE}"
    )

    print(
        f"history : {HISTORY_FILE}"
    )

    print(
        "rows    : "
        f"profile={len(profile_df)}, "
        f"current={len(current_df)}, "
        f"history={len(history_df)}"
    )


# ============================================================
# MAIN
# ============================================================

def run(symbols):

    symbols = list(
        dict.fromkeys(
            str(symbol)
            .strip()
            .upper()
            for symbol in symbols
            if str(symbol).strip()
        )
    )

    print(
        "=============================================="
    )

    print(
        " UNIVERSAL DRIVER ENGINE"
    )

    print(
        "=============================================="
    )

    print(
        f"symbols       : {len(symbols)}"
    )

    print(
        f"lookback      : {LOOKBACK_PERIOD}"
    )

    print(
        f"window        : {WINDOW}"
    )

    print(
        f"forward       : {FORWARD}"
    )

    print(
        f"min_history   : {MIN_HISTORY}"
    )

    print(
        f"max_drivers   : {MAX_DRIVERS}"
    )

    print(
        f"cluster_sim   : {CLUSTER_SIM}"
    )

    print(
        "=============================================="
    )

    all_profiles = []
    all_current = []
    all_history = []

    for symbol in symbols:

        try:

            profiles, current, history = (
                analyze_symbol(
                    symbol
                )
            )

            all_profiles.extend(
                profiles
            )

            all_current.extend(
                current
            )

            all_history.extend(
                history
            )

        except Exception as exc:

            print(
                f"[ERROR] {symbol}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    save_outputs(
        all_profiles,
        all_current,
        all_history,
    )

    if all_current:

        print()
        print(
            "===== CURRENT DRIVER SUMMARY ====="
        )

        print(
            pd.DataFrame(
                all_current
            ).to_string(
                index=False
            )
        )

    else:

        print()
        print(
            "[WARN] "
            "No current driver detected."
        )


if __name__ == "__main__":

    symbols = (
        load_symbols_from_project()
    )

    run(symbols)

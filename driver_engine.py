#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNIVERSAL DRIVER ENGINE v6.4
============================

목적
----
종목별 OHLCV 역사에서 반복되는 "운전수(Driver)" 구조를 탐색한다.

핵심 원칙
----------
1. 미래 FORWARD 기간이 완전히 존재하는 이벤트만 학습한다.
2. Pattern similarity와 Cycle recurrence를 분리한다.
3. cycle의 1차 증거는 반드시 인접 recurrence gap이다.
4. pairwise gap은 보조 증거일 뿐 cycle을 새로 만들 수 없다.
5. 1x / 2x / 3x harmonic recurrence를 하나의 cycle로 통합한다.
6. 3개 이벤트는 "반복 발견"이지 강한 검증이 아니다.
7. cycle stability와 cycle confidence를 분리한다.
8. 결과 재현성과 cycle 재현성을 분리한다.
9. 최근 재현성이 약하면 validation을 낮춘다.
10. 현재 패턴 일치도와 과거 결과 검증을 분리한다.
11. 서로 다른 cycle 배수 때문에 같은 Driver가 중복 생성되지 않는다.
12. 서로 다른 구조만 별도의 Driver로 남긴다.
13. MAX_DRIVERS는 상한이며 3개를 강제로 생성하지 않는다.
14. AMBIGUOUS / NO_HIT은 성공으로 취급하지 않는다.
15. 소표본 100% 적중을 그대로 100% 신뢰하지 않는다.
16. look-ahead bias를 방지한다.

출력
-----
03_RESULTS/daily/driver_profile.csv
03_RESULTS/daily/driver_current.csv
03_RESULTS/daily/driver_history.csv
"""

import hashlib
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

LOOKBACK_PERIOD = os.getenv(
    "DRIVER_LOOKBACK",
    "10y",
)

WINDOW = int(
    os.getenv(
        "DRIVER_WINDOW",
        "40",
    )
)

FORWARD = int(
    os.getenv(
        "DRIVER_FORWARD",
        "40",
    )
)

MIN_HISTORY = int(
    os.getenv(
        "DRIVER_MIN_HISTORY",
        "80",
    )
)

MAX_DRIVERS = int(
    os.getenv(
        "DRIVER_MAX_DRIVERS",
        "3",
    )
)

CURRENT_ACTIVE_SIM = float(
    os.getenv(
        "DRIVER_ACTIVE_SIM",
        "0.82",
    )
)

CURRENT_WATCH_SIM = float(
    os.getenv(
        "DRIVER_WATCH_SIM",
        "0.72",
    )
)

CLUSTER_SIM = float(
    os.getenv(
        "DRIVER_CLUSTER_SIM",
        "0.90",
    )
)

EVENT_SPACING = int(
    os.getenv(
        "DRIVER_EVENT_SPACING",
        "40",
    )
)

CYCLE_MIN_DAYS = int(
    os.getenv(
        "DRIVER_CYCLE_MIN_DAYS",
        "25",
    )
)

HARMONIC_TOLERANCE = float(
    os.getenv(
        "DRIVER_HARMONIC_TOLERANCE",
        "0.25",
    )
)

MAX_CYCLE_MULTIPLE = int(
    os.getenv(
        "DRIVER_MAX_CYCLE_MULTIPLE",
        "3",
    )
)

MIN_DRIVER_OCCURRENCES = int(
    os.getenv(
        "DRIVER_MIN_OCCURRENCES",
        "3",
    )
)

VALIDATED_OCCURRENCES = int(
    os.getenv(
        "DRIVER_VALIDATED_OCCURRENCES",
        "4",
    )
)

STABLE_OCCURRENCES = int(
    os.getenv(
        "DRIVER_STABLE_OCCURRENCES",
        "6",
    )
)

RECENT_OCCURRENCES = int(
    os.getenv(
        "DRIVER_RECENT_OCCURRENCES",
        "5",
    )
)

MIN_CYCLE_STABILITY = float(
    os.getenv(
        "DRIVER_MIN_CYCLE_STABILITY",
        "0.45",
    )
)

MIN_CYCLE_OBSERVATIONS = int(
    os.getenv(
        "DRIVER_MIN_CYCLE_OBSERVATIONS",
        "2",
    )
)

MIN_RESULT_REPRODUCIBILITY = float(
    os.getenv(
        "DRIVER_MIN_RESULT_REPRO",
        "0.45",
    )
)

MIN_RECENT_REPRODUCIBILITY = float(
    os.getenv(
        "DRIVER_MIN_RECENT_REPRO",
        "0.40",
    )
)

MIN_RESOLVED_RESULTS = int(
    os.getenv(
        "DRIVER_MIN_RESOLVED_RESULTS",
        "2",
    )
)

DUPLICATE_SIM = float(
    os.getenv(
        "DRIVER_DUPLICATE_SIM",
        "0.97",
    )
)

DUPLICATE_CYCLE_RATIO = float(
    os.getenv(
        "DRIVER_DUPLICATE_CYCLE_RATIO",
        "1.35",
    )
)

CYCLE_CONFIDENCE_FLOOR = float(
    os.getenv(
        "DRIVER_CYCLE_CONFIDENCE_FLOOR",
        "0.45",
    )
)


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def rnd(value, digits=2):
    value = safe_float(value)

    if not np.isfinite(value):
        return np.nan

    return round(value, digits)


def pct(value, digits=2):
    value = safe_float(value)

    if not np.isfinite(value):
        return np.nan

    return round(value * 100.0, digits)


def clamp01(value, default=0.0):
    value = safe_float(value, default)

    if not np.isfinite(value):
        return default

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def finite_values(values):
    result = []

    for value in values:
        value = safe_float(value)

        if np.isfinite(value):
            result.append(value)

    return pd.Series(
        result,
        dtype=float,
    )


def safe_mean(values):
    values = finite_values(values)

    if values.empty:
        return np.nan

    return float(values.mean())


def safe_median(values):
    values = finite_values(values)

    if values.empty:
        return np.nan

    return float(values.median())


def safe_std(values):
    values = finite_values(values)

    if len(values) < 2:
        return np.nan

    return float(values.std(ddof=0))


def date_string(value):
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return str(value)


# ============================================================
# DATA
# ============================================================

def clean_download(df):
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        flattened = []

        for column in out.columns:
            found = None

            for item in column:
                name = str(item)

                if name in {
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume",
                }:
                    found = name
                    break

            flattened.append(
                found
                if found
                else str(column[0])
            )

        out.columns = flattened

    rename = {}

    for column in out.columns:
        name = str(column).strip().lower()

        if name == "open":
            rename[column] = "Open"

        elif name == "high":
            rename[column] = "High"

        elif name == "low":
            rename[column] = "Low"

        elif name == "close":
            rename[column] = "Close"

        elif name == "adj close":
            rename[column] = "Adj Close"

        elif name == "volume":
            rename[column] = "Volume"

    out = out.rename(
        columns=rename
    )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if any(
        column not in out.columns
        for column in required
    ):
        return pd.DataFrame()

    for column in required:
        out[column] = pd.to_numeric(
            out[column],
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


def download_price(symbol):
    try:
        data = yf.download(
            symbol,
            period=LOOKBACK_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        return clean_download(data)

    except Exception as exc:
        print(
            f"[WARN] {symbol}: "
            f"download failed: {exc}"
        )

        return pd.DataFrame()


# ============================================================
# FEATURES
# ============================================================

def add_features(df):
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

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr14"] = tr.rolling(14).mean()

    x["atr_pct"] = (
        x["atr14"]
        / close.replace(
            0,
            np.nan,
        )
    )

    x["vol20"] = (
        x["ret1"].rolling(20).std()
        * np.sqrt(252)
    )

    median_volume = (
        volume
        .rolling(20)
        .median()
        .replace(
            0,
            np.nan,
        )
    )

    x["vol_ratio"] = (
        volume
        / median_volume
    )

    for ma in [
        "ma20",
        "ma50",
        "ma100",
        "ma200",
    ]:
        x[f"{ma}_gap"] = (
            close
            / x[ma].replace(
                0,
                np.nan,
            )
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
# NORMALIZATION
# ============================================================

def robust_zscore(values):
    array = np.asarray(
        values,
        dtype=float,
    )

    if array.size == 0:
        return array

    finite = array[
        np.isfinite(array)
    ]

    if finite.size == 0:
        return np.zeros_like(array)

    median = np.median(finite)

    mad = np.median(
        np.abs(
            finite - median
        )
    )

    filled = np.nan_to_num(
        array,
        nan=median,
    )

    if (
        not np.isfinite(mad)
        or mad < 1e-9
    ):
        std = np.std(finite)

        if (
            not np.isfinite(std)
            or std < 1e-9
        ):
            return np.zeros_like(
                array
            )

        return (
            filled
            - np.mean(finite)
        ) / std

    return (
        filled
        - median
    ) / (
        1.4826 * mad
    )


def safe_interpolate(values, length):
    array = np.asarray(
        values,
        dtype=float,
    )

    if len(array) != length:
        array = np.resize(
            array,
            length,
        )

    series = pd.Series(
        array
    )

    return (
        series
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .interpolate(
            limit_direction="both"
        )
        .fillna(0)
        .to_numpy()
    )


# ============================================================
# SIGNATURE
# ============================================================

def make_signature(window):
    n = len(window)

    if n < 10:
        return np.zeros(1)

    close = window[
        "Close"
    ].to_numpy(
        dtype=float
    )

    base = (
        close[0]
        if close[0] != 0
        else 1.0
    )

    price_path = (
        close / base - 1
    )

    price_index = np.linspace(
        0,
        n - 1,
        20,
    )

    price_part = np.interp(
        price_index,
        np.arange(n),
        price_path,
    )

    volume_ratio = (
        window["vol_ratio"]
        .fillna(1)
        .to_numpy(
            dtype=float
        )
    )

    volume_ratio = np.clip(
        volume_ratio,
        0,
        100,
    )

    volume_raw = np.log1p(
        volume_ratio
    )

    volume_part = np.interp(
        np.linspace(
            0,
            n - 1,
            8,
        ),
        np.arange(n),
        volume_raw,
    )

    ma_parts = []

    for column in [
        "ma20_gap",
        "ma50_gap",
        "ma100_gap",
    ]:
        values = safe_interpolate(
            window[column],
            n,
        )

        ma_parts.extend(
            np.interp(
                np.linspace(
                    0,
                    n - 1,
                    4,
                ),
                np.arange(n),
                values,
            )
        )

    volatility = safe_interpolate(
        window["vol20"],
        n,
    )

    volatility_part = np.interp(
        np.linspace(
            0,
            n - 1,
            4,
        ),
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
            np.asarray(
                ma_parts
            ),
            volatility_part,
            scalar,
        ]
    )

    signature = np.nan_to_num(
        signature,
        nan=0,
        posinf=0,
        neginf=0,
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

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if (
        norm_a < 1e-12
        or norm_b < 1e-12
    ):
        return 0.0

    value = (
        np.dot(a, b)
        / (
            norm_a
            * norm_b
        )
    )

    return float(
        np.clip(
            value,
            -1,
            1,
        )
    )


# ============================================================
# PATTERN
# ============================================================

def choose_pattern_label(window):
    close = window["Close"]

    start = safe_float(
        close.iloc[0]
    )

    end = safe_float(
        close.iloc[-1]
    )

    if (
        not np.isfinite(start)
        or start <= 0
    ):
        return "구조불명"

    total = (
        end / start - 1
    )

    half = max(
        len(close) // 2,
        1,
    )

    first = close.iloc[:half]
    second = close.iloc[half:]

    second_move = (
        second.iloc[-1]
        / second.iloc[0]
        - 1
    )

    peak = float(
        close.max()
    )

    trough = float(
        close.min()
    )

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

    ma20_gap = safe_float(
        last.get(
            "ma20_gap",
            0,
        ),
        0,
    )

    ma50_gap = safe_float(
        last.get(
            "ma50_gap",
            0,
        ),
        0,
    )

    slope20 = safe_float(
        last.get(
            "slope20",
            0,
        ),
        0,
    )

    current_vol = safe_float(
        window["vol20"].iloc[-1]
    )

    median_vol = safe_float(
        window["vol20"].median()
    )

    if (
        trough_return <= -0.18
        and trough_pos < len(close) * 0.80
        and total > -0.10
    ):
        return "급락→저점형성→회복"

    if (
        peak_return >= 0.20
        and trough_pos > peak_pos
        and second_move > 0.05
    ):
        return "급등→조정→재상승"

    if (
        np.isfinite(current_vol)
        and np.isfinite(median_vol)
        and median_vol > 0
        and current_vol > median_vol * 1.15
        and total > 0.10
    ):
        return "압축→돌파→확장"

    if (
        total > 0.12
        and ma20_gap > 0
        and ma50_gap > -0.02
        and slope20 > 0
    ):
        return "추세상승→확장"

    if (
        total < -0.12
        and ma20_gap < 0
        and ma50_gap < 0
        and slope20 < 0
    ):
        return "추세하락→반등시도"

    if abs(total) < 0.10:
        return "박스→반복회귀"

    return "추세전환형"


# ============================================================
# ORDERED OUTCOME
# ============================================================

def empty_ordered_outcome():
    return {
        "first_event": "NO_FUTURE",
        "first_day": np.nan,
        "plus5_first": np.nan,
        "plus10_first": np.nan,
        "plus20_first": np.nan,
        "minus5_first": np.nan,
        "days_to_plus5": np.nan,
        "days_to_plus10": np.nan,
        "days_to_plus20": np.nan,
        "days_to_minus5": np.nan,
        "outcome_status": "NO_FUTURE",
    }


def ordered_outcome(
    df,
    end_idx,
    forward=FORWARD,
):
    if (
        end_idx < 0
        or end_idx >= len(df) - 1
    ):
        return empty_ordered_outcome()

    entry = safe_float(
        df["Close"].iloc[end_idx]
    )

    if (
        not np.isfinite(entry)
        or entry <= 0
    ):
        return empty_ordered_outcome()

    # 반드시 완전한 forward window만 사용.
    if (
        end_idx
        + forward
        >= len(df)
    ):
        return empty_ordered_outcome()

    target5 = entry * 1.05
    target10 = entry * 1.10
    target20 = entry * 1.20
    stop5 = entry * 0.95

    first_plus5 = None
    first_plus10 = None
    first_plus20 = None
    first_minus5 = None

    for j in range(
        end_idx + 1,
        end_idx + 1 + forward,
    ):
        day = j - end_idx

        high = safe_float(
            df["High"].iloc[j]
        )

        low = safe_float(
            df["Low"].iloc[j]
        )

        if (
            first_plus5 is None
            and np.isfinite(high)
            and high >= target5
        ):
            first_plus5 = day

        if (
            first_plus10 is None
            and np.isfinite(high)
            and high >= target10
        ):
            first_plus10 = day

        if (
            first_plus20 is None
            and np.isfinite(high)
            and high >= target20
        ):
            first_plus20 = day

        if (
            first_minus5 is None
            and np.isfinite(low)
            and low <= stop5
        ):
            first_minus5 = day

    def positive_first(
        positive_day,
        negative_day,
    ):
        if positive_day is None:
            return 0.0

        if negative_day is None:
            return 1.0

        if positive_day < negative_day:
            return 1.0

        if positive_day > negative_day:
            return 0.0

        return np.nan

    def negative_first(
        negative_day,
        positive_day,
    ):
        if negative_day is None:
            return 0.0

        if positive_day is None:
            return 1.0

        if negative_day < positive_day:
            return 1.0

        if negative_day > positive_day:
            return 0.0

        return np.nan

    candidates = []

    if first_plus5 is not None:
        candidates.append(
            ("PLUS_5", first_plus5)
        )

    if first_minus5 is not None:
        candidates.append(
            ("MINUS_5", first_minus5)
        )

    if candidates:
        candidates.sort(
            key=lambda item: item[1]
        )

        if (
            len(candidates) >= 2
            and candidates[0][1]
            == candidates[1][1]
        ):
            first_event = "AMBIGUOUS"
            first_day = candidates[0][1]

        else:
            first_event = candidates[0][0]
            first_day = candidates[0][1]

    else:
        first_event = "NO_HIT"
        first_day = np.nan

    if first_event == "AMBIGUOUS":
        outcome_status = "AMBIGUOUS"

    elif first_event in {
        "PLUS_5",
        "MINUS_5",
    }:
        outcome_status = "RESOLVED"

    else:
        outcome_status = "NO_HIT"

    return {
        "first_event": first_event,
        "first_day": first_day,
        "plus5_first": positive_first(
            first_plus5,
            first_minus5,
        ),
        "plus10_first": positive_first(
            first_plus10,
            first_minus5,
        ),
        "plus20_first": positive_first(
            first_plus20,
            first_minus5,
        ),
        "minus5_first": negative_first(
            first_minus5,
            first_plus5,
        ),
        "days_to_plus5": (
            first_plus5
            if first_plus5 is not None
            else np.nan
        ),
        "days_to_plus10": (
            first_plus10
            if first_plus10 is not None
            else np.nan
        ),
        "days_to_plus20": (
            first_plus20
            if first_plus20 is not None
            else np.nan
        ),
        "days_to_minus5": (
            first_minus5
            if first_minus5 is not None
            else np.nan
        ),
        "outcome_status": outcome_status,
    }


# ============================================================
# EVENT
# ============================================================

def build_event(
    df,
    end_idx,
):
    start_idx = (
        end_idx
        - WINDOW
        + 1
    )

    if start_idx < 0:
        return None

    if (
        end_idx
        + FORWARD
        >= len(df)
    ):
        return None

    window = df.iloc[
        start_idx:
        end_idx + 1
    ]

    if len(window) < WINDOW:
        return None

    ordered = ordered_outcome(
        df,
        end_idx,
        FORWARD,
    )

    if (
        ordered["outcome_status"]
        == "NO_FUTURE"
    ):
        return None

    future = df.iloc[
        end_idx + 1:
        end_idx + 1 + FORWARD
    ]

    if len(future) < FORWARD:
        return None

    entry = safe_float(
        df["Close"].iloc[end_idx]
    )

    high_ret = (
        future["High"]
        / entry
        - 1
    )

    low_ret = (
        future["Low"]
        / entry
        - 1
    )

    close_ret = (
        future["Close"]
        / entry
        - 1
    )

    return {
        "date": df.index[end_idx],
        "end_idx": int(end_idx),
        "price": entry,
        "pattern": choose_pattern_label(
            window
        ),
        "signature": make_signature(
            window
        ),
        "forward_return": safe_float(
            close_ret.iloc[-1]
        ),
        "max_up": safe_float(
            high_ret.max()
        ),
        "max_down": safe_float(
            low_ret.min()
        ),
        "ordered": ordered,
    }


def extract_events(df):
    events = []

    first_end = max(
        WINDOW,
        MIN_HISTORY,
    )

    last_end = (
        len(df)
        - 1
        - FORWARD
    )

    if last_end <= first_end:
        return []

    last_event_idx = None

    for end_idx in range(
        first_end,
        last_end + 1,
    ):
        event = build_event(
            df,
            end_idx,
        )

        if event is None:
            continue

        if (
            last_event_idx is not None
            and (
                end_idx
                - last_event_idx
                < EVENT_SPACING
            )
        ):
            continue

        events.append(event)
        last_event_idx = end_idx

    return events


# ============================================================
# PATTERN CLUSTER
# ============================================================

def cluster_events(events):
    if not events:
        return []

    clusters = []

    for event in events:
        best = None
        best_similarity = -1.0

        for cluster in clusters:
            if (
                event["pattern"]
                != cluster["pattern"]
            ):
                continue

            similarity = cosine_similarity(
                event["signature"],
                cluster["centroid"],
            )

            if (
                similarity >= CLUSTER_SIM
                and similarity > best_similarity
            ):
                best = cluster
                best_similarity = similarity

        if best is None:
            clusters.append(
                {
                    "pattern":
                        event["pattern"],
                    "events":
                        [event],
                    "centroid":
                        event["signature"].copy(),
                }
            )

        else:
            best["events"].append(
                event
            )

            signatures = [
                x["signature"]
                for x in best["events"]
            ]

            centroid = np.mean(
                signatures,
                axis=0,
            )

            norm = np.linalg.norm(
                centroid
            )

            if norm > 1e-12:
                centroid = (
                    centroid / norm
                )

            best["centroid"] = centroid

    return clusters


# ============================================================
# CYCLE ENGINE v6.4
# ============================================================

def adjacent_gaps(events):
    if len(events) < 2:
        return []

    ordered = sorted(
        events,
        key=lambda x:
        pd.Timestamp(
            x["date"]
        ),
    )

    result = []

    for previous, current in zip(
        ordered[:-1],
        ordered[1:],
    ):
        gap = (
            pd.Timestamp(
                current["date"]
            )
            - pd.Timestamp(
                previous["date"]
            )
        ).days

        if gap >= CYCLE_MIN_DAYS:
            result.append(
                {
                    "gap": float(gap),
                    "from_date":
                        previous["date"],
                    "to_date":
                        current["date"],
                }
            )

    return result


def pairwise_gaps(events):
    """
    보조 검증용.

    중요:
    여기서는 cycle base를 생성하지 않는다.
    """

    if len(events) < 2:
        return []

    ordered = sorted(
        events,
        key=lambda x:
        pd.Timestamp(
            x["date"]
        ),
    )

    result = []

    for i in range(len(ordered)):
        for j in range(
            i + 1,
            len(ordered),
        ):
            gap = (
                pd.Timestamp(
                    ordered[j]["date"]
                )
                - pd.Timestamp(
                    ordered[i]["date"]
                )
            ).days

            if gap >= CYCLE_MIN_DAYS:
                result.append(
                    float(gap)
                )

    return result


def nearest_harmonic(
    gap,
    base,
):
    if (
        not np.isfinite(gap)
        or not np.isfinite(base)
        or base <= 0
    ):
        return None, np.inf

    candidates = []

    for multiple in range(
        1,
        MAX_CYCLE_MULTIPLE + 1,
    ):
        expected = (
            base * multiple
        )

        error = abs(
            gap - expected
        ) / max(
            expected,
            1.0,
        )

        candidates.append(
            (
                error,
                multiple,
            )
        )

    error, multiple = min(
        candidates,
        key=lambda x: x[0],
    )

    return (
        multiple,
        float(error),
    )


def fits_harmonic(
    gap,
    base,
):
    multiple, error = nearest_harmonic(
        gap,
        base,
    )

    return (
        multiple is not None
        and error <= HARMONIC_TOLERANCE
    )


def candidate_cycle_bases(
    adjacent,
):
    """
    adjacent gap만 사용한다.

    예:
      390
      780
      1170

    ->

      390 / 1
      780 / 2
      1170 / 3

    모두 같은 cycle base로 합쳐질 수 있다.
    """

    candidates = []

    for item in adjacent:
        gap = safe_float(
            item["gap"]
        )

        if (
            not np.isfinite(gap)
            or gap < CYCLE_MIN_DAYS
        ):
            continue

        for multiple in range(
            1,
            MAX_CYCLE_MULTIPLE + 1,
        ):
            base = (
                gap / multiple
            )

            if base >= CYCLE_MIN_DAYS:
                candidates.append(
                    float(base)
                )

    return candidates


def score_cycle_base(
    base,
    adjacent,
    pairwise,
):
    matched = []
    normalized = []
    multiples = []

    for item in adjacent:
        gap = item["gap"]

        multiple, error = nearest_harmonic(
            gap,
            base,
        )

        if (
            multiple is not None
            and error <= HARMONIC_TOLERANCE
        ):
            matched.append(
                item
            )

            normalized.append(
                gap / multiple
            )

            multiples.append(
                multiple
            )

    pairwise_support = 0

    for gap in pairwise:
        if fits_harmonic(
            gap,
            base,
        ):
            pairwise_support += 1

    if normalized:
        center = float(
            np.median(
                normalized
            )
        )

        errors = [
            abs(
                value - center
            )
            / max(
                center,
                1.0,
            )
            for value
            in normalized
        ]

        median_error = float(
            np.median(errors)
        )

        mean_error = float(
            np.mean(errors)
        )

        stability = clamp01(
            1.0 - median_error
        )

    else:
        center = np.nan
        median_error = np.nan
        mean_error = np.nan
        stability = 0.0

    score = (
        len(matched) * 100.0
        + min(
            pairwise_support,
            4,
        ) * 0.50
        - (
            0.0
            if not np.isfinite(
                mean_error
            )
            else mean_error * 20.0
        )
    )

    return {
        "base": float(base),
        "score": float(score),
        "matched": matched,
        "normalized": normalized,
        "multiples": multiples,
        "pairwise_support":
            pairwise_support,
        "center": center,
        "median_error":
            median_error,
        "mean_error":
            mean_error,
        "stability":
            stability,
    }


def select_cycle_base(events):
    """
    cycle 검증의 핵심.

    절대 하지 않는 것:
        pairwise gap을 이용해서 새로운 cycle을 만들어내기.

    반드시 하는 것:
        adjacent recurrence를 기준으로
        1x/2x/3x harmonic consensus를 찾는다.
    """

    adjacent = adjacent_gaps(
        events
    )

    pairwise = pairwise_gaps(
        events
    )

    if (
        len(adjacent)
        < MIN_CYCLE_OBSERVATIONS
    ):
        return {
            "valid": False,
            "reason":
                "INSUFFICIENT_ADJACENT_GAPS",
            "base": np.nan,
            "observations":
                len(adjacent),
            "stability": 0.0,
            "confidence": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "adjacent": adjacent,
            "pairwise": pairwise,
            "matched": [],
            "multiples": [],
        }

    candidates = candidate_cycle_bases(
        adjacent
    )

    if not candidates:
        return {
            "valid": False,
            "reason":
                "NO_CYCLE_BASE",
            "base": np.nan,
            "observations": 0,
            "stability": 0.0,
            "confidence": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "adjacent": adjacent,
            "pairwise": pairwise,
            "matched": [],
            "multiples": [],
        }

    scored = []

    for base in candidates:
        scored.append(
            score_cycle_base(
                base,
                adjacent,
                pairwise,
            )
        )

    # 같은 base 주변 후보는 하나로 통합.
    scored.sort(
        key=lambda x:
        x["base"]
    )

    merged = []

    for item in scored:
        placed = False

        for group in merged:
            center = group["center"]

            error = abs(
                item["base"] - center
            ) / max(
                center,
                1.0,
            )

            if (
                error
                <= HARMONIC_TOLERANCE
            ):
                group["items"].append(
                    item
                )

                group["center"] = float(
                    np.median(
                        [
                            x["base"]
                            for x
                            in group["items"]
                        ]
                    )
                )

                placed = True
                break

        if not placed:
            merged.append(
                {
                    "center":
                        item["base"],
                    "items":
                        [item],
                }
            )

    representatives = []

    for group in merged:
        representative = max(
            group["items"],
            key=lambda x: (
                x["score"],
                x["stability"],
                x["matched"]
                if isinstance(
                    x["matched"],
                    int,
                )
                else len(
                    x["matched"]
                ),
            )
        )

        representatives.append(
            representative
        )

    representatives.sort(
        key=lambda x: (
            -len(
                x["matched"]
            ),
            -x["stability"],
            x["mean_error"]
            if np.isfinite(
                x["mean_error"]
            )
            else 999,
            x["base"],
        )
    )

    best = representatives[0]

    normalized = best[
        "normalized"
    ]

    observations = len(
        normalized
    )

    if observations >= 2:
        cycle_mode = float(
            np.median(
                normalized
            )
        )

        deviations = [
            abs(
                x - cycle_mode
            )
            / max(
                cycle_mode,
                1.0,
            )
            for x in normalized
        ]

        stability = clamp01(
            1.0
            - float(
                np.median(
                    deviations
                )
            )
        )

    else:
        cycle_mode = np.nan
        stability = 0.0

    # --------------------------------------------------------
    # cycle confidence
    #
    # 2개 recurrence link만 맞아도 stability가 99%가
    # 되는 문제를 방지한다.
    # --------------------------------------------------------

    link_confidence = (
        observations
        / (
            observations
            + 2.5
        )
    )

    event_confidence = (
        len(events)
        / (
            len(events)
            + 3.0
        )
    )

    confidence = (
        link_confidence * 0.70
        + event_confidence * 0.30
    )

    # 2개 link는 절대로 강한 cycle로 취급하지 않는다.
    if observations <= 2:
        confidence = min(
            confidence,
            0.55,
        )

    valid = (
        observations
        >= MIN_CYCLE_OBSERVATIONS
        and np.isfinite(
            cycle_mode
        )
        and stability
        >= MIN_CYCLE_STABILITY
    )

    return {
        "valid":
            bool(valid),

        "reason":
            "VALID"
            if valid
            else
            "LOW_CYCLE_STABILITY",

        "base":
            cycle_mode,

        "observations":
            observations,

        "stability":
            float(stability),

        "confidence":
            float(
                clamp01(
                    confidence
                )
            ),

        "mean":
            safe_mean(
                normalized
            ),

        "std":
            safe_std(
                normalized
            ),

        "adjacent":
            adjacent,

        "pairwise":
            pairwise,

        "matched":
            normalized,

        "matched_adjacent":
            best["matched"],

        "multiples":
            best["multiples"],

        "pairwise_support":
            best["pairwise_support"],

        "mean_error":
            best["mean_error"],
    }


def classify_cycle(
    cycle_result
):
    base = safe_float(
        cycle_result.get(
            "base"
        )
    )

    if (
        not np.isfinite(base)
        or base <= 0
    ):
        return "미검증"

    if base < 180:
        return "단기"

    return "장기"


# ============================================================
# CYCLE GROUP
# ============================================================

def cycle_group_events(
    events,
    cycle_result,
):
    if not events:
        return []

    base = safe_float(
        cycle_result.get(
            "base"
        )
    )

    stability = safe_float(
        cycle_result.get(
            "stability"
        )
    )

    observations = int(
        cycle_result.get(
            "observations",
            0,
        )
    )

    if (
        not np.isfinite(base)
        or base <= 0
        or observations
        < MIN_CYCLE_OBSERVATIONS
        or not np.isfinite(stability)
        or stability
        < MIN_CYCLE_STABILITY
    ):
        return []

    ordered = sorted(
        events,
        key=lambda x:
        pd.Timestamp(
            x["date"]
        ),
    )

    matched_edges = []

    for previous, current in zip(
        ordered[:-1],
        ordered[1:],
    ):
        gap = (
            pd.Timestamp(
                current["date"]
            )
            - pd.Timestamp(
                previous["date"]
            )
        ).days

        if gap < CYCLE_MIN_DAYS:
            continue

        multiple, error = nearest_harmonic(
            gap,
            base,
        )

        if (
            multiple is not None
            and error <= HARMONIC_TOLERANCE
        ):
            matched_edges.append(
                {
                    "previous":
                        previous,
                    "current":
                        current,
                    "gap":
                        gap,
                    "multiple":
                        multiple,
                    "error":
                        error,
                }
            )

    if (
        len(matched_edges)
        < MIN_CYCLE_OBSERVATIONS
    ):
        return []

    # --------------------------------------------------------
    # 연속 chain만 인정
    # --------------------------------------------------------

    chains = []

    current_chain = [
        matched_edges[0]["previous"],
        matched_edges[0]["current"],
    ]

    for edge in matched_edges[1:]:
        previous_date = pd.Timestamp(
            edge["previous"]["date"]
        )

        last_date = pd.Timestamp(
            current_chain[-1]["date"]
        )

        if previous_date == last_date:
            current_chain.append(
                edge["current"]
            )

        else:
            chains.append(
                current_chain
            )

            current_chain = [
                edge["previous"],
                edge["current"],
            ]

    chains.append(
        current_chain
    )

    chains = [
        chain
        for chain in chains
        if len(chain)
        >= MIN_DRIVER_OCCURRENCES
    ]

    if not chains:
        return []

    # 가장 많은 실제 recurrence를 가진 chain.
    chains.sort(
        key=lambda chain: (
            -len(chain),
            -pd.Timestamp(
                chain[-1]["date"]
            ).value,
        )
    )

    return [
        chains[0]
    ]


# ============================================================
# OUTCOME STATS
# ============================================================

def calculate_ordered_stats(
    events
):
    plus5 = []
    plus10 = []
    plus20 = []
    minus5 = []

    ambiguous = 0
    no_hit = 0
    resolved = 0

    for event in events:
        ordered = event[
            "ordered"
        ]

        status = ordered[
            "outcome_status"
        ]

        if status == "AMBIGUOUS":
            ambiguous += 1

        elif status == "NO_HIT":
            no_hit += 1

        elif status == "RESOLVED":
            resolved += 1

        for key, target in [
            (
                "plus5_first",
                plus5,
            ),
            (
                "plus10_first",
                plus10,
            ),
            (
                "plus20_first",
                plus20,
            ),
            (
                "minus5_first",
                minus5,
            ),
        ]:
            value = safe_float(
                ordered[key]
            )

            if np.isfinite(value):
                target.append(
                    value
                )

    success_count = 0

    for event in events:
        if (
            event["ordered"][
                "first_event"
            ]
            == "PLUS_5"
        ):
            success_count += 1

    return {
        "plus5_valid_n":
            len(plus5),

        "plus10_valid_n":
            len(plus10),

        "plus20_valid_n":
            len(plus20),

        "minus5_valid_n":
            len(minus5),

        "plus5_first_count":
            sum(
                x == 1.0
                for x in plus5
            ),

        "plus10_first_count":
            sum(
                x == 1.0
                for x in plus10
            ),

        "plus20_first_count":
            sum(
                x == 1.0
                for x in plus20
            ),

        "minus5_first_count":
            sum(
                x == 1.0
                for x in minus5
            ),

        "plus5_first_rate":
            safe_mean(plus5),

        "plus10_first_rate":
            safe_mean(plus10),

        "plus20_first_rate":
            safe_mean(plus20),

        "minus5_first_rate":
            safe_mean(minus5),

        "ambiguous_count":
            ambiguous,

        "no_hit_count":
            no_hit,

        "resolved_count":
            resolved,

        "success_count":
            success_count,

        "success_rate":
            (
                success_count
                / resolved
                if resolved > 0
                else np.nan
            ),
    }


# ============================================================
# RESULT REPRODUCIBILITY
# ============================================================

def result_reproducibility(
    events
):
    if not events:
        return {
            "result_reproducibility":
                np.nan,
            "raw_result_reproducibility":
                np.nan,
            "resolved_rate":
                np.nan,
            "ambiguous_rate":
                np.nan,
            "no_hit_rate":
                np.nan,
            "resolved_n":
                0,
            "positive_n":
                0,
            "negative_n":
                0,
        }

    resolved = []
    ambiguous = 0
    no_hit = 0

    for event in events:
        first_event = event[
            "ordered"
        ][
            "first_event"
        ]

        if first_event == "PLUS_5":
            resolved.append(
                1.0
            )

        elif first_event == "MINUS_5":
            resolved.append(
                0.0
            )

        elif first_event == "AMBIGUOUS":
            ambiguous += 1

        else:
            no_hit += 1

    total = len(events)
    resolved_n = len(resolved)

    positive_n = sum(
        x == 1.0
        for x in resolved
    )

    negative_n = (
        resolved_n
        - positive_n
    )

    if resolved_n:
        raw = (
            positive_n
            / resolved_n
        )

        resolved_rate = (
            resolved_n
            / total
        )

        # Beta-like 소표본 shrinkage.
        confidence = (
            resolved_n
            / (
                resolved_n
                + 2.0
            )
        )

        adjusted = (
            raw * confidence
            + 0.50
            * (
                1.0
                - confidence
            )
        )

    else:
        raw = np.nan
        resolved_rate = 0.0
        adjusted = np.nan

    return {
        "result_reproducibility":
            adjusted,

        "raw_result_reproducibility":
            raw,

        "resolved_rate":
            resolved_rate,

        "ambiguous_rate":
            ambiguous / total,

        "no_hit_rate":
            no_hit / total,

        "resolved_n":
            resolved_n,

        "positive_n":
            positive_n,

        "negative_n":
            negative_n,
    }


# ============================================================
# RECENT
# ============================================================

def recent_statistics(
    events
):
    if not events:
        return {
            "recent_occurrences": 0,
            "recent_plus5_first_pct":
                np.nan,
            "recent_plus10_first_pct":
                np.nan,
            "recent_plus20_first_pct":
                np.nan,
            "recent_minus5_first_pct":
                np.nan,
            "recent_result_reproducibility_pct":
                np.nan,
            "recent_resolved_rate_pct":
                np.nan,
            "recent_ambiguous_rate_pct":
                np.nan,
            "recent_no_hit_rate_pct":
                np.nan,
        }

    ordered = sorted(
        events,
        key=lambda x:
        pd.Timestamp(
            x["date"]
        ),
    )

    recent = ordered[
        -RECENT_OCCURRENCES:
    ]

    stats = calculate_ordered_stats(
        recent
    )

    repro = result_reproducibility(
        recent
    )

    return {
        "recent_occurrences":
            len(recent),

        "recent_plus5_first_pct":
            pct(
                stats[
                    "plus5_first_rate"
                ]
            ),

        "recent_plus10_first_pct":
            pct(
                stats[
                    "plus10_first_rate"
                ]
            ),

        "recent_plus20_first_pct":
            pct(
                stats[
                    "plus20_first_rate"
                ]
            ),

        "recent_minus5_first_pct":
            pct(
                stats[
                    "minus5_first_rate"
                ]
            ),

        "recent_result_reproducibility_pct":
            pct(
                repro[
                    "result_reproducibility"
                ]
            ),

        "recent_resolved_rate_pct":
            pct(
                repro[
                    "resolved_rate"
                ]
            ),

        "recent_ambiguous_rate_pct":
            pct(
                repro[
                    "ambiguous_rate"
                ]
            ),

        "recent_no_hit_rate_pct":
            pct(
                repro[
                    "no_hit_rate"
                ]
            ),
    }


# ============================================================
# TIMING
# ============================================================

def calculate_cycle_timing(
    events,
    cycle_mode,
    as_of,
):
    if not events:
        return {
            "days_since_last_occurrence":
                np.nan,
            "expected_next_cycle":
                np.nan,
            "cycle_phase_pct":
                np.nan,
        }

    ordered = sorted(
        events,
        key=lambda x:
        pd.Timestamp(
            x["date"]
        ),
    )

    last_date = pd.Timestamp(
        ordered[-1]["date"]
    )

    current_date = pd.Timestamp(
        as_of
    )

    days_since = (
        current_date
        - last_date
    ).days

    cycle = safe_float(
        cycle_mode
    )

    if (
        not np.isfinite(cycle)
        or cycle <= 0
    ):
        return {
            "days_since_last_occurrence":
                days_since,
            "expected_next_cycle":
                np.nan,
            "cycle_phase_pct":
                np.nan,
        }

    expected = (
        last_date
        + pd.Timedelta(
            days=float(cycle)
        )
    )

    return {
        "days_since_last_occurrence":
            days_since,

        "expected_next_cycle":
            date_string(
                expected
            ),

        "cycle_phase_pct":
            (
                days_since
                / cycle
                * 100.0
            ),
    }


# ============================================================
# TARGET
# ============================================================

def calculate_targets(
    current_price,
    events,
):
    if (
        not np.isfinite(
            current_price
        )
        or current_price <= 0
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    ups = finite_values(
        [
            x["max_up"]
            for x in events
        ]
    )

    downs = finite_values(
        [
            x["max_down"]
            for x in events
        ]
    )

    if ups.empty:
        target1 = (
            current_price * 1.05
        )

        target2 = (
            current_price * 1.10
        )

    else:
        median_up = float(
            ups.median()
        )

        upper_up = float(
            ups.quantile(0.75)
        )

        target1 = (
            current_price
            * (
                1
                + max(
                    0.05,
                    min(
                        median_up,
                        0.20,
                    ),
                )
            )
        )

        target2 = (
            current_price
            * (
                1
                + max(
                    0.10,
                    min(
                        upper_up,
                        0.40,
                    ),
                )
            )
        )

    if downs.empty:
        invalidation = (
            current_price * 0.95
        )

    else:
        median_down = float(
            downs.median()
        )

        invalidation = (
            current_price
            * (
                1
                + min(
                    -0.05,
                    max(
                        median_down,
                        -0.30,
                    ),
                )
            )
        )

    return (
        target1,
        target2,
        invalidation,
    )


# ============================================================
# VALIDATION
# ============================================================

def validation_status(
    occurrences,
    cycle_observations,
    cycle_stability,
    cycle_confidence,
    result_repro,
    resolved_n,
    recent_repro,
):
    if (
        occurrences
        < MIN_DRIVER_OCCURRENCES
    ):
        return (
            "INSUFFICIENT_OCCURRENCES"
        )

    if (
        cycle_observations
        < MIN_CYCLE_OBSERVATIONS
    ):
        return (
            "LOW_CYCLE_REPEAT"
        )

    if (
        not np.isfinite(
            safe_float(
                cycle_stability
            )
        )
        or cycle_stability
        < MIN_CYCLE_STABILITY
    ):
        return (
            "LOW_CYCLE_STABILITY"
        )

    if (
        not np.isfinite(
            safe_float(
                cycle_confidence
            )
        )
        or cycle_confidence
        < CYCLE_CONFIDENCE_FLOOR
    ):
        return (
            "LOW_CYCLE_CONFIDENCE"
        )

    if (
        resolved_n
        < MIN_RESOLVED_RESULTS
    ):
        return (
            "LOW_RESULT_SAMPLE"
        )

    if (
        np.isfinite(
            safe_float(
                result_repro
            )
        )
        and result_repro
        < MIN_RESULT_REPRODUCIBILITY
    ):
        return (
            "LOW_RESULT_REPRO"
        )

    if (
        occurrences
        >= VALIDATED_OCCURRENCES
        and np.isfinite(
            safe_float(
                recent_repro
            )
        )
        and recent_repro
        < MIN_RECENT_REPRODUCIBILITY
    ):
        return (
            "RECENT_RESULT_WEAK"
        )

    if (
        occurrences
        < VALIDATED_OCCURRENCES
    ):
        return "REPEAT_DETECTED"

    if (
        occurrences
        < STABLE_OCCURRENCES
    ):
        return "VALIDATED"

    return "STABLE"


def driver_state(
    current_similarity,
    validation,
):
    similarity = safe_float(
        current_similarity
    )

    weak = {
        "INSUFFICIENT_OCCURRENCES",
        "LOW_CYCLE_REPEAT",
        "LOW_CYCLE_STABILITY",
        "LOW_CYCLE_CONFIDENCE",
        "LOW_RESULT_SAMPLE",
        "LOW_RESULT_REPRO",
        "RECENT_RESULT_WEAK",
    }

    if validation in weak:
        return "PROVISIONAL"

    if (
        np.isfinite(similarity)
        and similarity
        >= CURRENT_ACTIVE_SIM
    ):
        return "ACTIVE"

    if (
        np.isfinite(similarity)
        and similarity
        >= CURRENT_WATCH_SIM
    ):
        return "WATCH"

    return "INACTIVE"


# ============================================================
# QUALITY
# ============================================================

def driver_quality(
    occurrences,
    cycle_stability,
    cycle_confidence,
    result_repro,
    recent_repro,
    current_similarity,
):
    sim = clamp01(
        current_similarity
    )

    cycle = clamp01(
        cycle_stability
    )

    confidence = clamp01(
        cycle_confidence
    )

    repro = clamp01(
        result_repro
    )

    recent = clamp01(
        recent_repro
    )

    occurrence = clamp01(
        occurrences / 8.0
    )

    return float(
        sim * 0.20
        + cycle * 0.15
        + confidence * 0.10
        + repro * 0.30
        + recent * 0.15
        + occurrence * 0.10
    )


# ============================================================
# SCORE
# ============================================================

def driver_score(
    current_similarity,
    occurrences,
    plus5_first_rate,
    cycle_stability,
    cycle_confidence,
    recent_plus5,
    result_repro,
    recent_repro,
):
    sim = clamp01(
        current_similarity
    )

    occurrence = clamp01(
        occurrences / 8.0
    )

    plus5 = clamp01(
        plus5_first_rate
    )

    cycle = clamp01(
        cycle_stability
    )

    confidence = clamp01(
        cycle_confidence
    )

    recent = clamp01(
        recent_plus5
    )

    repro = clamp01(
        result_repro
    )

    recent_repro_value = clamp01(
        recent_repro
    )

    return float(
        sim * 0.18
        + occurrence * 0.08
        + plus5 * 0.12
        + cycle * 0.12
        + confidence * 0.10
        + recent * 0.08
        + repro * 0.22
        + recent_repro_value * 0.10
    )


# ============================================================
# DUPLICATE
# ============================================================

def cycle_ratio_is_harmonic(
    c1,
    c2,
):
    if (
        not np.isfinite(c1)
        or not np.isfinite(c2)
        or c1 <= 0
        or c2 <= 0
    ):
        return False

    ratio = (
        max(c1, c2)
        / min(c1, c2)
    )

    nearest = round(
        ratio
    )

    if (
        nearest < 1
        or nearest
        > MAX_CYCLE_MULTIPLE
    ):
        return False

    error = (
        abs(
            ratio - nearest
        )
        / max(
            nearest,
            1,
        )
    )

    return (
        error
        <= HARMONIC_TOLERANCE
    )


def is_duplicate_driver(
    candidate,
    selected,
):
    for existing in selected:
        similarity = cosine_similarity(
            candidate["centroid"],
            existing["centroid"],
        )

        c1 = safe_float(
            candidate["cycle_mode"]
        )

        c2 = safe_float(
            existing["cycle_mode"]
        )

        # 같은 구조 + 같은/harmonic cycle
        # -> 하나의 Driver.
        if (
            similarity
            >= DUPLICATE_SIM
        ):
            if (
                np.isfinite(c1)
                and np.isfinite(c2)
            ):
                ratio = (
                    max(c1, c2)
                    / max(
                        min(c1, c2),
                        1.0,
                    )
                )

                if (
                    ratio
                    <= DUPLICATE_CYCLE_RATIO
                    or cycle_ratio_is_harmonic(
                        c1,
                        c2,
                    )
                ):
                    return True

            else:
                return True

    return False


# ============================================================
# ID
# ============================================================

def make_base_cluster_id(
    symbol,
    pattern,
    centroid,
):
    vector = np.asarray(
        centroid,
        dtype=float,
    )

    fingerprint = np.round(
        vector,
        3,
    ).tobytes()

    raw = (
        f"{symbol}|"
        f"{pattern}|"
        f"{hashlib.sha1(fingerprint).hexdigest()}"
    )

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:10]

    return (
        f"{symbol}-BASE-{digest}"
    )


def make_cluster_id(
    symbol,
    cycle_type,
    base_cluster_id,
):
    raw = (
        f"{symbol}|"
        f"{base_cluster_id}|"
        f"{cycle_type}"
    )

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:10]

    return (
        f"{symbol}-"
        f"{cycle_type}-"
        f"{digest}"
    )


# ============================================================
# HISTORY ROW
# ============================================================

def make_history_row(
    symbol,
    rank,
    driver_name,
    cluster_id,
    pattern,
    cycle_type,
    cycle_mode,
    event,
    base_cluster_id,
):
    ordered = event[
        "ordered"
    ]

    return {
        "symbol":
            symbol,

        "driver_rank":
            rank,

        "driver_name":
            driver_name,

        "cluster_id":
            cluster_id,

        "base_cluster_id":
            base_cluster_id,

        "date":
            date_string(
                event["date"]
            ),

        "price":
            rnd(
                event["price"],
                4,
            ),

        "pattern":
            pattern,

        "cycle_type":
            cycle_type,

        "cycle_mode_days":
            rnd(
                cycle_mode,
                1,
            ),

        "forward_return_pct":
            rnd(
                event[
                    "forward_return"
                ] * 100,
                2,
            ),

        "max_up_pct":
            rnd(
                event[
                    "max_up"
                ] * 100,
                2,
            ),

        "max_down_pct":
            rnd(
                event[
                    "max_down"
                ] * 100,
                2,
            ),

        "first_event":
            ordered[
                "first_event"
            ],

        "first_day":
            ordered[
                "first_day"
            ],

        "plus5_first":
            ordered[
                "plus5_first"
            ],

        "plus10_first":
            ordered[
                "plus10_first"
            ],

        "plus20_first":
            ordered[
                "plus20_first"
            ],

        "minus5_first":
            ordered[
                "minus5_first"
            ],

        "days_to_plus5":
            ordered[
                "days_to_plus5"
            ],

        "days_to_plus10":
            ordered[
                "days_to_plus10"
            ],

        "days_to_plus20":
            ordered[
                "days_to_plus20"
            ],

        "days_to_minus5":
            ordered[
                "days_to_minus5"
            ],

        "outcome_status":
            ordered[
                "outcome_status"
            ],
    }


# ============================================================
# ANALYZE SYMBOL
# ============================================================

def analyze_symbol(symbol):
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

    if len(raw) < MIN_HISTORY:
        print(
            f"[SKIP] {symbol}: "
            f"history={len(raw)} "
            f"< {MIN_HISTORY}"
        )

        return [], [], []

    df = add_features(
        raw
    )

    df = df.dropna(
        subset=[
            "Close",
            "ma20",
            "ma50",
            "ma100",
            "ma200",
        ]
    ).copy()

    if len(df) < (
        MIN_HISTORY
        + FORWARD
    ):
        print(
            f"[SKIP] {symbol}: "
            "not enough usable history"
        )

        return [], [], []

    events = extract_events(
        df
    )

    print(
        f"[INFO] {symbol}: "
        f"raw_events={len(events)}"
    )

    if len(events) < (
        MIN_DRIVER_OCCURRENCES
    ):
        print(
            f"[SKIP] {symbol}: "
            "not enough recurring events"
        )

        return [], [], []

    pattern_clusters = cluster_events(
        events
    )

    current_price = safe_float(
        df["Close"].iloc[-1]
    )

    as_of = df.index[-1]

    current_signature = make_signature(
        df.iloc[-WINDOW:]
    )

    candidates = []

    # ========================================================
    # CANDIDATE BUILD
    # ========================================================

    for cluster in pattern_clusters:
        cluster_events_list = (
            cluster["events"]
        )

        if len(
            cluster_events_list
        ) < MIN_DRIVER_OCCURRENCES:
            continue

        cycle_result = select_cycle_base(
            cluster_events_list
        )

        if not cycle_result.get(
            "valid",
            False,
        ):
            continue

        cycle_mode = safe_float(
            cycle_result.get(
                "base"
            )
        )

        cycle_observations = int(
            cycle_result.get(
                "observations",
                0,
            )
        )

        cycle_stability = safe_float(
            cycle_result.get(
                "stability"
            )
        )

        cycle_confidence = safe_float(
            cycle_result.get(
                "confidence",
                0.0,
            )
        )

        if (
            cycle_observations
            < MIN_CYCLE_OBSERVATIONS
        ):
            continue

        if (
            not np.isfinite(
                cycle_stability
            )
            or cycle_stability
            < MIN_CYCLE_STABILITY
        ):
            continue

        if (
            not np.isfinite(
                cycle_confidence
            )
            or cycle_confidence
            < CYCLE_CONFIDENCE_FLOOR
        ):
            continue

        grouped = cycle_group_events(
            cluster_events_list,
            cycle_result,
        )

        if not grouped:
            continue

        event_group = grouped[0]

        if len(event_group) < (
            MIN_DRIVER_OCCURRENCES
        ):
            continue

        signatures = [
            event["signature"]
            for event in event_group
        ]

        centroid = np.mean(
            signatures,
            axis=0,
        )

        norm = np.linalg.norm(
            centroid
        )

        if norm > 1e-12:
            centroid = (
                centroid / norm
            )

        current_similarity = cosine_similarity(
            current_signature,
            centroid,
        )

        if (
            not np.isfinite(
                current_similarity
            )
            or current_similarity
            < CURRENT_WATCH_SIM
        ):
            continue

        stats = calculate_ordered_stats(
            event_group
        )

        recent = recent_statistics(
            event_group
        )

        repro = result_reproducibility(
            event_group
        )

        result_repro = safe_float(
            repro.get(
                "result_reproducibility"
            )
        )

        recent_repro = safe_float(
            recent.get(
                "recent_result_reproducibility_pct"
            )
        )

        recent_repro_rate = (
            recent_repro / 100.0
            if np.isfinite(
                recent_repro
            )
            else np.nan
        )

        recent_plus5 = safe_float(
            recent.get(
                "recent_plus5_first_pct",
                np.nan,
            )
        )

        recent_plus5_rate = (
            recent_plus5 / 100.0
            if np.isfinite(
                recent_plus5
            )
            else np.nan
        )

        validation = validation_status(
            len(event_group),
            cycle_observations,
            cycle_stability,
            cycle_confidence,
            result_repro,
            int(
                repro.get(
                    "resolved_n",
                    0,
                )
            ),
            recent_repro_rate,
        )

        score = driver_score(
            current_similarity,
            len(event_group),
            stats[
                "plus5_first_rate"
            ],
            cycle_stability,
            cycle_confidence,
            recent_plus5_rate,
            result_repro,
            recent_repro_rate,
        )

        quality = driver_quality(
            len(event_group),
            cycle_stability,
            cycle_confidence,
            result_repro,
            recent_repro_rate,
            current_similarity,
        )

        state = driver_state(
            current_similarity,
            validation,
        )

        target1, target2, invalidation = (
            calculate_targets(
                current_price,
                event_group,
            )
        )

        cycle_type = classify_cycle(
            cycle_result
        )

        timing = calculate_cycle_timing(
            event_group,
            cycle_mode,
            as_of,
        )

        base_cluster_id = make_base_cluster_id(
            symbol,
            cluster["pattern"],
            centroid,
        )

        cluster_id = make_cluster_id(
            symbol,
            cycle_type,
            base_cluster_id,
        )

        candidates.append(
            {
                "events":
                    event_group,

                "pattern":
                    cluster["pattern"],

                "centroid":
                    centroid,

                "current_similarity":
                    current_similarity,

                "occurrences":
                    len(event_group),

                "cycle_type":
                    cycle_type,

                "cycle_mode":
                    cycle_mode,

                "cycle_observations":
                    cycle_observations,

                "cycle_stability":
                    cycle_stability,

                "cycle_confidence":
                    cycle_confidence,

                "cycle_mean":
                    cycle_result.get(
                        "mean",
                        np.nan,
                    ),

                "cycle_std":
                    cycle_result.get(
                        "std",
                        np.nan,
                    ),

                "ordered":
                    stats,

                "recent":
                    recent,

                "repro":
                    repro,

                "result_repro":
                    result_repro,

                "recent_repro":
                    recent_repro_rate,

                "timing":
                    timing,

                "validation":
                    validation,

                "state":
                    state,

                "quality":
                    quality,

                "score":
                    score,

                "target1":
                    target1,

                "target2":
                    target2,

                "invalidation":
                    invalidation,

                "base_cluster_id":
                    base_cluster_id,

                "cluster_id":
                    cluster_id,
            }
        )

    # ========================================================
    # SORT
    # ========================================================

    candidates.sort(
        key=lambda item: (
            -safe_float(
                item["score"],
                0,
            ),
            -safe_float(
                item["quality"],
                0,
            ),
            -safe_float(
                item[
                    "current_similarity"
                ],
                0,
            ),
            -safe_float(
                item[
                    "result_repro"
                ],
                0,
            ),
            -safe_float(
                item[
                    "cycle_confidence"
                ],
                0,
            ),
            -int(
                item[
                    "occurrences"
                ]
            ),
        )
    )

    # ========================================================
    # SELECT
    # ========================================================

    selected = []

    for candidate in candidates:
        if len(selected) >= MAX_DRIVERS:
            break

        if is_duplicate_driver(
            candidate,
            selected,
        ):
            continue

        selected.append(
            candidate
        )

    profiles = []
    current_rows = []
    history_rows = []

    current_pattern = choose_pattern_label(
        df.iloc[-WINDOW:]
    )

    ma20 = safe_float(
        df["ma20"].iloc[-1]
    )

    ma50 = safe_float(
        df["ma50"].iloc[-1]
    )

    ma100 = safe_float(
        df["ma100"].iloc[-1]
    )

    if (
        ma20 > ma50
        and ma50 > ma100
    ):
        phase = "UPTREND"

    elif (
        ma20 < ma50
        and ma50 < ma100
    ):
        phase = "DOWNTREND"

    else:
        phase = "TRANSITION"

    # ========================================================
    # PROFILE
    # ========================================================

    for rank, item in enumerate(
        selected,
        start=1,
    ):
        driver_name = (
            f"DRIVER_{rank}"
        )

        ordered = item[
            "ordered"
        ]

        recent = item[
            "recent"
        ]

        repro = item[
            "repro"
        ]

        timing = item[
            "timing"
        ]

        pattern_match = cosine_similarity(
            current_signature,
            item["centroid"],
        )

        profile = {
            "symbol":
                symbol,

            "driver_rank":
                rank,

            "driver_name":
                driver_name,

            "cluster_id":
                item["cluster_id"],

            "base_cluster_id":
                item[
                    "base_cluster_id"
                ],

            "pattern":
                item["pattern"],

            "historical_pattern":
                item["pattern"],

            "current_pattern":
                current_pattern,

            "pattern_match_pct":
                pct(
                    pattern_match
                ),

            "cycle_type":
                item["cycle_type"],

            "cycle_mode_days":
                rnd(
                    item["cycle_mode"],
                    1,
                ),

            "cycle_stability_pct":
                pct(
                    item[
                        "cycle_stability"
                    ]
                ),

            "cycle_confidence_pct":
                pct(
                    item[
                        "cycle_confidence"
                    ]
                ),

            "cycle_observations":
                item[
                    "cycle_observations"
                ],

            "state":
                item["state"],

            "validation_status":
                item[
                    "validation"
                ],

            "phase":
                phase,

            "current_price":
                rnd(
                    current_price,
                    4,
                ),

            "current_similarity_pct":
                pct(
                    item[
                        "current_similarity"
                    ]
                ),

            "driver_score_pct":
                pct(
                    item["score"]
                ),

            "driver_quality_pct":
                pct(
                    item["quality"]
                ),

            "success_rate_pct":
                pct(
                    ordered[
                        "success_rate"
                    ]
                ),

            "result_reproducibility_pct":
                pct(
                    item[
                        "result_repro"
                    ]
                ),

            "resolved_rate_pct":
                pct(
                    repro[
                        "resolved_rate"
                    ]
                ),

            "ambiguous_rate_pct":
                pct(
                    repro[
                        "ambiguous_rate"
                    ]
                ),

            "no_hit_rate_pct":
                pct(
                    repro[
                        "no_hit_rate"
                    ]
                ),

            "plus5_first_pct":
                pct(
                    ordered[
                        "plus5_first_rate"
                    ]
                ),

            "plus10_first_pct":
                pct(
                    ordered[
                        "plus10_first_rate"
                    ]
                ),

            "plus20_first_pct":
                pct(
                    ordered[
                        "plus20_first_rate"
                    ]
                ),

            "minus5_first_pct":
                pct(
                    ordered[
                        "minus5_first_rate"
                    ]
                ),

            "recent_plus5_first_pct":
                recent[
                    "recent_plus5_first_pct"
                ],

            "recent_plus10_first_pct":
                recent[
                    "recent_plus10_first_pct"
                ],

            "recent_plus20_first_pct":
                recent[
                    "recent_plus20_first_pct"
                ],

            "recent_minus5_first_pct":
                recent[
                    "recent_minus5_first_pct"
                ],

            "recent_result_reproducibility_pct":
                recent[
                    "recent_result_reproducibility_pct"
                ],

            "days_since_last_occurrence":
                timing[
                    "days_since_last_occurrence"
                ],

            "expected_next_cycle":
                timing[
                    "expected_next_cycle"
                ],

            "cycle_phase_pct":
                rnd(
                    timing[
                        "cycle_phase_pct"
                    ],
                    1,
                ),

            "avg_forward_return_pct":
                rnd(
                    safe_mean(
                        [
                            e[
                                "forward_return"
                            ]
                            for e
                            in item["events"]
                        ]
                    ) * 100,
                    2,
                ),

            "median_forward_return_pct":
                rnd(
                    safe_median(
                        [
                            e[
                                "forward_return"
                            ]
                            for e
                            in item["events"]
                        ]
                    ) * 100,
                    2,
                ),

            "median_max_up_pct":
                rnd(
                    safe_median(
                        [
                            e[
                                "max_up"
                            ]
                            for e
                            in item["events"]
                        ]
                    ) * 100,
                    2,
                ),

            "median_max_down_pct":
                rnd(
                    safe_median(
                        [
                            e[
                                "max_down"
                            ]
                            for e
                            in item["events"]
                        ]
                    ) * 100,
                    2,
                ),

            "target1":
                rnd(
                    item[
                        "target1"
                    ],
                    4,
                ),

            "target2":
                rnd(
                    item[
                        "target2"
                    ],
                    4,
                ),

            "invalidation":
                rnd(
                    item[
                        "invalidation"
                    ],
                    4,
                ),

            "occurrences":
                item["occurrences"],

            "first_seen":
                min(
                    date_string(
                        e["date"]
                    )
                    for e
                    in item["events"]
                ),

            "last_seen":
                max(
                    date_string(
                        e["date"]
                    )
                    for e
                    in item["events"]
                ),

            "cycle_status":
                (
                    "VALIDATED"
                    if (
                        item[
                            "cycle_observations"
                        ]
                        >= MIN_CYCLE_OBSERVATIONS
                        and
                        item[
                            "cycle_confidence"
                        ]
                        >= CYCLE_CONFIDENCE_FLOOR
                    )
                    else
                    "INSUFFICIENT_REPEAT"
                ),
        }

        profiles.append(
            profile
        )

        for event in item[
            "events"
        ]:
            history_rows.append(
                make_history_row(
                    symbol,
                    rank,
                    driver_name,
                    item[
                        "cluster_id"
                    ],
                    item[
                        "pattern"
                    ],
                    item[
                        "cycle_type"
                    ],
                    item[
                        "cycle_mode"
                    ],
                    event,
                    item[
                        "base_cluster_id"
                    ],
                )
            )

    # ========================================================
    # CURRENT
    # ========================================================

    active_profiles = [
        p
        for p in profiles
        if p["state"]
        in {
            "ACTIVE",
            "WATCH",
        }
    ]

    if active_profiles:
        top = active_profiles[0]

        current_rows.append(
            {
                "symbol":
                    symbol,

                "as_of":
                    date_string(
                        df.index[-1]
                    ),

                "current_price":
                    top[
                        "current_price"
                    ],

                "active_driver":
                    top[
                        "driver_name"
                    ],

                "cluster_id":
                    top[
                        "cluster_id"
                    ],

                "base_cluster_id":
                    top[
                        "base_cluster_id"
                    ],

                "pattern":
                    top[
                        "pattern"
                    ],

                "historical_pattern":
                    top[
                        "historical_pattern"
                    ],

                "current_pattern":
                    top[
                        "current_pattern"
                    ],

                "pattern_match_pct":
                    top[
                        "pattern_match_pct"
                    ],

                "cycle_type":
                    top[
                        "cycle_type"
                    ],

                "cycle_mode_days":
                    top[
                        "cycle_mode_days"
                    ],

                "cycle_stability_pct":
                    top[
                        "cycle_stability_pct"
                    ],

                "cycle_confidence_pct":
                    top[
                        "cycle_confidence_pct"
                    ],

                "cycle_observations":
                    top[
                        "cycle_observations"
                    ],

                "state":
                    top["state"],

                "validation_status":
                    top[
                        "validation_status"
                    ],

                "phase":
                    top["phase"],

                "similarity_pct":
                    top[
                        "current_similarity_pct"
                    ],

                "driver_score_pct":
                    top[
                        "driver_score_pct"
                    ],

                "driver_quality_pct":
                    top[
                        "driver_quality_pct"
                    ],

                "success_rate_pct":
                    top[
                        "success_rate_pct"
                    ],

                "result_reproducibility_pct":
                    top[
                        "result_reproducibility_pct"
                    ],

                "resolved_rate_pct":
                    top[
                        "resolved_rate_pct"
                    ],

                "ambiguous_rate_pct":
                    top[
                        "ambiguous_rate_pct"
                    ],

                "no_hit_rate_pct":
                    top[
                        "no_hit_rate_pct"
                    ],

                "plus5_first_pct":
                    top[
                        "plus5_first_pct"
                    ],

                "plus10_first_pct":
                    top[
                        "plus10_first_pct"
                    ],

                "plus20_first_pct":
                    top[
                        "plus20_first_pct"
                    ],

                "minus5_first_pct":
                    top[
                        "minus5_first_pct"
                    ],

                "recent_plus5_first_pct":
                    top[
                        "recent_plus5_first_pct"
                    ],

                "recent_plus10_first_pct":
                    top[
                        "recent_plus10_first_pct"
                    ],

                "recent_plus20_first_pct":
                    top[
                        "recent_plus20_first_pct"
                    ],

                "recent_minus5_first_pct":
                    top[
                        "recent_minus5_first_pct"
                    ],

                "recent_result_reproducibility_pct":
                    top[
                        "recent_result_reproducibility_pct"
                    ],

                "days_since_last_occurrence":
                    top[
                        "days_since_last_occurrence"
                    ],

                "expected_next_cycle":
                    top[
                        "expected_next_cycle"
                    ],

                "cycle_phase_pct":
                    top[
                        "cycle_phase_pct"
                    ],

                "target1":
                    top["target1"],

                "target2":
                    top["target2"],

                "invalidation":
                    top[
                        "invalidation"
                    ],

                "occurrences":
                    top[
                        "occurrences"
                    ],

                "cycle_status":
                    top[
                        "cycle_status"
                    ],
            }
        )

    print(
        f"[OK] {symbol}: "
        f"history={len(df)} "
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
            f"cluster={profile['cluster_id']} | "
            f"sim={safe_float(profile['current_similarity_pct']):.1f}% | "
            f"+5first={safe_float(profile['plus5_first_pct']):.1f}% | "
            f"+10first={safe_float(profile['plus10_first_pct']):.1f}% | "
            f"+20first={safe_float(profile['plus20_first_pct']):.1f}% | "
            f"-5first={safe_float(profile['minus5_first_pct']):.1f}% | "
            f"result_repro={safe_float(profile['result_reproducibility_pct']):.1f}% | "
            f"recent_repro={safe_float(profile['recent_result_reproducibility_pct']):.1f}% | "
            f"cycle={safe_float(profile['cycle_mode_days']):.1f}d | "
            f"cycle_stab={safe_float(profile['cycle_stability_pct']):.1f}% | "
            f"cycle_conf={safe_float(profile['cycle_confidence_pct']):.1f}% | "
            f"cycle_obs={profile['cycle_observations']} | "
            f"occ={profile['occurrences']} | "
            f"quality={safe_float(profile['driver_quality_pct']):.1f}% | "
            f"score={safe_float(profile['driver_score_pct']):.1f}% | "
            f"validation={profile['validation_status']} | "
            f"T1={safe_float(profile['target1']):.2f} | "
            f"T2={safe_float(profile['target2']):.2f} | "
            f"INV={safe_float(profile['invalidation']):.2f}"
        )

    return (
        profiles,
        current_rows,
        history_rows,
    )


# ============================================================
# SYMBOLS
# ============================================================

def load_symbols_from_project():
    cli_symbols = [
        str(value).strip().upper()
        for value in sys.argv[1:]
        if str(value).strip()
    ]

    if cli_symbols:
        return list(
            dict.fromkeys(
                cli_symbols
            )
        )

    env_symbols = os.getenv(
        "DRIVER_SYMBOLS",
        "",
    )

    if env_symbols.strip():
        return list(
            dict.fromkeys(
                value.strip().upper()
                for value
                in env_symbols.split(",")
                if value.strip()
            )
        )

    try:
        from selected_symbols import (
            SELECTED_SYMBOLS
        )

        symbols = [
            str(value).strip().upper()
            for value
            in SELECTED_SYMBOLS
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
# SAVE
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
        " UNIVERSAL DRIVER ENGINE v6.4"
    )

    print(
        "=============================================="
    )

    print(
        f"symbols                 : {len(symbols)}"
    )

    print(
        f"lookback                : {LOOKBACK_PERIOD}"
    )

    print(
        f"window                  : {WINDOW}"
    )

    print(
        f"forward                 : {FORWARD}"
    )

    print(
        f"min_history             : {MIN_HISTORY}"
    )

    print(
        f"max_drivers             : {MAX_DRIVERS}"
    )

    print(
        f"cluster_sim             : {CLUSTER_SIM}"
    )

    print(
        f"event_spacing           : {EVENT_SPACING}"
    )

    print(
        f"cycle_min               : {CYCLE_MIN_DAYS}"
    )

    print(
        f"harmonic_tolerance      : "
        f"{HARMONIC_TOLERANCE}"
    )

    print(
        f"max_cycle_multiple      : "
        f"{MAX_CYCLE_MULTIPLE}"
    )

    print(
        f"min_driver_occurrences  : "
        f"{MIN_DRIVER_OCCURRENCES}"
    )

    print(
        f"min_cycle_observations  : "
        f"{MIN_CYCLE_OBSERVATIONS}"
    )

    print(
        f"min_cycle_stability     : "
        f"{MIN_CYCLE_STABILITY}"
    )

    print(
        f"cycle_confidence_floor  : "
        f"{CYCLE_CONFIDENCE_FLOOR}"
    )

    print(
        f"min_result_repro        : "
        f"{MIN_RESULT_REPRODUCIBILITY}"
    )

    print(
        f"min_recent_repro        : "
        f"{MIN_RECENT_REPRODUCIBILITY}"
    )

    print(
        f"min_resolved_results    : "
        f"{MIN_RESOLVED_RESULTS}"
    )

    print(
        f"validated_occurrences   : "
        f"{VALIDATED_OCCURRENCES}"
    )

    print(
        f"stable_occurrences      : "
        f"{STABLE_OCCURRENCES}"
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
            "No current Driver detected."
        )

    print()
    print(
        "===== DRIVER ENGINE END ====="
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    symbols = (
        load_symbols_from_project()
    )

    run(symbols)

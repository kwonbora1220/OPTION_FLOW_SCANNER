#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNIVERSAL DRIVER ENGINE v6.2
============================

종목별 과거 OHLCV에서 반복되는 "운전수(Driver)"를 탐색한다.

v6.2 핵심 변경
--------------
1. 미래 40일이 완전히 존재하는 역사 이벤트만 학습
2. Pattern Cluster + Cycle Cluster 구조 유지
3. Cycle 계산은 "인접 이벤트 간격"을 1차 근거로 사용
4. 전체 pairwise gap은 보조 검증으로만 사용
5. 1x / 2x / 3x harmonic cycle 통합
6. 2개 이벤트만 있는 경우 cycle validation 금지
7. 서로 다른 cycle을 억지로 별도 driver로 만들지 않음
8. 서로 다른 구조의 pattern만 별도 driver로 분리
9. 최근 재현성 계산
10. cycle stability 계산
11. validation 단계 강화
12. 최대 3개지만 실제로 검증되는 driver만 출력

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

CYCLE_SPLIT_RATIO = float(
    os.getenv(
        "DRIVER_CYCLE_SPLIT_RATIO",
        "2.50",
    )
)

CYCLE_TOLERANCE = float(
    os.getenv(
        "DRIVER_CYCLE_TOLERANCE",
        "0.30",
    )
)

HARMONIC_TOLERANCE = float(
    os.getenv(
        "DRIVER_HARMONIC_TOLERANCE",
        "0.25",
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

MAX_CYCLE_MULTIPLE = int(
    os.getenv(
        "DRIVER_MAX_CYCLE_MULTIPLE",
        "3",
    )
)

# 너무 비슷한 두 driver를 중복으로 뽑지 않기 위한 기준
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


# ============================================================
# HELPERS
# ============================================================

def rnd(value, digits=2):

    try:

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return round(value, digits)

    except Exception:
        return np.nan


def pct(value, digits=2):

    try:

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return round(value * 100.0, digits)

    except Exception:
        return np.nan


def safe_float(value, default=np.nan):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


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

    series = finite_values(values)

    if series.empty:
        return np.nan

    return float(series.mean())


def safe_median(values):

    series = finite_values(values)

    if series.empty:
        return np.nan

    return float(series.median())


def safe_std(values):

    series = finite_values(values)

    if len(series) < 2:
        return np.nan

    return float(series.std(ddof=0))


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
        volume
        / median_volume
    )

    x["ma20_gap"] = (
        close
        / x["ma20"].replace(
            0,
            np.nan,
        )
        - 1
    )

    x["ma50_gap"] = (
        close
        / x["ma50"].replace(
            0,
            np.nan,
        )
        - 1
    )

    x["ma100_gap"] = (
        close
        / x["ma100"].replace(
            0,
            np.nan,
        )
        - 1
    )

    x["ma200_gap"] = (
        close
        / x["ma200"].replace(
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

        filled = np.nan_to_num(
            array,
            nan=median,
        )

        return (
            filled
            - np.mean(finite)
        ) / std

    filled = np.nan_to_num(
        array,
        nan=median,
    )

    return (
        filled - median
    ) / (
        1.4826 * mad
    )


def safe_interpolate(
    values,
    length,
):

    array = np.asarray(
        values,
        dtype=float,
    )

    if len(array) != length:

        array = np.resize(
            array,
            length,
        )

    series = pd.Series(array)

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

    volume_index = np.linspace(
        0,
        n - 1,
        8,
    )

    volume_part = np.interp(
        volume_index,
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

        index = np.linspace(
            0,
            n - 1,
            4,
        )

        ma_parts.extend(
            np.interp(
                index,
                np.arange(n),
                values,
            )
        )

    volatility = safe_interpolate(
        window["vol20"],
        n,
    )

    volatility_index = np.linspace(
        0,
        n - 1,
        4,
    )

    volatility_part = np.interp(
        volatility_index,
        np.arange(n),
        volatility,
    )

    last = window.iloc[-1]

    scalar = np.array(
        [
            last.get(
                "hh20",
                0,
            ),
            last.get(
                "ll20",
                0,
            ),
            last.get(
                "hh60",
                0,
            ),
            last.get(
                "ll60",
                0,
            ),
            last.get(
                "slope20",
                0,
            ),
            last.get(
                "slope50",
                0,
            ),
            last.get(
                "atr_pct",
                0,
            ),
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

    return (
        signature / norm
    )


def cosine_similarity(
    a,
    b,
):

    if len(a) != len(b):
        return 0.0

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if (
        norm_a < 1e-12
        or norm_b < 1e-12
    ):

        return 0.0

    similarity = (
        np.dot(a, b)
        / (
            norm_a
            * norm_b
        )
    )

    return float(
        np.clip(
            similarity,
            -1,
            1,
        )
    )


# ============================================================
# PATTERN LABEL
# ============================================================

def choose_pattern_label(
    window,
):

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

    first = close.iloc[
        :half
    ]

    second = close.iloc[
        half:
    ]

    first_move = (
        first.iloc[-1]
        / first.iloc[0]
        - 1
    )

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
        and trough_pos
        < len(close) * 0.80
        and total > -0.10
    ):

        return (
            "급락→저점형성→회복"
        )

    if (
        peak_return >= 0.20
        and trough_pos > peak_pos
        and second_move > 0.05
    ):

        return (
            "급등→조정→재상승"
        )

    if (
        np.isfinite(current_vol)
        and np.isfinite(median_vol)
        and median_vol > 0
        and current_vol
        > median_vol * 1.15
        and total > 0.10
    ):

        return (
            "압축→돌파→확장"
        )

    if (
        total > 0.12
        and ma20_gap > 0
        and ma50_gap > -0.02
        and slope20 > 0
    ):

        return (
            "추세상승→확장"
        )

    if (
        total < -0.12
        and ma20_gap < 0
        and ma50_gap < 0
        and slope20 < 0
    ):

        return (
            "추세하락→반등시도"
        )

    if abs(total) < 0.10:

        return (
            "박스→반복회귀"
        )

    return "추세전환형"


# ============================================================
# ORDERED OUTCOME
# ============================================================

def empty_ordered_outcome():

    return {
        "first_event":
            "NO_FUTURE",

        "first_day":
            np.nan,

        "plus5_first":
            np.nan,

        "plus10_first":
            np.nan,

        "plus20_first":
            np.nan,

        "minus5_first":
            np.nan,

        "days_to_plus5":
            np.nan,

        "days_to_plus10":
            np.nan,

        "days_to_plus20":
            np.nan,

        "days_to_minus5":
            np.nan,

        "outcome_status":
            "NO_FUTURE",
    }


def ordered_outcome(
    df,
    end_idx,
    forward=FORWARD,
):

    if end_idx >= len(df) - 1:

        return empty_ordered_outcome()

    entry = safe_float(
        df["Close"].iloc[end_idx]
    )

    if (
        not np.isfinite(entry)
        or entry <= 0
    ):

        return empty_ordered_outcome()

    target5 = (
        entry * 1.05
    )

    target10 = (
        entry * 1.10
    )

    target20 = (
        entry * 1.20
    )

    stop5 = (
        entry * 0.95
    )

    end = min(
        len(df),
        end_idx + 1 + forward,
    )

    first_plus5 = None
    first_plus10 = None
    first_plus20 = None
    first_minus5 = None

    for j in range(
        end_idx + 1,
        end,
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

        if (
            first_plus20 is not None
            and first_minus5 is not None
        ):

            break

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
            (
                "PLUS_5",
                first_plus5,
            )
        )

    if first_minus5 is not None:

        candidates.append(
            (
                "MINUS_5",
                first_minus5,
            )
        )

    if candidates:

        candidates.sort(
            key=lambda x: x[1]
        )

        if (
            len(candidates) >= 2
            and candidates[0][1]
            == candidates[1][1]
        ):

            first_event = (
                "AMBIGUOUS"
            )

            first_day = (
                candidates[0][1]
            )

        else:

            first_event = (
                candidates[0][0]
            )

            first_day = (
                candidates[0][1]
            )

    else:

        first_event = "NONE"
        first_day = np.nan

    if first_event == "AMBIGUOUS":

        outcome_status = (
            "AMBIGUOUS"
        )

    elif first_event in {
        "PLUS_5",
        "MINUS_5",
    }:

        outcome_status = (
            "RESOLVED"
        )

    else:

        outcome_status = (
            "NO_HIT"
        )

    return {

        "first_event":
            first_event,

        "first_day":
            first_day,

        "plus5_first":
            positive_first(
                first_plus5,
                first_minus5,
            ),

        "plus10_first":
            positive_first(
                first_plus10,
                first_minus5,
            ),

        "plus20_first":
            positive_first(
                first_plus20,
                first_minus5,
            ),

        "minus5_first":
            negative_first(
                first_minus5,
                first_plus5,
            ),

        "days_to_plus5":
            (
                first_plus5
                if first_plus5 is not None
                else np.nan
            ),

        "days_to_plus10":
            (
                first_plus10
                if first_plus10 is not None
                else np.nan
            ),

        "days_to_plus20":
            (
                first_plus20
                if first_plus20 is not None
                else np.nan
            ),

        "days_to_minus5":
            (
                first_minus5
                if first_minus5 is not None
                else np.nan
            ),

        "outcome_status":
            outcome_status,
    }


# ============================================================
# FUTURE SUMMARY
# ============================================================

def future_summary(
    df,
    end_idx,
):

    entry = safe_float(
        df["Close"].iloc[end_idx]
    )

    if (
        not np.isfinite(entry)
        or entry <= 0
    ):

        return {}

    future = df.iloc[
        end_idx + 1:
        min(
            end_idx + 1 + FORWARD,
            len(df),
        )
    ]

    if future.empty:
        return {}

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

    ordered = ordered_outcome(
        df,
        end_idx,
        forward=FORWARD,
    )

    return {

        "forward_return":
            safe_float(
                close_ret.iloc[-1]
            ),

        "max_up":
            safe_float(
                high_ret.max()
            ),

        "max_down":
            safe_float(
                low_ret.min()
            ),

        "ordered":
            ordered,
    }


# ============================================================
# CYCLE ENGINE v6.2
# ============================================================

def adjacent_gaps(events):

    if len(events) < 2:
        return []

    ordered = sorted(
        events,
        key=lambda e: pd.Timestamp(
            e["date"]
        ),
    )

    gaps = []

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

            gaps.append(
                {
                    "gap": float(gap),
                    "from_date":
                        previous["date"],
                    "to_date":
                        current["date"],
                }
            )

    return gaps


def pairwise_gaps(events):

    if len(events) < 2:
        return []

    ordered = sorted(
        events,
        key=lambda e: pd.Timestamp(
            e["date"]
        ),
    )

    gaps = []

    for i in range(
        len(ordered)
    ):

        for j in range(
            i + 1,
            len(ordered)
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

                gaps.append(
                    float(gap)
                )

    return gaps


def harmonic_ratio(
    gap,
    base,
):

    if (
        not np.isfinite(gap)
        or not np.isfinite(base)
        or base <= 0
    ):

        return np.nan

    ratio = gap / base

    nearest = max(
        1,
        min(
            MAX_CYCLE_MULTIPLE,
            int(
                round(ratio)
            ),
        ),
    )

    return ratio / nearest


def fits_harmonic(
    gap,
    base,
    tolerance=HARMONIC_TOLERANCE,
):

    if (
        not np.isfinite(gap)
        or not np.isfinite(base)
        or base <= 0
    ):

        return False

    ratio = gap / base

    nearest = int(
        round(ratio)
    )

    if nearest < 1:
        return False

    if nearest > MAX_CYCLE_MULTIPLE:
        return False

    relative_error = abs(
        ratio - nearest
    ) / nearest

    return (
        relative_error
        <= tolerance
    )


def candidate_cycle_bases(
    gaps
):

    values = sorted(
        set(
            int(round(gap))
            for gap in gaps
            if (
                np.isfinite(gap)
                and gap >= CYCLE_MIN_DAYS
            )
        )
    )

    candidates = set()

    for gap in values:

        candidates.add(
            float(gap)
        )

        for multiple in range(
            2,
            MAX_CYCLE_MULTIPLE + 1,
        ):

            base = gap / multiple

            if base >= CYCLE_MIN_DAYS:

                candidates.add(
                    round(
                        base,
                        2,
                    )
                )

    return sorted(
        candidates
    )


def cycle_fit_score(
    base,
    adjacent,
    pairwise,
):

    if not adjacent:
        return None

    adjacent_values = [
        float(item["gap"])
        for item in adjacent
    ]

    adjacent_fit = [
        gap
        for gap in adjacent_values
        if fits_harmonic(
            gap,
            base,
        )
    ]

    pairwise_fit = [
        gap
        for gap in pairwise
        if fits_harmonic(
            gap,
            base,
        )
    ]

    adjacent_support = len(
        adjacent_fit
    )

    pairwise_support = len(
        pairwise_fit
    )

    # 인접 gap이 핵심.
    # pairwise는 보조점수.
    score = (
        adjacent_support * 10
        + pairwise_support * 0.25
    )

    return {
        "base": float(base),
        "adjacent_support":
            adjacent_support,
        "pairwise_support":
            pairwise_support,
        "adjacent_values":
            adjacent_values,
        "adjacent_fit":
            adjacent_fit,
        "pairwise_fit":
            pairwise_fit,
        "score":
            float(score),
    }


def select_cycle_base(
    events
):

    adjacent = adjacent_gaps(
        events
    )

    if len(adjacent) < 1:

        return {
            "valid": False,
            "reason":
                "NO_ADJACENT_GAP",
            "base": np.nan,
            "observations": 0,
            "stability": np.nan,
            "adjacent": [],
            "pairwise": [],
        }

    pairwise = pairwise_gaps(
        events
    )

    gaps = [
        item["gap"]
        for item in adjacent
    ]

    candidates = candidate_cycle_bases(
        gaps
    )

    if not candidates:

        return {
            "valid": False,
            "reason":
                "NO_CYCLE_BASE",
            "base": np.nan,
            "observations": 0,
            "stability": np.nan,
            "adjacent": adjacent,
            "pairwise": pairwise,
        }

    fits = []

    for base in candidates:

        result = cycle_fit_score(
            base,
            adjacent,
            pairwise,
        )

        if result is not None:

            fits.append(
                result
            )

    if not fits:

        return {
            "valid": False,
            "reason":
                "NO_CYCLE_FIT",
            "base": np.nan,
            "observations": 0,
            "stability": np.nan,
            "adjacent": adjacent,
            "pairwise": pairwise,
        }

    # --------------------------------------------------------
    # 우선순위
    #
    # 1. 인접 support
    # 2. 평균 오차가 작은 base
    # 3. base가 지나치게 작은 harmonic을 선택하지 않도록
    # --------------------------------------------------------

    def fit_error(item):

        base = item["base"]

        errors = []

        for gap in item[
            "adjacent_values"
        ]:

            if fits_harmonic(
                gap,
                base,
            ):

                ratio = (
                    gap / base
                )

                nearest = max(
                    1,
                    min(
                        MAX_CYCLE_MULTIPLE,
                        int(
                            round(ratio)
                        ),
                    ),
                )

                errors.append(
                    abs(
                        ratio
                        - nearest
                    )
                    / nearest
                )

        if not errors:
            return 999.0

        return float(
            np.mean(errors)
        )

    fits.sort(
        key=lambda item: (
            -item[
                "adjacent_support"
            ],
            fit_error(item),
            -item[
                "pairwise_support"
            ],
            item["base"],
        )
    )

    best = fits[0]

    observations = (
        best["adjacent_support"]
    )

    # --------------------------------------------------------
    # cycle stability
    # --------------------------------------------------------

    matched = []

    for gap in best[
        "adjacent_values"
    ]:

        if fits_harmonic(
            gap,
            best["base"],
        ):

            ratio = gap / best["base"]

            multiple = max(
                1,
                min(
                    MAX_CYCLE_MULTIPLE,
                    int(
                        round(ratio)
                    ),
                ),
            )

            normalized_gap = (
                gap / multiple
            )

            matched.append(
                normalized_gap
            )

    if len(matched) >= 2:

        median_cycle = float(
            np.median(matched)
        )

        deviations = [
            abs(
                value
                - median_cycle
            )
            / max(
                median_cycle,
                1,
            )
            for value in matched
        ]

        mean_deviation = float(
            np.mean(
                deviations
            )
        )

        stability = max(
            0.0,
            1.0
            - mean_deviation
        )

    elif len(matched) == 1:

        median_cycle = float(
            matched[0]
        )

        stability = 1.0

    else:

        median_cycle = (
            best["base"]
        )

        stability = 0.0

    valid = (
        observations
        >= MIN_CYCLE_OBSERVATIONS
    )

    return {

        "valid":
            valid,

        "reason":
            (
                "VALID"
                if valid
                else
                "INSUFFICIENT_RECURRENCE"
            ),

        "base":
            float(
                median_cycle
            ),

        "observations":
            int(observations),

        "stability":
            float(stability),

        "mean":
            safe_mean(matched),

        "std":
            safe_std(matched),

        "adjacent":
            adjacent,

        "pairwise":
            pairwise,

        "matched":
            matched,
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

    if (
        not np.isfinite(base)
        or base <= 0
    ):

        return list(events)

    ordered = sorted(
        events,
        key=lambda e: pd.Timestamp(
            e["date"]
        ),
    )

    # --------------------------------------------------------
    # 인접 gap을 기준으로 같은 cycle sequence를 연결한다.
    # harmonic 1x/2x/3x 모두 같은 운전수로 인정.
    # --------------------------------------------------------

    groups = []

    current = [
        ordered[0]
    ]

    for event in ordered[1:]:

        previous = current[-1]

        gap = (
            pd.Timestamp(
                event["date"]
            )
            - pd.Timestamp(
                previous["date"]
            )
        ).days

        if fits_harmonic(
            gap,
            base,
        ):

            current.append(
                event
            )

        else:

            if len(current) >= 2:

                groups.append(
                    current
                )

            current = [
                event
            ]

    if len(current) >= 2:

        groups.append(
            current
        )

    # 연결된 sequence가 없으면
    # 전체 events를 하나의 그룹으로 유지.
    if not groups:

        return [
            ordered
        ]

    # groups 사이에 실제로 중복 event가 없는지 확인.
    result = []

    seen = set()

    for group in groups:

        clean_group = []

        for event in group:

            key = str(
                event["date"]
            )

            if key not in seen:

                clean_group.append(
                    event
                )

                seen.add(key)

        if clean_group:

            result.append(
                clean_group
            )

    return result


# ============================================================
# EVENT EXTRACTION
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

    window = df.iloc[
        start_idx:
        end_idx + 1
    ]

    if len(window) < WINDOW:
        return None

    signature = make_signature(
        window
    )

    pattern = choose_pattern_label(
        window
    )

    future = future_summary(
        df,
        end_idx,
    )

    if not future:
        return None

    date = df.index[
        end_idx
    ]

    close = safe_float(
        df["Close"].iloc[end_idx]
    )

    if (
        not np.isfinite(close)
        or close <= 0
    ):

        return None

    return {

        "date":
            date,

        "end_idx":
            int(end_idx),

        "price":
            close,

        "pattern":
            pattern,

        "signature":
            signature,

        "forward_return":
            future[
                "forward_return"
            ],

        "max_up":
            future[
                "max_up"
            ],

        "max_down":
            future[
                "max_down"
            ],

        "ordered":
            future[
                "ordered"
            ],
    }


def extract_events(df):

    events = []

    # --------------------------------------------------------
    # 중요:
    # 마지막 FORWARD일은 완전한 outcome이 없으므로 제외.
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # 너무 가까운 이벤트를 제거.
        # 같은 상승/하락 한 구간을 여러 번 세지 않음.
        # ----------------------------------------------------

        if (
            last_event_idx is not None
            and (
                end_idx
                - last_event_idx
                < EVENT_SPACING
            )
        ):

            # 현재 event가 이전보다
            # 더 현재에 가까운 event이므로
            # 기본적으로 이전 event 유지.
            continue

        events.append(
            event
        )

        last_event_idx = (
            end_idx
        )

    return events


# ============================================================
# PATTERN CLUSTER
# ============================================================

def cluster_events(
    events
):

    if not events:
        return []

    clusters = []

    for event in events:

        best_cluster = None
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
                similarity
                >= CLUSTER_SIM
                and similarity
                > best_similarity
            ):

                best_cluster = cluster
                best_similarity = (
                    similarity
                )

        if best_cluster is None:

            clusters.append(
                {
                    "pattern":
                        event[
                            "pattern"
                        ],

                    "events":
                        [event],

                    "centroid":
                        event[
                            "signature"
                        ].copy(),
                }
            )

        else:

            best_cluster[
                "events"
            ].append(
                event
            )

            signatures = [
                item["signature"]
                for item
                in best_cluster[
                    "events"
                ]
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
                    centroid
                    / norm
                )

            best_cluster[
                "centroid"
            ] = centroid

    return clusters


# ============================================================
# BASE CLUSTER ID
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
        raw.encode(
            "utf-8"
        )
    ).hexdigest()[:10]

    return (
        f"{symbol}-BASE-{digest}"
    )


def make_cluster_id(
    symbol,
    cycle_type,
    base_cluster_id,
    cycle_mode,
):

    raw = (
        f"{symbol}|"
        f"{base_cluster_id}|"
        f"{cycle_type}|"
        f"{round(cycle_mode, 1)}"
    )

    digest = hashlib.sha1(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()[:10]

    return (
        f"{symbol}-"
        f"{cycle_type}-"
        f"{digest}"
    )


# ============================================================
# ORDERED STATISTICS
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

    first_plus5 = 0
    first_plus10 = 0
    first_plus20 = 0
    first_minus5 = 0

    success_count = 0

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

        value = ordered[
            "plus5_first"
        ]

        if np.isfinite(
            safe_float(value)
        ):

            plus5.append(
                value
            )

            if (
                safe_float(value)
                == 1.0
            ):

                first_plus5 += 1

        value = ordered[
            "plus10_first"
        ]

        if np.isfinite(
            safe_float(value)
        ):

            plus10.append(
                value
            )

            if (
                safe_float(value)
                == 1.0
            ):

                first_plus10 += 1

        value = ordered[
            "plus20_first"
        ]

        if np.isfinite(
            safe_float(value)
        ):

            plus20.append(
                value
            )

            if (
                safe_float(value)
                == 1.0
            ):

                first_plus20 += 1

        value = ordered[
            "minus5_first"
        ]

        if np.isfinite(
            safe_float(value)
        ):

            minus5.append(
                value
            )

            if (
                safe_float(value)
                == 1.0
            ):

                first_minus5 += 1

        if (
            ordered[
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
            first_plus5,

        "plus10_first_count":
            first_plus10,

        "plus20_first_count":
            first_plus20,

        "minus5_first_count":
            first_minus5,

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
# RECENT STATISTICS
# ============================================================

def recent_statistics(
    events
):

    if not events:

        return {
            "recent_occurrences":
                0,

            "recent_plus5_first_pct":
                np.nan,

            "recent_plus10_first_pct":
                np.nan,

            "recent_plus20_first_pct":
                np.nan,

            "recent_minus5_first_pct":
                np.nan,
        }

    ordered = sorted(
        events,
        key=lambda e: pd.Timestamp(
            e["date"]
        ),
    )

    recent = ordered[
        -RECENT_OCCURRENCES:
    ]

    stats = calculate_ordered_stats(
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
    }


# ============================================================
# VALIDATION
# ============================================================

def validation_status(
    occurrences,
    cycle_observations,
    cycle_stability,
):

    if occurrences < MIN_DRIVER_OCCURRENCES:

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
        np.isfinite(
            safe_float(
                cycle_stability
            )
        )
        and cycle_stability
        < MIN_CYCLE_STABILITY
    ):

        return (
            "LOW_CYCLE_STABILITY"
        )

    if occurrences < VALIDATED_OCCURRENCES:

        return "REPEAT_DETECTED"

    if occurrences < STABLE_OCCURRENCES:

        return "VALIDATED"

    return "STABLE"


def driver_state(
    current_similarity,
    validation,
):

    if validation in {
        "INSUFFICIENT_OCCURRENCES",
        "LOW_CYCLE_REPEAT",
    }:

        return "PROVISIONAL"

    if validation == (
        "LOW_CYCLE_STABILITY"
    ):

        if (
            current_similarity
            >= CURRENT_ACTIVE_SIM
        ):

            return "WATCH"

        return "INACTIVE"

    if (
        current_similarity
        >= CURRENT_ACTIVE_SIM
    ):

        return "ACTIVE"

    if (
        current_similarity
        >= CURRENT_WATCH_SIM
    ):

        return "WATCH"

    return "INACTIVE"


# ============================================================
# TARGET / INVALIDATION
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
            event[
                "max_up"
            ]
            for event in events
        ]
    )

    downs = finite_values(
        [
            event[
                "max_down"
            ]
            for event in events
        ]
    )

    if ups.empty:

        target1_return = 0.05
        target2_return = 0.10

    else:

        target1_return = max(
            0.05,
            min(
                0.10,
                float(
                    ups.median()
                ),
            ),
        )

        target2_return = max(
            target1_return,
            min(
                0.30,
                float(
                    ups.quantile(
                        0.75
                    )
                ),
            ),
        )

    if downs.empty:

        invalidation_return = (
            -0.05
        )

    else:

        invalidation_return = min(
            -0.05,
            float(
                downs.median()
            ),
        )

    target1 = (
        current_price
        * (
            1
            + target1_return
        )
    )

    target2 = (
        current_price
        * (
            1
            + target2_return
        )
    )

    invalidation = (
        current_price
        * (
            1
            + invalidation_return
        )
    )

    return (
        target1,
        target2,
        invalidation,
    )


# ============================================================
# DRIVER SCORE
# ============================================================

def driver_score(
    current_similarity,
    occurrences,
    plus5_first_rate,
    cycle_stability,
    recent_plus5,
):

    def clamp(
        value,
        default=0.0,
    ):

        value = safe_float(
            value,
            default,
        )

        if not np.isfinite(value):
            return default

        return float(
            np.clip(
                value,
                0,
                1,
            )
        )

    sim = clamp(
        current_similarity
    )

    occurrence_score = clamp(
        min(
            occurrences / 8.0,
            1.0,
        )
    )

    plus5 = clamp(
        plus5_first_rate
    )

    cycle = clamp(
        cycle_stability
    )

    recent = clamp(
        recent_plus5
    )

    score = (
        sim * 0.30
        + occurrence_score * 0.15
        + plus5 * 0.20
        + cycle * 0.20
        + recent * 0.15
    )

    return float(
        score
    )


# ============================================================
# DUPLICATE CHECK
# ============================================================

def is_duplicate_driver(
    candidate,
    selected,
):

    for existing in selected:

        base_similarity = (
            cosine_similarity(
                candidate[
                    "cluster"
                ][
                    "centroid"
                ],
                existing[
                    "cluster"
                ][
                    "centroid"
                ],
            )
        )

        if (
            base_similarity
            >= DUPLICATE_SIM
        ):

            c1 = safe_float(
                candidate[
                    "cycle_mode"
                ]
            )

            c2 = safe_float(
                existing[
                    "cycle_mode"
                ]
            )

            if (
                np.isfinite(c1)
                and np.isfinite(c2)
            ):

                ratio = (
                    max(c1, c2)
                    / max(
                        min(c1, c2),
                        1,
                    )
                )

                if (
                    ratio
                    < DUPLICATE_CYCLE_RATIO
                ):

                    return True

            else:

                return True

    return False


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

def analyze_symbol(
    symbol
):

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

        return (
            [],
            [],
            [],
        )

    if len(raw) < MIN_HISTORY:

        print(
            f"[SKIP] {symbol}: "
            f"history={len(raw)} "
            f"< {MIN_HISTORY}"
        )

        return (
            [],
            [],
            [],
        )

    df = add_features(
        raw
    )

    # feature warmup
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

        return (
            [],
            [],
            [],
        )

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

        return (
            [],
            [],
            [],
        )

    # ========================================================
    # Pattern clustering
    # ========================================================

    pattern_clusters = cluster_events(
        events
    )

    candidates = []

    # ========================================================
    # 각 pattern cluster마다 cycle 분석
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

        # ----------------------------------------------------
        # cycle이 검증되지 않은 pattern은
        # 무조건 별도 driver로 만들지 않는다.
        # 단, 3개 이상 occurrence가 있으면
        # "반복은 있으나 cycle 미검증" 상태로 보존.
        # ----------------------------------------------------

        if (
            not np.isfinite(
                cycle_mode
            )
            or cycle_mode <= 0
        ):

            cycle_mode = np.nan

        cycle_type = classify_cycle(
            cycle_result
        )

        grouped_event_sets = []

        if cycle_result.get(
            "valid",
            False,
        ):

            grouped_event_sets = (
                cycle_group_events(
                    cluster_events_list,
                    cycle_result,
                )
            )

        else:

            grouped_event_sets = [
                cluster_events_list
            ]

        for event_group in (
            grouped_event_sets
        ):

            if len(event_group) < (
                MIN_DRIVER_OCCURRENCES
            ):

                continue

            # ------------------------------------------------
            # cycle group 내부에서
            # 현재 구조와 가장 가까운 historical event
            # 를 기준으로 centroid similarity 계산
            # ------------------------------------------------

            signatures = [
                event[
                    "signature"
                ]
                for event
                in event_group
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

            cluster_copy = {
                "pattern":
                    cluster[
                        "pattern"
                    ],

                "events":
                    event_group,

                "centroid":
                    centroid,
            }

            candidates.append(
                {
                    "cluster":
                        cluster_copy,

                    "pattern":
                        cluster[
                            "pattern"
                        ],

                    "cycle_type":
                        cycle_type,

                    "cycle_mode":
                        cycle_mode,

                    "cycle_mean":
                        safe_float(
                            cycle_result.get(
                                "mean"
                            )
                        ),

                    "cycle_std":
                        safe_float(
                            cycle_result.get(
                                "std"
                            )
                        ),

                    "cycle_stability":
                        cycle_stability,

                    "cycle_observations":
                        cycle_observations,

                    "events":
                        event_group,
                }
            )

    if not candidates:

        print(
            f"[SKIP] {symbol}: "
            "no driver candidates"
        )

        return (
            [],
            [],
            [],
        )

    # ========================================================
    # Current similarity
    # ========================================================

    current_window = df.iloc[
        -WINDOW:
    ]

    current_signature = (
        make_signature(
            current_window
        )
    )

    current_price = safe_float(
        df["Close"].iloc[-1]
    )

    current_pattern = (
        choose_pattern_label(
            current_window
        )
    )

    phase = (
        "UPTREND"
        if (
            safe_float(
                df["ma20_gap"].iloc[-1],
                0,
            ) > 0
            and safe_float(
                df["ma50_gap"].iloc[-1],
                0,
            ) > 0
        )
        else
        "DOWNTREND"
        if (
            safe_float(
                df["ma20_gap"].iloc[-1],
                0,
            ) < 0
            and safe_float(
                df["ma50_gap"].iloc[-1],
                0,
            ) < 0
        )
        else
        "TRANSITION"
    )

    for candidate in candidates:

        candidate[
            "current_similarity"
        ] = cosine_similarity(
            current_signature,
            candidate[
                "cluster"
            ][
                "centroid"
            ],
        )

        stats = calculate_ordered_stats(
            candidate["events"]
        )

        recent = recent_statistics(
            candidate["events"]
        )

        candidate[
            "ordered"
        ] = stats

        candidate[
            "recent"
        ] = recent

        candidate[
            "occurrences"
        ] = len(
            candidate["events"]
        )

        candidate[
            "success_count"
        ] = stats[
            "success_count"
        ]

        candidate[
            "success_rate"
        ] = stats[
            "success_rate"
        ]

        candidate[
            "score"
        ] = driver_score(
            candidate[
                "current_similarity"
            ],
            candidate[
                "occurrences"
            ],
            stats[
                "plus5_first_rate"
            ],
            candidate[
                "cycle_stability"
            ],
            (
                safe_float(
                    recent[
                        "recent_plus5_first_pct"
                )
                / 100
                )
                if np.isfinite(
                    safe_float(
                        recent[
                            "recent_plus5_first_pct"
                        ]
                    )
                )
                else np.nan
            ),
        )

        candidate[
            "validation_status"
        ] = validation_status(
            candidate[
                "occurrences"
            ],
            candidate[
                "cycle_observations"
            ],
            candidate[
                "cycle_stability"
            ],
        )

        candidate[
            "state"
        ] = driver_state(
            candidate[
                "current_similarity"
            ],
            candidate[
                "validation_status"
            ],
        )

        (
            target1,
            target2,
            invalidation,
        ) = calculate_targets(
            current_price,
            candidate[
                "events"
            ],
        )

        candidate[
            "target1"
        ] = target1

        candidate[
            "target2"
        ] = target2

        candidate[
            "invalidation"
        ] = invalidation

        base_cluster_id = (
            make_base_cluster_id(
                symbol,
                candidate[
                    "pattern"
                ],
                candidate[
                    "cluster"
                ][
                    "centroid"
                ],
            )
        )

        candidate[
            "base_cluster_id"
        ] = base_cluster_id

        candidate[
            "cluster_id"
        ] = make_cluster_id(
            symbol,
            candidate[
                "cycle_type"
            ],
            base_cluster_id,
            (
                candidate[
                    "cycle_mode"
                ]
                if np.isfinite(
                    safe_float(
                        candidate[
                            "cycle_mode"
                        ]
                    )
                )
                else 0
            ),
        )

        candidate[
            "forward_returns"
        ] = finite_values(
            [
                event[
                    "forward_return"
                ]
                for event
                in candidate[
                    "events"
                ]
            ]
        )

        candidate[
            "max_ups"
        ] = finite_values(
            [
                event[
                    "max_up"
                ]
                for event
                in candidate[
                    "events"
                ]
            ]
        )

        candidate[
            "max_downs"
        ] = finite_values(
            [
                event[
                    "max_down"
                ]
                for event
                in candidate[
                    "events"
                ]
            ]
        )

    # ========================================================
    # Candidate filtering
    # ========================================================

    # 검증되지 않은 단순 duplicate 제거
    candidates.sort(
        key=lambda item: (
            -safe_float(
                item["score"],
                0,
            ),

            -safe_float(
                item[
                    "current_similarity"
                ],
                0,
            ),

            -item[
                "occurrences"
            ],

            -safe_float(
                item[
                    "cycle_stability"
                ],
                0,
            ),
        )
    )

    selected = []

    for candidate in candidates:

        if is_duplicate_driver(
            candidate,
            selected,
        ):

            continue

        selected.append(
            candidate
        )

        if len(selected) >= (
            MAX_DRIVERS
        ):

            break

    if not selected:

        return (
            [],
            [],
            [],
        )

    # ========================================================
    # Driver rank
    # ========================================================

    profiles = []
    current_rows = []
    history_rows = []

    for rank, item in enumerate(
        selected,
        start=1,
    ):

        cluster_events_list = (
            item[
                "cluster"
            ][
                "events"
            ]
        )

        driver_name = (
            f"DRIVER_{rank}"
        )

        display_pattern = (
            f"[{item['cycle_type']}] "
            f"{item['pattern']}"
        )

        item[
            "driver_rank"
        ] = rank

        item[
            "driver_name"
        ] = driver_name

        item[
            "display_pattern"
        ] = display_pattern

        profile = {

            "symbol":
                symbol,

            "driver_rank":
                rank,

            "driver_name":
                driver_name,

            "cluster_id":
                item[
                    "cluster_id"
                ],

            "base_cluster_id":
                item[
                    "base_cluster_id"
                ],

            "pattern":
                display_pattern,

            "cycle_type":
                item[
                    "cycle_type"
                ],

            "cycle_mode_days":
                rnd(
                    item[
                        "cycle_mode"
                    ],
                    1,
                ),

            "cycle_stability_pct":
                pct(
                    item[
                        "cycle_stability"
                    ]
                ),

            "cycle_observations":
                item[
                    "cycle_observations"
                ],

            "state":
                item[
                    "state"
                ],

            "validation_status":
                item[
                    "validation_status"
                ],

            "phase":
                phase,

            "current_pattern":
                current_pattern,

            "history_days":
                len(df),

            "current_price":
                rnd(
                    current_price,
                    4,
                ),

            "current_similarity_pct":
                rnd(
                    item[
                        "current_similarity"
                    ] * 100,
                    2,
                ),

            "driver_score_pct":
                rnd(
                    item[
                        "score"
                    ] * 100,
                    2,
                ),

            "occurrences":
                item[
                    "occurrences"
                ],

            "success_count":
                item[
                    "success_count"
                ],

            "success_rate_pct":
                pct(
                    item[
                        "success_rate"
                    ]
                ),

            "plus5_first_count":
                item[
                    "ordered"
                ][
                    "plus5_first_count"
                ],

            "plus5_first_rate_pct":
                pct(
                    item[
                        "ordered"
                    ][
                        "plus5_first_rate"
                    ]
                ),

            "plus5_valid_n":
                item[
                    "ordered"
                ][
                    "plus5_valid_n"
                ],

            "plus10_first_count":
                item[
                    "ordered"
                ][
                    "plus10_first_count"
                ],

            "plus10_first_rate_pct":
                pct(
                    item[
                        "ordered"
                    ][
                        "plus10_first_rate"
                    ]
                ),

            "plus10_valid_n":
                item[
                    "ordered"
                ][
                    "plus10_valid_n"
                ],

            "plus20_first_count":
                item[
                    "ordered"
                ][
                    "plus20_first_count"
                ],

            "plus20_first_rate_pct":
                pct(
                    item[
                        "ordered"
                    ][
                        "plus20_first_rate"
                    ]
                ),

            "plus20_valid_n":
                item[
                    "ordered"
                ][
                    "plus20_valid_n"
                ],

            "minus5_first_count":
                item[
                    "ordered"
                ][
                    "minus5_first_count"
                ],

            "minus5_first_rate_pct":
                pct(
                    item[
                        "ordered"
                    ][
                        "minus5_first_rate"
                    ]
                ),

            "minus5_valid_n":
                item[
                    "ordered"
                ][
                    "minus5_valid_n"
                ],

            "ambiguous_count":
                item[
                    "ordered"
                ][
                    "ambiguous_count"
                ],

            "no_hit_count":
                item[
                    "ordered"
                ][
                    "no_hit_count"
                ],

            "recent_occurrences":
                item[
                    "recent"
                ][
                    "recent_occurrences"
                ],

            "recent_plus5_first_pct":
                item[
                    "recent"
                ][
                    "recent_plus5_first_pct"
                ],

            "recent_plus10_first_pct":
                item[
                    "recent"
                ][
                    "recent_plus10_first_pct"
                ],

            "recent_plus20_first_pct":
                item[
                    "recent"
                ][
                    "recent_plus20_first_pct"
                ],

            "recent_minus5_first_pct":
                item[
                    "recent"
                ][
                    "recent_minus5_first_pct"
                ],

            "avg_forward_return_pct":
                rnd(
                    item[
                        "forward_returns"
                    ].mean()
                    * 100
                    if not item[
                        "forward_returns"
                    ].empty
                    else np.nan,
                    2,
                ),

            "median_forward_return_pct":
                rnd(
                    item[
                        "forward_returns"
                    ].median()
                    * 100
                    if not item[
                        "forward_returns"
                    ].empty
                    else np.nan,
                    2,
                ),

            "median_max_up_pct":
                rnd(
                    item[
                        "max_ups"
                    ].median()
                    * 100
                    if not item[
                        "max_ups"
                    ].empty
                    else np.nan,
                    2,
                ),

            "median_max_down_pct":
                rnd(
                    item[
                        "max_downs"
                    ].median()
                    * 100
                    if not item[
                        "max_downs"
                    ].empty
                    else np.nan,
                    2,
                ),

            "mean_cycle_days":
                rnd(
                    item[
                        "cycle_mean"
                    ],
                    1,
                ),

            "cycle_std_days":
                rnd(
                    item[
                        "cycle_std"
                    ],
                    1,
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

            "first_seen":
                min(
                    date_string(
                        event["date"]
                    )
                    for event
                    in cluster_events_list
                ),

            "last_seen":
                max(
                    date_string(
                        event["date"]
                    )
                    for event
                    in cluster_events_list
                ),
        }

        profiles.append(
            profile
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        for event in cluster_events_list:

            history_rows.append(
                make_history_row(
                    symbol,
                    rank,
                    driver_name,
                    item[
                        "cluster_id"
                    ],
                    display_pattern,
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
    # CURRENT DRIVER
    # ========================================================

    if profiles:

        top = profiles[0]

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

                "cycle_observations":
                    top[
                        "cycle_observations"
                    ],

                "state":
                    top[
                        "state"
                    ],

                "validation_status":
                    top[
                        "validation_status"
                    ],

                "phase":
                    top[
                        "phase"
                    ],

                "current_pattern":
                    top[
                        "current_pattern"
                    ],

                "similarity_pct":
                    top[
                        "current_similarity_pct"
                    ],

                "driver_score_pct":
                    top[
                        "driver_score_pct"
                    ],

                "success_rate_pct":
                    top[
                        "success_rate_pct"
                    ],

                "plus5_first_pct":
                    top[
                        "plus5_first_rate_pct"
                    ],

                "plus10_first_pct":
                    top[
                        "plus10_first_rate_pct"
                    ],

                "plus20_first_pct":
                    top[
                        "plus20_first_rate_pct"
                    ],

                "minus5_first_pct":
                    top[
                        "minus5_first_rate_pct"
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

                "target1":
                    top[
                        "target1"
                    ],

                "target2":
                    top[
                        "target2"
                    ],

                "invalidation":
                    top[
                        "invalidation"
                    ],

                "occurrences":
                    top[
                        "occurrences"
                    ],

                "cycle_status":
                    (
                        "VALIDATED"
                        if (
                            top[
                                "cycle_observations"
                            ]
                            >= MIN_CYCLE_OBSERVATIONS
                        )
                        else
                        "INSUFFICIENT_REPEAT"
                    ),
            }
        )

    # ========================================================
    # CONSOLE
    # ========================================================

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
            f"+5first={safe_float(profile['plus5_first_rate_pct']):.1f}% | "
            f"+10first={safe_float(profile['plus10_first_rate_pct']):.1f}% | "
            f"+20first={safe_float(profile['plus20_first_rate_pct']):.1f}% | "
            f"-5first={safe_float(profile['minus5_first_rate_pct']):.1f}% | "
            f"cycle={safe_float(profile['cycle_mode_days']):.1f}d | "
            f"cycle_stab={safe_float(profile['cycle_stability_pct']):.1f}% | "
            f"cycle_obs={profile['cycle_observations']} | "
            f"occ={profile['occurrences']} | "
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
# SYMBOL LOADING
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
# MAIN RUN
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
        " UNIVERSAL DRIVER ENGINE v6.2"
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
        f"cycle_split_ratio      : {CYCLE_SPLIT_RATIO}"
    )

    print(
        f"cycle_tolerance        : {CYCLE_TOLERANCE}"
    )

    print(
        f"harmonic_tolerance     : {HARMONIC_TOLERANCE}"
    )

    print(
        f"max_cycle_multiple     : {MAX_CYCLE_MULTIPLE}"
    )

    print(
        f"min_driver_occurrences : {MIN_DRIVER_OCCURRENCES}"
    )

    print(
        f"min_cycle_observations : {MIN_CYCLE_OBSERVATIONS}"
    )

    print(
        f"validated_occurrences  : {VALIDATED_OCCURRENCES}"
    )

    print(
        f"stable_occurrences     : {STABLE_OCCURRENCES}"
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


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    symbols = (
        load_symbols_from_project()
    )

    run(symbols)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNIVERSAL DRIVER ENGINE v6.3
============================

종목별 과거 OHLCV에서 반복되는 "운전수(Driver)"를 탐색한다.

v6.3 핵심
---------
1. 미래 FORWARD일이 완전히 존재하는 역사 이벤트만 학습
2. Pattern Cluster + Cycle Cluster 구조 유지
3. 인접 이벤트 gap을 cycle의 1차 근거로 사용
4. pairwise gap은 보조 검증으로만 사용
5. 1x / 2x / 3x harmonic cycle 통합
6. 3개 이벤트 미만에서는 cycle 검증 금지
7. 서로 다른 cycle 배수는 같은 Driver로 통합
8. 구조적으로 다른 pattern/signature만 별도 Driver
9. cycle base를 최소 harmonic base 기준으로 정규화
10. 최근 재현성 계산
11. cycle stability 계산
12. 현재 similarity와 historical reproducibility 분리
13. cycle 반복과 가격 결과 반복을 분리
14. 결과 재현성(result reproducibility) 계산
15. 최근 결과 재현성 별도 계산
16. AMBIGUOUS / NO_HIT 분리
17. 표본 수가 적은 Driver의 성공률 과대평가 방지
18. 최근 발생일 / 다음 예상 cycle 계산
19. cycle phase 계산
20. Driver 중복 제거 강화
21. MAX_DRIVERS는 상한일 뿐 3개를 강제로 만들지 않음
22. 낮은 cycle stability Driver 제외
23. 낮은 현재 similarity Driver 제외
24. +5 / +10 / +20 / -5 ordered outcome 유지
25. look-ahead bias 방지

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

# 결과 재현성 최소 기준.
# cycle이 반복돼도 결과가 반복되지 않으면 강한 Driver로 보지 않는다.
MIN_RESULT_REPRODUCIBILITY = float(
    os.getenv(
        "DRIVER_MIN_RESULT_REPRO",
        "0.45",
    )
)

# 최근 결과 재현성이 이 값보다 낮으면 품질을 낮춘다.
MIN_RECENT_REPRODUCIBILITY = float(
    os.getenv(
        "DRIVER_MIN_RECENT_REPRO",
        "0.40",
    )
)

# 결과 재현성 계산에서 최소 resolved 표본.
MIN_RESOLVED_RESULTS = int(
    os.getenv(
        "DRIVER_MIN_RESOLVED_RESULTS",
        "2",
    )
)

# 너무 가까운 이벤트가 하나의 패턴을 과도하게 대표하는 것을 방지.
RESULT_LOOKBACK = int(
    os.getenv(
        "DRIVER_RESULT_LOOKBACK",
        "5",
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


# ============================================================
# DATA DOWNLOAD
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
            return np.zeros_like(array)

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

    target5 = entry * 1.05
    target10 = entry * 1.10
    target20 = entry * 1.20
    stop5 = entry * 0.95

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
        first_event = "NONE"
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

    future_end = (
        end_idx
        + 1
        + FORWARD
    )

    if future_end > len(df):
        return {}

    future = df.iloc[
        end_idx + 1:
        future_end
    ]

    if len(future) < FORWARD:
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

    window = df.iloc[
        start_idx:
        end_idx + 1
    ]

    if len(window) < WINDOW:
        return None

    future = future_summary(
        df,
        end_idx,
    )

    if not future:
        return None

    signature = make_signature(
        window
    )

    pattern = choose_pattern_label(
        window
    )

    date = df.index[end_idx]

    close = safe_float(
        df["Close"].iloc[end_idx]
    )

    if (
        not np.isfinite(close)
        or close <= 0
    ):
        return None

    return {
        "date": date,
        "end_idx": int(end_idx),
        "price": close,
        "pattern": pattern,
        "signature": signature,
        "forward_return":
            future["forward_return"],
        "max_up":
            future["max_up"],
        "max_down":
            future["max_down"],
        "ordered":
            future["ordered"],
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
                similarity >= CLUSTER_SIM
                and similarity > best_similarity
            ):
                best_cluster = cluster
                best_similarity = similarity

        if best_cluster is None:
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
            best_cluster[
                "events"
            ].append(event)

            signatures = [
                item["signature"]
                for item
                in best_cluster["events"]
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

            best_cluster[
                "centroid"
            ] = centroid

    return clusters


# ============================================================
# CYCLE ENGINE
# ============================================================

def adjacent_gaps(events):
    if len(events) < 2:
        return []

    ordered = sorted(
        events,
        key=lambda event:
        pd.Timestamp(
            event["date"]
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
                    "gap":
                        float(gap),
                    "from_date":
                        previous["date"],
                    "to_date":
                        current["date"],
                }
            )

    return result


def pairwise_gaps(events):
    if len(events) < 2:
        return []

    ordered = sorted(
        events,
        key=lambda event:
        pd.Timestamp(
            event["date"]
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
        return (
            None,
            np.inf,
        )

    ratio = gap / base

    multiple = int(
        round(ratio)
    )

    multiple = max(
        1,
        min(
            MAX_CYCLE_MULTIPLE,
            multiple,
        ),
    )

    normalized = (
        gap / multiple
    )

    error = abs(
        normalized - base
    ) / max(
        base,
        1.0,
    )

    return (
        multiple,
        error,
    )


def fits_harmonic(
    gap,
    base,
    tolerance=HARMONIC_TOLERANCE,
):
    multiple, error = nearest_harmonic(
        gap,
        base,
    )

    if multiple is None:
        return False

    return (
        error <= tolerance
    )


def candidate_cycle_bases(gaps):
    clean = sorted(
        {
            float(round(gap))
            for gap in gaps
            if (
                np.isfinite(gap)
                and gap >= CYCLE_MIN_DAYS
            )
        }
    )

    if not clean:
        return []

    candidates = set()

    for gap in clean:
        for multiple in range(
            1,
            MAX_CYCLE_MULTIPLE + 1,
        ):
            base = (
                gap / multiple
            )

            if base >= CYCLE_MIN_DAYS:
                candidates.add(
                    round(
                        base,
                        1,
                    )
                )

    return sorted(
        candidates
    )


def cycle_candidate_score(
    base,
    adjacent,
    pairwise,
):
    adjacent_values = [
        item["gap"]
        for item in adjacent
    ]

    matched_adjacent = []
    normalized = []

    for gap in adjacent_values:
        multiple, error = nearest_harmonic(
            gap,
            base,
        )

        if (
            multiple is not None
            and error <= HARMONIC_TOLERANCE
        ):
            matched_adjacent.append(
                gap
            )

            normalized.append(
                gap / multiple
            )

    matched_pairwise = []

    for gap in pairwise:
        if fits_harmonic(
            gap,
            base,
        ):
            matched_pairwise.append(
                gap
            )

    adjacent_support = len(
        matched_adjacent
    )

    pairwise_support = len(
        matched_pairwise
    )

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

        mean_error = float(
            np.mean(errors)
        )

    else:
        center = np.nan
        mean_error = 999.0

    score = (
        adjacent_support * 100.0
        + pairwise_support * 0.50
        - mean_error * 20.0
    )

    return {
        "base":
            float(base),

        "score":
            float(score),

        "adjacent_support":
            adjacent_support,

        "pairwise_support":
            pairwise_support,

        "matched_adjacent":
            matched_adjacent,

        "matched_pairwise":
            matched_pairwise,

        "normalized":
            normalized,

        "mean_error":
            mean_error,

        "center":
            center,
    }


def select_cycle_base(events):
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
            "stability": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "adjacent": adjacent,
            "pairwise": pairwise,
            "matched": [],
        }

    candidates = candidate_cycle_bases(
        [
            item["gap"]
            for item in adjacent
        ]
    )

    if not candidates:
        return {
            "valid": False,
            "reason":
                "NO_CYCLE_BASE",
            "base": np.nan,
            "observations": 0,
            "stability": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "adjacent": adjacent,
            "pairwise": pairwise,
            "matched": [],
        }

    scored = []

    for base in candidates:
        scored.append(
            cycle_candidate_score(
                base,
                adjacent,
                pairwise,
            )
        )

    scored.sort(
        key=lambda item: (
            -item[
                "adjacent_support"
            ],
            item[
                "mean_error"
            ],
            -item[
                "pairwise_support"
            ],
            item[
                "base"
            ],
        )
    )

    best = scored[0]

    matched = best[
        "normalized"
    ]

    observations = len(
        matched
    )

    if observations >= 2:
        median_cycle = float(
            np.median(
                matched
            )
        )

        deviations = [
            abs(
                value
                - median_cycle
            )
            / max(
                median_cycle,
                1.0,
            )
            for value
            in matched
        ]

        stability = max(
            0.0,
            1.0
            - float(
                np.mean(
                    deviations
                )
            ),
        )

    elif observations == 1:
        median_cycle = float(
            matched[0]
        )

        stability = 1.0

    else:
        median_cycle = np.nan
        stability = 0.0

    valid = (
        observations
        >= MIN_CYCLE_OBSERVATIONS
    )

    return {
        "valid":
            bool(valid),

        "reason": (
            "VALID"
            if valid
            else
            "INSUFFICIENT_RECURRENCE"
        ),

        "base": (
            float(
                median_cycle
            )
            if np.isfinite(
                median_cycle
            )
            else np.nan
        ),

        "observations":
            int(observations),

        "stability":
            float(stability),

        "mean":
            safe_mean(
                matched
            ),

        "std":
            safe_std(
                matched
            ),

        "adjacent":
            adjacent,

        "pairwise":
            pairwise,

        "matched":
            matched,

        "matched_adjacent":
            best[
                "matched_adjacent"
            ],

        "mean_error":
            best[
                "mean_error"
            ],
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

    observations = int(
        cycle_result.get(
            "observations",
            0,
        )
    )

    stability = safe_float(
        cycle_result.get(
            "stability"
        )
    )

    if (
        not np.isfinite(base)
        or base <= 0
        or observations
        < MIN_CYCLE_OBSERVATIONS
        or (
            np.isfinite(stability)
            and stability
            < MIN_CYCLE_STABILITY
        )
    ):
        return []

    ordered = sorted(
        events,
        key=lambda event:
        pd.Timestamp(
            event["date"]
        ),
    )

    gaps = adjacent_gaps(
        ordered
    )

    if (
        len(gaps)
        < MIN_CYCLE_OBSERVATIONS
    ):
        return []

    matched_count = 0

    for item in gaps:
        if fits_harmonic(
            item["gap"],
            base,
        ):
            matched_count += 1

    if (
        matched_count
        < MIN_CYCLE_OBSERVATIONS
    ):
        return []

    return [
        ordered
    ]


# ============================================================
# ORDERED STATS
# ============================================================

def calculate_ordered_stats(events):
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

        value = safe_float(
            ordered[
                "plus5_first"
            ]
        )

        if np.isfinite(value):
            plus5.append(value)

            if value == 1.0:
                first_plus5 += 1

        value = safe_float(
            ordered[
                "plus10_first"
            ]
        )

        if np.isfinite(value):
            plus10.append(value)

            if value == 1.0:
                first_plus10 += 1

        value = safe_float(
            ordered[
                "plus20_first"
            ]
        )

        if np.isfinite(value):
            plus20.append(value)

            if value == 1.0:
                first_plus20 += 1

        value = safe_float(
            ordered[
                "minus5_first"
            ]
        )

        if np.isfinite(value):
            minus5.append(value)

            if value == 1.0:
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

        "success_rate": (
            success_count / resolved
            if resolved > 0
            else np.nan
        ),
    }


# ============================================================
# RESULT REPRODUCIBILITY
# ============================================================

def result_reproducibility(events):
    """
    cycle이 반복되는 것과 실제 결과가 반복되는 것은 다르다.

    이 함수는 Driver가 발생했을 때
    실제로 +5% 방향 결과가 반복됐는지를 계산한다.

    resolved 결과만 별도 계산하며,
    AMBIGUOUS / NO_HIT은 성공으로 취급하지 않는다.
    """

    if not events:
        return {
            "result_reproducibility":
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

    resolved_events = []
    ambiguous = 0
    no_hit = 0

    for event in events:
        first_event = event[
            "ordered"
        ]["first_event"]

        if first_event == "PLUS_5":
            resolved_events.append(
                1.0
            )

        elif first_event == "MINUS_5":
            resolved_events.append(
                0.0
            )

        elif first_event == "AMBIGUOUS":
            ambiguous += 1

        else:
            no_hit += 1

    total = len(events)

    resolved_n = len(
        resolved_events
    )

    positive_n = int(
        sum(
            value == 1.0
            for value
            in resolved_events
        )
    )

    negative_n = (
        resolved_n
        - positive_n
    )

    if resolved_n > 0:
        raw_repro = (
            positive_n
            / resolved_n
        )

        resolved_rate = (
            resolved_n
            / total
        )

    else:
        raw_repro = np.nan
        resolved_rate = 0.0

    ambiguous_rate = (
        ambiguous / total
        if total
        else np.nan
    )

    no_hit_rate = (
        no_hit / total
        if total
        else np.nan
    )

    # --------------------------------------------------------
    # 표본 보정
    #
    # 3/3 = 100%라고 바로 강한 Driver로 판단하지 않는다.
    #
    # resolved 표본이 증가할수록 보정값이 원래 값에 접근.
    # --------------------------------------------------------

    if resolved_n > 0:
        confidence = (
            resolved_n
            / (
                resolved_n
                + 2.0
            )
        )

        baseline = 0.50

        adjusted = (
            raw_repro * confidence
            + baseline
            * (1.0 - confidence)
        )

    else:
        adjusted = np.nan

    return {
        "result_reproducibility":
            adjusted,

        "raw_result_reproducibility":
            raw_repro,

        "resolved_rate":
            resolved_rate,

        "ambiguous_rate":
            ambiguous_rate,

        "no_hit_rate":
            no_hit_rate,

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

def recent_statistics(events):
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
        key=lambda event:
        pd.Timestamp(
            event["date"]
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
# CYCLE TIMING
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
        key=lambda event:
        pd.Timestamp(
            event["date"]
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

    expected_next = (
        last_date
        + pd.Timedelta(
            days=float(cycle)
        )
    )

    phase = (
        days_since
        / cycle
    )

    # phase는 표시용으로 0~200% 정도까지 허용.
    phase_pct = (
        phase * 100.0
    )

    return {
        "days_since_last_occurrence":
            days_since,

        "expected_next_cycle":
            date_string(
                expected_next
            ),

        "cycle_phase_pct":
            phase_pct,
    }


# ============================================================
# TARGETS
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
            event["max_up"]
            for event in events
        ]
    )

    downs = finite_values(
        [
            event["max_down"]
            for event in events
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

        upper_up = float(
            ups.quantile(0.75)
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
    result_repro,
    resolved_n,
    recent_repro,
):
    if (
        occurrences
        < MIN_DRIVER_OCCURRENCES
    ):
        return "INSUFFICIENT_OCCURRENCES"

    if (
        cycle_observations
        < MIN_CYCLE_OBSERVATIONS
    ):
        return "LOW_CYCLE_REPEAT"

    if (
        np.isfinite(
            safe_float(
                cycle_stability
            )
        )
        and cycle_stability
        < MIN_CYCLE_STABILITY
    ):
        return "LOW_CYCLE_STABILITY"

    # 결과 표본이 아직 너무 적으면
    # cycle 자체는 반복돼도 결과 Driver로는
    # REPEAT_DETECTED까지만 허용.
    if (
        resolved_n
        < MIN_RESOLVED_RESULTS
    ):
        return "LOW_RESULT_SAMPLE"

    if (
        np.isfinite(
            safe_float(
                result_repro
            )
        )
        and result_repro
        < MIN_RESULT_REPRODUCIBILITY
    ):
        return "LOW_RESULT_REPRO"

    if (
        np.isfinite(
            safe_float(
                recent_repro
            )
        )
        and recent_repro
        < MIN_RECENT_REPRODUCIBILITY
        and occurrences >= VALIDATED_OCCURRENCES
    ):
        return "RECENT_RESULT_WEAK"

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

    weak_validation = {
        "INSUFFICIENT_OCCURRENCES",
        "LOW_CYCLE_REPEAT",
        "LOW_CYCLE_STABILITY",
        "LOW_RESULT_SAMPLE",
        "LOW_RESULT_REPRO",
    }

    if validation in weak_validation:
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
# DRIVER QUALITY
# ============================================================

def driver_quality(
    occurrences,
    cycle_stability,
    result_repro,
    recent_repro,
    current_similarity,
):
    """
    0~100 품질.

    중요:
    이것은 방향성 추천 점수가 아니다.
    Driver 자체가 얼마나 재현 가능한 구조인지 보는 품질값이다.
    """

    sim = clamp01(
        current_similarity
    )

    cycle = clamp01(
        cycle_stability
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

    quality = (
        sim * 0.20
        + cycle * 0.20
        + repro * 0.30
        + recent * 0.20
        + occurrence * 0.10
    )

    return float(
        quality
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
    result_repro,
    recent_repro,
):
    sim = clamp01(
        current_similarity
    )

    occurrence_score = clamp01(
        min(
            occurrences / 8.0,
            1.0,
        )
    )

    plus5 = clamp01(
        plus5_first_rate
    )

    cycle = clamp01(
        cycle_stability
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

    # v6.2.1보다 결과 재현성을 더 중요하게 둔다.
    score = (
        sim * 0.20
        + occurrence_score * 0.10
        + plus5 * 0.15
        + cycle * 0.15
        + recent * 0.10
        + repro * 0.20
        + recent_repro_value * 0.10
    )

    return float(
        score
    )


# ============================================================
# DUPLICATE DRIVER
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
            candidate[
                "centroid"
            ],
            existing[
                "centroid"
            ],
        )

        if similarity < DUPLICATE_SIM:
            continue

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

        # 동일 cycle 또는 harmonic cycle이면
        # 같은 구조로 간주.
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
            ):
                return True

            if cycle_ratio_is_harmonic(
                c1,
                c2,
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
):
    raw = (
        f"{symbol}|"
        f"{base_cluster_id}|"
        f"{cycle_type}"
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

    pattern_clusters = cluster_events(
        events
    )

    candidates = []

    current_price = safe_float(
        df["Close"].iloc[-1]
    )

    as_of = df.index[-1]

    # ========================================================
    # BUILD CANDIDATES
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

        if not cycle_result.get(
            "valid",
            False,
        ):
            continue

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

        current_signature = make_signature(
            df.iloc[-WINDOW:]
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

        if np.isfinite(
            recent_repro
        ):
            recent_repro_rate = (
                recent_repro / 100.0
            )

        else:
            recent_repro_rate = np.nan

        validation = validation_status(
            len(event_group),
            cycle_observations,
            cycle_stability,
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
            (
                safe_float(
                    recent.get(
                        "recent_plus5_first_pct",
                        np.nan,
                    )
                ) / 100.0
                if np.isfinite(
                    safe_float(
                        recent.get(
                            "recent_plus5_first_pct",
                            np.nan,
                        )
                    )
                )
                else np.nan
            ),
            result_repro,
            recent_repro_rate,
        )

        state = driver_state(
            current_similarity,
            validation,
        )

        quality = driver_quality(
            len(event_group),
            cycle_stability,
            result_repro,
            recent_repro_rate,
            current_similarity,
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

        base_cluster_id = (
            make_base_cluster_id(
                symbol,
                cluster["pattern"],
                centroid,
            )
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
                item["current_similarity"],
                0,
            ),

            -int(
                item["occurrences"]
            ),

            -safe_float(
                item["result_repro"],
                0,
            ),

            -safe_float(
                item["cycle_stability"],
                0,
            ),
        )
    )

    # ========================================================
    # SELECT DISTINCT DRIVERS
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

    # ========================================================
    # BUILD OUTPUT
    # ========================================================

    profiles = []
    current_rows = []
    history_rows = []

    current_pattern = choose_pattern_label(
        df.iloc[-WINDOW:]
    )

    phase = (
        "UPTREND"
        if (
            safe_float(
                df["ma20"].iloc[-1]
            )
            > safe_float(
                df["ma50"].iloc[-1]
            )
            > safe_float(
                df["ma100"].iloc[-1]
            )
        )
        else
        "DOWNTREND"
        if (
            safe_float(
                df["ma20"].iloc[-1]
            )
            < safe_float(
                df["ma50"].iloc[-1]
            )
            < safe_float(
                df["ma100"].iloc[-1]
            )
        )
        else
        "TRANSITION"
    )

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
                item[
                    "pattern"
                ],

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

            "cycle_mean_days":
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

            "current_price":
                rnd(
                    current_price,
                    4,
                ),

            "current_pattern":
                current_pattern,

            "current_similarity_pct":
                pct(
                    item[
                        "current_similarity"
                    ]
                ),

            "driver_score_pct":
                pct(
                    item[
                        "score"
                    ]
                ),

            "driver_quality_pct":
                pct(
                    item[
                        "quality"
                    ]
                ),

            "state":
                item[
                    "state"
                ],

            "validation_status":
                item[
                    "validation"
                ],

            "phase":
                phase,

            "occurrences":
                item[
                    "occurrences"
                ],

            # --------------------------------------------
            # 전체 결과 재현성
            # --------------------------------------------

            "result_reproducibility_pct":
                pct(
                    item[
                        "result_repro"
                    ]
                ),

            "raw_result_reproducibility_pct":
                pct(
                    repro.get(
                        "raw_result_reproducibility",
                        np.nan,
                    )
                ),

            "resolved_rate_pct":
                pct(
                    repro.get(
                        "resolved_rate",
                        np.nan,
                    )
                ),

            "ambiguous_rate_pct":
                pct(
                    repro.get(
                        "ambiguous_rate",
                        np.nan,
                    )
                ),

            "no_hit_rate_pct":
                pct(
                    repro.get(
                        "no_hit_rate",
                        np.nan,
                    )
                ),

            "resolved_n":
                repro.get(
                    "resolved_n",
                    0,
                ),

            "positive_n":
                repro.get(
                    "positive_n",
                    0,
                ),

            "negative_n":
                repro.get(
                    "negative_n",
                    0,
                ),

            # --------------------------------------------
            # +5 / +10 / +20 / -5
            # --------------------------------------------

            "success_rate_pct":
                pct(
                    ordered[
                        "success_rate"
                    ]
                ),

            "plus5_first_count":
                ordered[
                    "plus5_first_count"
                ],

            "plus5_first_rate_pct":
                pct(
                    ordered[
                        "plus5_first_rate"
                    ]
                ),

            "plus5_valid_n":
                ordered[
                    "plus5_valid_n"
                ],

            "plus10_first_count":
                ordered[
                    "plus10_first_count"
                ],

            "plus10_first_rate_pct":
                pct(
                    ordered[
                        "plus10_first_rate"
                    ]
                ),

            "plus10_valid_n":
                ordered[
                    "plus10_valid_n"
                ],

            "plus20_first_count":
                ordered[
                    "plus20_first_count"
                ],

            "plus20_first_rate_pct":
                pct(
                    ordered[
                        "plus20_first_rate"
                    ]
                ),

            "plus20_valid_n":
                ordered[
                    "plus20_valid_n"
                ],

            "minus5_first_count":
                ordered[
                    "minus5_first_count"
                ],

            "minus5_first_rate_pct":
                pct(
                    ordered[
                        "minus5_first_rate"
                    ]
                ),

            "minus5_valid_n":
                ordered[
                    "minus5_valid_n"
                ],

            "ambiguous_count":
                ordered[
                    "ambiguous_count"
                ],

            "no_hit_count":
                ordered[
                    "no_hit_count"
                ],

            # --------------------------------------------
            # 최근 재현
            # --------------------------------------------

            "recent_occurrences":
                recent[
                    "recent_occurrences"
                ],

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

            "recent_resolved_rate_pct":
                recent[
                    "recent_resolved_rate_pct"
                ],

            "recent_ambiguous_rate_pct":
                recent[
                    "recent_ambiguous_rate_pct"
                ],

            "recent_no_hit_rate_pct":
                recent[
                    "recent_no_hit_rate_pct"
                ],

            # --------------------------------------------
            # cycle timing
            # --------------------------------------------

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

            # --------------------------------------------
            # future statistics
            # --------------------------------------------

            "avg_forward_return_pct":
                rnd(
                    safe_mean(
                        [
                            event[
                                "forward_return"
                            ]
                            for event
                            in item["events"]
                        ]
                    ) * 100,
                    2,
                ),

            "median_forward_return_pct":
                rnd(
                    safe_median(
                        [
                            event[
                                "forward_return"
                            ]
                            for event
                            in item["events"]
                        ]
                    ) * 100,
                    2,
                ),

            "median_max_up_pct":
                rnd(
                    safe_median(
                        [
                            event[
                                "max_up"
                            ]
                            for event
                            in item["events"]
                        ]
                    ) * 100,
                    2,
                ),

            "median_max_down_pct":
                rnd(
                    safe_median(
                        [
                            event[
                                "max_down"
                            ]
                            for event
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

            "first_seen":
                min(
                    date_string(
                        event["date"]
                    )
                    for event
                    in item["events"]
                ),

            "last_seen":
                max(
                    date_string(
                        event["date"]
                    )
                    for event
                    in item["events"]
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
        profile
        for profile in profiles
        if profile[
            "state"
        ] in {
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
            f"result_repro={safe_float(profile['result_reproducibility_pct']):.1f}% | "
            f"recent_repro={safe_float(profile['recent_result_reproducibility_pct']):.1f}% | "
            f"cycle={safe_float(profile['cycle_mode_days']):.1f}d | "
            f"cycle_stab={safe_float(profile['cycle_stability_pct']):.1f}% | "
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
        " UNIVERSAL DRIVER ENGINE v6.3"
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

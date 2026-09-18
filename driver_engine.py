
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNIVERSAL DRIVER ENGINE v6
==========================

목적
----
종목별 과거 OHLCV에서 반복되는 "운전수(Driver)"를 탐색한다.

v6 핵심
-------
1. 선후관계 적중률
   +5% 먼저 / -5% 먼저
   +10% 먼저 / -5% 먼저
   +20% 먼저 / -5% 먼저

2. 같은 봉에서 target/stop이 동시에 발생하면
   순서를 알 수 없으므로 AMBIGUOUS 처리.

3. Driver cycle 분리
   짧은 cycle과 긴 cycle을 분리한다.

4. cycle은 단순 median이 아니라
   recurrence gap의 mode를 사용한다.

5. 1-cycle / 2-cycle 배수 관계를 고려하여
   동일한 기본 반복주기의 중복을 줄인다.

6. cluster_id와 driver_rank를 분리한다.
   DRIVER_1/2/3 순위가 바뀌어도
   cluster_id로 동일 Driver를 추적할 수 있다.

7. 최소 반복 횟수 검증
   1회      -> PROVISIONAL
   2~3회    -> REPEAT_DETECTED
   4~5회    -> VALIDATED
   6회 이상 -> STABLE

8. 최근 재현성
   최근 occurrence들의 선후관계를 별도로 계산한다.

9. cycle 안정성
   실제 cycle에 해당하는 gap만 이용하여 계산한다.

10. 기존 출력 파일 유지
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


# 동일한 움직임을 여러 사건으로 세지 않기 위한 최소 간격
EVENT_SPACING = int(
    os.getenv(
        "DRIVER_EVENT_SPACING",
        "40",
    )
)


# Cycle 최소 기간
CYCLE_MIN_DAYS = int(
    os.getenv(
        "DRIVER_CYCLE_MIN_DAYS",
        "25",
    )
)


# 단기/장기 분리를 위한 최소 배율
CYCLE_SPLIT_RATIO = float(
    os.getenv(
        "DRIVER_CYCLE_SPLIT_RATIO",
        "2.50",
    )
)


# cycle mode 주변 허용 오차
CYCLE_TOLERANCE = float(
    os.getenv(
        "DRIVER_CYCLE_TOLERANCE",
        "0.30",
    )
)


# 최소 Driver occurrence
MIN_DRIVER_OCCURRENCES = int(
    os.getenv(
        "DRIVER_MIN_OCCURRENCES",
        "2",
    )
)


# VALIDATED 최소 occurrence
VALIDATED_OCCURRENCES = int(
    os.getenv(
        "DRIVER_VALIDATED_OCCURRENCES",
        "4",
    )
)


# STABLE 최소 occurrence
STABLE_OCCURRENCES = int(
    os.getenv(
        "DRIVER_STABLE_OCCURRENCES",
        "6",
    )
)


# 최근 occurrence 계산 개수
RECENT_OCCURRENCES = int(
    os.getenv(
        "DRIVER_RECENT_OCCURRENCES",
        "5",
    )
)


# cycle 최소 안정성
MIN_CYCLE_STABILITY = float(
    os.getenv(
        "DRIVER_MIN_CYCLE_STABILITY",
        "0.45",
    )
)


# 같은 cycle의 1x / 2x / 3x 관계 허용
HARMONIC_TOLERANCE = float(
    os.getenv(
        "DRIVER_HARMONIC_TOLERANCE",
        "0.25",
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

        return round(
            value,
            digits,
        )

    except Exception:
        return np.nan


def pct(value, digits=2):

    try:

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return round(
            value * 100.0,
            digits,
        )

    except Exception:
        return np.nan


def finite_values(values):

    result = []

    for value in values:

        try:

            value = float(value)

            if np.isfinite(value):
                result.append(value)

        except Exception:
            pass

    return pd.Series(
        result,
        dtype=float,
    )


def safe_float(value, default=np.nan):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def safe_rate(values):

    clean = []

    for value in values:

        value = safe_float(
            value
        )

        if np.isfinite(value):
            clean.append(value)

    if not clean:
        return np.nan

    return float(
        np.mean(clean)
    )


# ============================================================
# DATA
# ============================================================

def clean_download(df):

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(
        out.columns,
        pd.MultiIndex,
    ):

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

        name = str(
            column
        ).strip().lower()

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

        return clean_download(
            data
        )

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

    x["ret1"] = (
        close.pct_change()
    )

    x["ma20"] = (
        close.rolling(20).mean()
    )

    x["ma50"] = (
        close.rolling(50).mean()
    )

    x["ma100"] = (
        close.rolling(100).mean()
    )

    x["ma200"] = (
        close.rolling(200).mean()
    )

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (
                high
                - previous_close
            ).abs(),
            (
                low
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr14"] = (
        tr.rolling(14).mean()
    )

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
# ROBUST NORMALIZATION
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
        return np.zeros_like(
            array
        )

    median = np.median(
        finite
    )

    mad = np.median(
        np.abs(
            finite - median
        )
    )

    if (
        not np.isfinite(mad)
        or mad < 1e-9
    ):

        std = np.std(
            finite
        )

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
        filled
        - median
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
        close / base
        - 1
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
        window[
            "vol_ratio"
        ]
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
# HUMAN PATTERN
# ============================================================

def choose_pattern_label(window):

    close = window[
        "Close"
    ]

    start = float(
        close.iloc[0]
    )

    end = float(
        close.iloc[-1]
    )

    if start <= 0:
        return "구조불명"

    total = (
        end / start
        - 1
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
        peak / start
        - 1
    )

    trough_return = (
        trough / start
        - 1
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
        window[
            "vol20"
        ].iloc[-1]
    )

    median_vol = safe_float(
        window[
            "vol20"
        ].median()
    )

    if (
        trough_return <= -0.18
        and trough_pos
        < len(close) * 0.80
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
        and current_vol
        > median_vol * 1.15
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

    # --------------------------------------------------------
    # 첫 도달일 탐색
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 독립적인 선후관계
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 전체 최초 이벤트
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 전체 outcome 상태
    # --------------------------------------------------------

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

    # 중요:
    # ordered_outcome은 여기에서 단 한 번만 호출한다.
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

        **ordered,
    }


# ============================================================
# EVENT EXTRACTION
# ============================================================

def extract_events(df):

    events = []

    first_end = max(
        WINDOW,
        MIN_HISTORY,
    )

    last_end = (
        len(df) - 1 - FORWARD
    )

    if last_end <= first_end:
        return events

    for end_idx in range(
        first_end,
        last_end + 1,
    ):

        window = df.iloc[
            end_idx - WINDOW + 1:
            end_idx + 1
        ].copy()

        if len(window) < WINDOW:
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

        outcome = future_summary(
            df,
            end_idx,
        )

        if not outcome:
            continue

        event = {

            "end_idx":
                end_idx,

            "date":
                df.index[end_idx],

            "signature":
                signature,

            "label":
                choose_pattern_label(
                    window
                ),

            **outcome,
        }

        events.append(
            event
        )

    # --------------------------------------------------------
    # 중복 사건 제거
    # --------------------------------------------------------

    selected = []

    last_selected_idx = -10**9

    for event in events:

        if (
            event["end_idx"]
            - last_selected_idx
            >= EVENT_SPACING
        ):

            selected.append(
                event
            )

            last_selected_idx = (
                event["end_idx"]
            )

    return selected


# ============================================================
# SIGNATURE CLUSTERING
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
                cluster[
                    "centroid"
                ],
            )

            if (
                similarity
                > best_similarity
            ):

                best_similarity = (
                    similarity
                )

                best_idx = idx

        if (
            best_idx is not None
            and best_similarity
            >= threshold
        ):

            cluster = clusters[
                best_idx
            ]

            cluster[
                "events"
            ].append(event)

            centroid = np.mean(
                [
                    item[
                        "signature"
                    ]
                    for item
                    in cluster[
                        "events"
                    ]
                ],
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

            cluster[
                "centroid"
            ] = centroid

        else:

            clusters.append(
                {
                    "centroid":
                        signature.copy(),

                    "events":
                        [event],
                }
            )

    return clusters


# ============================================================
# CYCLE PAIRS
# ============================================================

def cycle_pairs(events):

    ordered = sorted(
        events,
        key=lambda x:
        pd.Timestamp(
            x["date"]
        ),
    )

    pairs = []

    for i in range(
        len(ordered)
    ):

        date_i = pd.Timestamp(
            ordered[i]["date"]
        )

        for j in range(
            i + 1,
            len(ordered)
        ):

            date_j = pd.Timestamp(
                ordered[j]["date"]
            )

            gap = (
                date_j
                - date_i
            ).days

            if gap < CYCLE_MIN_DAYS:
                continue

            pairs.append(
                {
                    "left":
                        ordered[i],

                    "right":
                        ordered[j],

                    "gap":
                        gap,
                }
            )

    return pairs


# ============================================================
# 1D LOG KMEANS
# ============================================================

def one_dimensional_kmeans(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
        & (
            values
            >= CYCLE_MIN_DAYS
        )
    ]

    if len(values) < 4:
        return None

    logs = np.log(
        values
    )

    c1 = float(
        np.min(logs)
    )

    c2 = float(
        np.max(logs)
    )

    if abs(c2 - c1) < 0.10:
        return None

    for _ in range(50):

        d1 = np.abs(
            logs - c1
        )

        d2 = np.abs(
            logs - c2
        )

        group1 = logs[
            d1 <= d2
        ]

        group2 = logs[
            d2 < d1
        ]

        if (
            len(group1) == 0
            or len(group2) == 0
        ):
            return None

        new_c1 = float(
            np.mean(group1)
        )

        new_c2 = float(
            np.mean(group2)
        )

        if (
            abs(new_c1 - c1)
            < 1e-6
            and
            abs(new_c2 - c2)
            < 1e-6
        ):
            break

        c1 = new_c1
        c2 = new_c2

    centers = sorted(
        [
            np.exp(c1),
            np.exp(c2),
        ]
    )

    return centers


# ============================================================
# CYCLE TYPE
# ============================================================

def cycle_type_from_days(days):

    days = safe_float(
        days
    )

    if not np.isfinite(days):
        return "단기"

    return (
        "단기"
        if days < 180
        else "장기"
    )


# ============================================================
# HARMONIC CHECK
# ============================================================

def harmonic_distance(
    gap,
    base,
):

    gap = safe_float(
        gap
    )

    base = safe_float(
        base
    )

    if (
        not np.isfinite(gap)
        or not np.isfinite(base)
        or gap <= 0
        or base <= 0
    ):
        return np.inf

    best = np.inf

    for multiplier in [
        1,
        2,
        3,
    ]:

        expected = (
            base * multiplier
        )

        distance = abs(
            gap - expected
        ) / expected

        best = min(
            best,
            distance,
        )

    return best


# ============================================================
# CYCLE GROUP BUILDER
# ============================================================

def build_cycle_group(
    events,
    mode,
    cycle_type,
):

    mode = safe_float(
        mode
    )

    if (
        not np.isfinite(mode)
        or mode <= 0
    ):
        return None

    tolerance = max(
        10.0,
        mode * CYCLE_TOLERANCE,
    )

    pairs = cycle_pairs(
        events
    )

    matched_pairs = []

    for pair in pairs:

        gap = pair[
            "gap"
        ]

        # 1x / 2x / 3x cycle 관계까지 허용
        ratio_distance = (
            harmonic_distance(
                gap,
                mode,
            )
        )

        direct_distance = (
            abs(
                gap - mode
            )
            / mode
        )

        if (
            direct_distance
            <= CYCLE_TOLERANCE
            or ratio_distance
            <= HARMONIC_TOLERANCE
        ):

            matched_pairs.append(
                pair
            )

    if not matched_pairs:
        return None

    event_ids = set()

    for pair in matched_pairs:

        event_ids.add(
            id(
                pair["left"]
            )
        )

        event_ids.add(
            id(
                pair["right"]
            )
        )

    group_events = [
        event
        for event in events
        if id(event)
        in event_ids
    ]

    if len(group_events) < 2:
        return None

    direct_gaps = []

    for pair in matched_pairs:

        gap = pair[
            "gap"
        ]

        # 표시 cycle은 기본 mode 주변의
        # direct 1x gap을 우선 사용
        if (
            abs(
                gap - mode
            )
            / mode
            <= CYCLE_TOLERANCE
        ):

            direct_gaps.append(
                gap
            )

    if direct_gaps:

        actual_mode = float(
            np.median(
                direct_gaps
            )
        )

    else:

        actual_mode = float(
            mode
        )

    return {

        "events":
            group_events,

        "cycle_type":
            cycle_type,

        "cycle_mode":
            actual_mode,

        "cycle_observations":
            len(
                matched_pairs
            ),

        "cycle_pairs":
            matched_pairs,
    }


# ============================================================
# CYCLE GROUPS
# ============================================================

def cycle_groups(events):

    """
    동일 signature cluster 안에서
    서로 다른 recurrence cycle을 분리한다.

    예:

        58, 61, 57
        581, 590, 575

    → 단기 ~59
    → 장기 ~582

    추가로:

        178
        356

    같은 1x/2x 관계를
    같은 기본 cycle로 인식할 수 있도록 한다.
    """

    if len(events) < 2:

        return [
            {
                "events":
                    list(events),

                "cycle_type":
                    "단기",

                "cycle_mode":
                    np.nan,

                "cycle_observations":
                    0,

                "cycle_pairs":
                    [],
            }
        ]

    pairs = cycle_pairs(
        events
    )

    if not pairs:

        return [
            {
                "events":
                    list(events),

                "cycle_type":
                    "단기",

                "cycle_mode":
                    np.nan,

                "cycle_observations":
                    0,

                "cycle_pairs":
                    [],
            }
        ]

    gaps = np.array(
        [
            pair["gap"]
            for pair in pairs
        ],
        dtype=float,
    )

    gaps = gaps[
        np.isfinite(gaps)
    ]

    if len(gaps) == 0:

        return []

    # --------------------------------------------------------
    # 1개 cycle
    # --------------------------------------------------------

    single_mode = float(
        np.median(gaps)
    )

    centers = None

    if (
        len(gaps) >= 4
        and (
            gaps.max()
            / max(
                gaps.min(),
                1,
            )
        )
        >= CYCLE_SPLIT_RATIO
    ):

        centers = one_dimensional_kmeans(
            gaps
        )

    # --------------------------------------------------------
    # 2개 cycle
    # --------------------------------------------------------

    if (
        centers is not None
        and len(centers) == 2
    ):

        short_mode = centers[0]
        long_mode = centers[1]

        groups = []

        short_group = build_cycle_group(
            events,
            short_mode,
            "단기",
        )

        long_group = build_cycle_group(
            events,
            long_mode,
            "장기",
        )

        if short_group is not None:
            groups.append(
                short_group
            )

        if long_group is not None:
            groups.append(
                long_group
            )

        # 두 cycle이 사실상 같은 cycle이면
        # 하나로 합친다.
        if len(groups) == 2:

            a = safe_float(
                groups[0][
                    "cycle_mode"
                ]
            )

            b = safe_float(
                groups[1][
                    "cycle_mode"
                ]
            )

            if (
                np.isfinite(a)
                and np.isfinite(b)
                and max(a, b)
                / max(
                    min(a, b),
                    1,
                )
                < CYCLE_SPLIT_RATIO
            ):

                groups = [
                    {
                        "events":
                            list(events),

                        "cycle_type":
                            cycle_type_from_days(
                                np.median(
                                    gaps
                                )
                            ),

                        "cycle_mode":
                            float(
                                np.median(
                                    gaps
                                )
                            ),

                        "cycle_observations":
                            len(gaps),

                        "cycle_pairs":
                            pairs,
                    }
                ]

        if groups:
            return groups

    # --------------------------------------------------------
    # single cycle fallback
    # --------------------------------------------------------

    cycle_type = (
        cycle_type_from_days(
            single_mode
        )
    )

    group = build_cycle_group(
        events,
        single_mode,
        cycle_type,
    )

    if group is not None:

        return [
            group
        ]

    return [
        {
            "events":
                list(events),

            "cycle_type":
                cycle_type,

            "cycle_mode":
                single_mode,

            "cycle_observations":
                len(gaps),

            "cycle_pairs":
                pairs,
        }
    ]


# ============================================================
# CYCLE STATISTICS
# ============================================================

def cycle_statistics(
    events,
    cycle_mode=np.nan,
):

    if len(events) < 2:

        return (
            np.nan,
            np.nan,
            np.nan,
            0,
            np.nan,
        )

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
            current
            - previous
        ).days

        if gap < CYCLE_MIN_DAYS:
            continue

        if np.isfinite(
            safe_float(
                cycle_mode
            )
        ):

            distance = (
                abs(
                    gap
                    - cycle_mode
                )
                / cycle_mode
            )

            # 1x cycle만 직접 통계에 사용
            if (
                distance
                > CYCLE_TOLERANCE
            ):
                continue

        gaps.append(
            gap
        )

    if not gaps:

        return (
            np.nan,
            np.nan,
            np.nan,
            0,
            np.nan,
        )

    median_gap = float(
        np.median(gaps)
    )

    mean_gap = float(
        np.mean(gaps)
    )

    std_gap = float(
        np.std(gaps)
    )

    # CV 기반 안정성
    if mean_gap > 0:

        cv = (
            std_gap
            / mean_gap
        )

        stability = float(
            np.clip(
                1.0 - cv,
                0.0,
                1.0,
            )
        )

    else:

        stability = np.nan

    return (
        median_gap,
        mean_gap,
        std_gap,
        len(gaps),
        stability,
    )


# ============================================================
# STABILITY
# ============================================================

def cycle_stability_from_pairs(
    cycle_pairs,
    cycle_mode,
):

    if not cycle_pairs:
        return np.nan

    mode = safe_float(
        cycle_mode
    )

    if (
        not np.isfinite(mode)
        or mode <= 0
    ):
        return np.nan

    direct = []

    for pair in cycle_pairs:

        gap = safe_float(
            pair["gap"]
        )

        if not np.isfinite(gap):
            continue

        distance = (
            abs(
                gap - mode
            )
            / mode
        )

        if (
            distance
            <= CYCLE_TOLERANCE
        ):

            direct.append(
                gap
            )

    if len(direct) < 2:
        return np.nan

    median = float(
        np.median(direct)
    )

    deviations = [
        abs(
            value - median
        ) / median
        for value in direct
        if median > 0
    ]

    if not deviations:
        return np.nan

    mean_deviation = float(
        np.mean(
            deviations
        )
    )

    return float(
        np.clip(
            1.0
            - (
                mean_deviation
                / max(
                    CYCLE_TOLERANCE,
                    1e-9,
                )
            ),
            0.0,
            1.0,
        )
    )


# ============================================================
# CLUSTER IDS
# ============================================================

def make_base_cluster_id(
    symbol,
    pattern,
    events,
):

    signatures = [
        event[
            "signature"
        ]
        for event in events
    ]

    if signatures:

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

        quantized = np.round(
            centroid,
            3,
        )

        fingerprint = ",".join(
            f"{value:.3f}"
            for value
            in quantized
        )

    else:

        fingerprint = "EMPTY"

    raw = (
        f"{symbol}|"
        f"{pattern}|"
        f"{fingerprint}"
    )

    digest = hashlib.sha1(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()[:10]

    return (
        f"{symbol}-BASE-{digest}"
    )


def make_cycle_cluster_id(
    symbol,
    base_cluster_id,
    cycle_type,
    cycle_mode,
):

    mode_value = (
        "NA"
        if not np.isfinite(
            safe_float(
                cycle_mode
            )
        )
        else str(
            int(
                round(
                    float(
                        cycle_mode
                    )
                )
            )
        )
    )

    raw = (
        f"{symbol}|"
        f"{base_cluster_id}|"
        f"{cycle_type}|"
        f"{mode_value}"
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


def make_cluster_id(
    symbol,
    pattern,
    cycle_type,
    cycle_mode,
    events,
):

    base = make_base_cluster_id(
        symbol,
        pattern,
        events,
    )

    return make_cycle_cluster_id(
        symbol,
        base,
        cycle_type,
        cycle_mode,
    )


# ============================================================
# ORDERED STATS
# ============================================================

def ordered_stats(
    events,
):

    total = len(events)

    if total == 0:

        return {

            "plus5_first_count":
                0,

            "plus5_first_rate":
                np.nan,

            "plus10_first_count":
                0,

            "plus10_first_rate":
                np.nan,

            "plus20_first_count":
                0,

            "plus20_first_rate":
                np.nan,

            "minus5_first_count":
                0,

            "minus5_first_rate":
                np.nan,

            "plus5_valid_n":
                0,

            "plus10_valid_n":
                0,

            "plus20_valid_n":
                0,

            "minus5_valid_n":
                0,

            "ambiguous_count":
                0,

            "no_hit_count":
                0,
        }


    def field_stats(
        field
    ):

        values = []

        for event in events:

            value = safe_float(
                event.get(
                    field,
                    np.nan,
                )
            )

            if np.isfinite(value):

                values.append(
                    value
                )

        if not values:

            return (
                0,
                np.nan,
                0,
            )

        wins = sum(
            value == 1.0
            for value in values
        )

        return (
            wins,
            float(
                wins
                / len(values)
            ),
            len(values),
        )


    plus5_count, plus5_rate, plus5_n = (
        field_stats(
            "plus5_first"
        )
    )

    plus10_count, plus10_rate, plus10_n = (
        field_stats(
            "plus10_first"
        )
    )

    plus20_count, plus20_rate, plus20_n = (
        field_stats(
            "plus20_first"
        )
    )

    minus5_count, minus5_rate, minus5_n = (
        field_stats(
            "minus5_first"
        )
    )


    ambiguous_count = sum(
        event.get(
            "first_event"
        ) == "AMBIGUOUS"
        for event in events
    )


    no_hit_count = sum(
        event.get(
            "outcome_status"
        ) == "NO_HIT"
        for event in events
    )


    return {

        "plus5_first_count":
            plus5_count,

        "plus5_first_rate":
            plus5_rate,

        "plus10_first_count":
            plus10_count,

        "plus10_first_rate":
            plus10_rate,

        "plus20_first_count":
            plus20_count,

        "plus20_first_rate":
            plus20_rate,

        "minus5_first_count":
            minus5_count,

        "minus5_first_rate":
            minus5_rate,

        "plus5_valid_n":
            plus5_n,

        "plus10_valid_n":
            plus10_n,

        "plus20_valid_n":
            plus20_n,

        "minus5_valid_n":
            minus5_n,

        "ambiguous_count":
            ambiguous_count,

        "no_hit_count":
            no_hit_count,
    }


# ============================================================
# RECENT STATS
# ============================================================

def recent_ordered_stats(
    events,
    count=RECENT_OCCURRENCES,
):

    ordered = sorted(
        events,
        key=lambda x:
        pd.Timestamp(
            x["date"]
        ),
    )

    recent = ordered[
        -count:
    ]

    stats = ordered_stats(
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
# VALIDATION STATUS
# ============================================================

def validation_status(
    occurrences,
    cycle_stability,
    cycle_observations,
):

    if occurrences < 2:

        return "PROVISIONAL"

    if occurrences < VALIDATED_OCCURRENCES:

        return "REPEAT_DETECTED"

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

    if cycle_observations < 2:

        return "INSUFFICIENT_CYCLE"

    if occurrences >= STABLE_OCCURRENCES:

        return "STABLE"

    return "VALIDATED"


# ============================================================
# DRIVER STATE
# ============================================================

def driver_state(
    similarity,
    occurrences,
    history_length,
    validation,
):

    if occurrences < 2:

        if history_length < 300:
            return "PROVISIONAL"

        return "INACTIVE"


    if validation in {
        "LOW_CYCLE_STABILITY",
        "INSUFFICIENT_CYCLE",
    }:

        if (
            similarity
            >= CURRENT_ACTIVE_SIM
        ):

            return "WATCH"

        return "INACTIVE"


    if (
        similarity
        >= CURRENT_ACTIVE_SIM
    ):

        return "ACTIVE"


    if (
        similarity
        >= CURRENT_WATCH_SIM
    ):

        return "WATCH"


    return "INACTIVE"


# ============================================================
# DRIVER COUNT
# ============================================================

def allowed_driver_count(
    history_length,
):

    if history_length < 300:

        return 1

    if history_length < 700:

        return min(
            2,
            MAX_DRIVERS,
        )

    return MAX_DRIVERS


# ============================================================
# TARGETS
# ============================================================

def historical_levels(
    events,
    current_price,
):

    ups = finite_values(
        event["max_up"]
        for event in events
    )

    downs = finite_values(
        event["max_down"]
        for event in events
    )

    if ups.empty:

        target1 = (
            current_price
            * 1.05
        )

        target2 = (
            current_price
            * 1.10
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
            current_price
            * 0.95
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
# ANALYZE SYMBOL
# ============================================================

def analyze_symbol(
    symbol,
):

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
        subset=[
            "Close"
        ]
    ).copy()

    history_length = len(
        usable
    )


    if (
        history_length
        < MIN_HISTORY
    ):

        print(
            f"[SKIP] {symbol}: "
            f"history={history_length} "
            f"< {MIN_HISTORY}"
        )

        return [], [], []


    events = extract_events(
        usable
    )


    print(
        f"[INFO] {symbol}: "
        f"raw_events={len(events)}"
    )


    # ========================================================
    # 단일 사건
    # ========================================================

    if len(events) == 1:

        event = events[0]

        current_window = usable.iloc[
            -WINDOW:
        ].copy()

        current_signature = (
            make_signature(
                current_window
            )
        )

        similarity = cosine_similarity(
            current_signature,
            event["signature"],
        )

        current_price = safe_float(
            usable[
                "Close"
            ].iloc[-1]
        )

        pattern = event[
            "label"
        ]

        target1, target2, invalidation = (
            historical_levels(
                [event],
                current_price,
            )
        )

        base_cluster_id = (
            make_base_cluster_id(
                symbol,
                pattern,
                [event],
            )
        )

        cluster_id = (
            make_cycle_cluster_id(
                symbol,
                base_cluster_id,
                "단기",
                np.nan,
            )
        )

        phase = current_phase(
            usable
        )

        current = {

            "symbol":
                symbol,

            "as_of":
                usable.index[-1]
                .date()
                .isoformat(),

            "current_price":
                rnd(
                    current_price,
                    4,
                ),

            "active_driver":
                "DRIVER_1",

            "cluster_id":
                cluster_id,

            "base_cluster_id":
                base_cluster_id,

            "pattern":
                f"[단기] {pattern}",

            "cycle_type":
                "단기",

            "cycle_mode_days":
                np.nan,

            "cycle_stability_pct":
                np.nan,

            "state":
                "PROVISIONAL",

            "phase":
                phase,

            "similarity_pct":
                rnd(
                    similarity * 100,
                    2,
                ),

            "success_rate_pct":
                np.nan,

            "plus5_first_pct":
                np.nan,

            "plus10_first_pct":
                np.nan,

            "plus20_first_pct":
                np.nan,

            "minus5_first_pct":
                np.nan,

            "recent_occurrences":
                1,

            "recent_plus5_first_pct":
                np.nan,

            "recent_plus10_first_pct":
                np.nan,

            "recent_plus20_first_pct":
                np.nan,

            "recent_minus5_first_pct":
                np.nan,

            "target1":
                rnd(
                    target1,
                    4,
                ),

            "target2":
                rnd(
                    target2,
                    4,
                ),

            "invalidation":
                rnd(
                    invalidation,
                    4,
                ),

            "occurrences":
                1,

            "cycle_status":
                "INSUFFICIENT_REPEAT",

            "validation_status":
                "PROVISIONAL",
        }


        profile = {

            "symbol":
                symbol,

            "driver_rank":
                1,

            "driver_name":
                "DRIVER_1",

            "cluster_id":
                cluster_id,

            "base_cluster_id":
                base_cluster_id,

            "pattern":
                f"[단기] {pattern}",

            "cycle_type":
                "단기",

            "cycle_mode_days":
                np.nan,

            "cycle_stability_pct":
                np.nan,

            "state":
                "PROVISIONAL",

            "validation_status":
                "PROVISIONAL",

            "phase":
                phase,

            "history_days":
                history_length,

            "current_price":
                rnd(
                    current_price,
                    4,
                ),

            "current_similarity_pct":
                rnd(
                    similarity * 100,
                    2,
                ),

            "occurrences":
                1,

            "success_count":
                np.nan,

            "success_rate_pct":
                np.nan,

            "plus5_first_count":
                np.nan,

            "plus5_first_rate_pct":
                np.nan,

            "plus5_valid_n":
                np.nan,

            "plus10_first_count":
                np.nan,

            "plus10_first_rate_pct":
                np.nan,

            "plus10_valid_n":
                np.nan,

            "plus20_first_count":
                np.nan,

            "plus20_first_rate_pct":
                np.nan,

            "plus20_valid_n":
                np.nan,

            "minus5_first_count":
                np.nan,

            "minus5_first_rate_pct":
                np.nan,

            "minus5_valid_n":
                np.nan,

            "ambiguous_count":
                np.nan,

            "no_hit_count":
                np.nan,

            "recent_occurrences":
                1,

            "recent_plus5_first_pct":
                np.nan,

            "recent_plus10_first_pct":
                np.nan,

            "recent_plus20_first_pct":
                np.nan,

            "recent_minus5_first_pct":
                np.nan,

            "avg_forward_return_pct":
                pct(
                    event.get(
                        "forward_return",
                        np.nan,
                    )
                ),

            "median_forward_return_pct":
                pct(
                    event.get(
                        "forward_return",
                        np.nan,
                    )
                ),

            "median_max_up_pct":
                pct(
                    event.get(
                        "max_up",
                        np.nan,
                    )
                ),

            "median_max_down_pct":
                pct(
                    event.get(
                        "max_down",
                        np.nan,
                    )
                ),

            "median_cycle_days":
                np.nan,

            "mean_cycle_days":
                np.nan,

            "cycle_std_days":
                np.nan,

            "cycle_observations":
                0,

            "target1":
                rnd(
                    target1,
                    4,
                ),

            "target2":
                rnd(
                    target2,
                    4,
                ),

            "invalidation":
                rnd(
                    invalidation,
                    4,
                ),

            "first_seen":
                pd.Timestamp(
                    event["date"]
                )
                .date()
                .isoformat(),

            "last_seen":
                pd.Timestamp(
                    event["date"]
                )
                .date()
                .isoformat(),
        }


        history = make_history_row(
            symbol,
            1,
            "DRIVER_1",
            cluster_id,
            f"[단기] {pattern}",
            "단기",
            np.nan,
            event,
        )

        print(
            f"[OK] {symbol}: "
            f"PROVISIONAL "
            f"cluster={cluster_id}"
        )

        return (
            [profile],
            [current],
            [history],
        )


    # ========================================================
    # 반복 사건 부족
    # ========================================================

    if len(events) < 2:

        print(
            f"[SKIP] {symbol}: "
            "not enough recurring events"
        )

        return [], [], []


    # ========================================================
    # 1차 Pattern Cluster
    # ========================================================

    raw_clusters = cluster_patterns(
        events,
        CLUSTER_SIM,
    )

    recurring_clusters = [
        cluster
        for cluster in raw_clusters
        if len(
            cluster["events"]
        ) >= MIN_DRIVER_OCCURRENCES
    ]


    if not recurring_clusters:

        print(
            f"[SKIP] {symbol}: "
            "no recurring pattern cluster"
        )

        return [], [], []


    # ========================================================
    # 2차 Cycle Split
    # ========================================================

    cycle_clusters = []

    for cluster in recurring_clusters:

        groups = cycle_groups(
            cluster["events"]
        )

        for group in groups:

            if len(
                group["events"]
            ) < MIN_DRIVER_OCCURRENCES:

                continue

            cycle_clusters.append(
                
              for group in groups:

            if len(
                group["events"]
            ) < MIN_DRIVER_OCCURRENCES:

                continue

            # 원래 1차 Pattern Cluster의 centroid를
            # 2차 Cycle Cluster까지 반드시 전달한다.
            #
            # 이후 DRIVER 중복 제거에서
            #
            # candidate["cluster"]["centroid"]
            #
            # 를 사용하기 때문에 필수다.
            source_centroid = cluster.get(
                "centroid"
            )

            if source_centroid is None:

                signatures = [
                    event["signature"]
                    for event
                    in group["events"]
                ]

                if signatures:

                    source_centroid = np.mean(
                        signatures,
                        axis=0,
                    )

                    norm = np.linalg.norm(
                        source_centroid
                    )

                    if norm > 1e-12:

                        source_centroid = (
                            source_centroid
                            / norm
                        )

                else:

                    source_centroid = np.zeros(
                        len(
                            group["events"][0][
                                "signature"
                            ]
                        )
                    )

            # pattern label
            labels = [
                event["label"]
                for event
                in group["events"]
            ]

            modes = pd.Series(
                labels
            ).mode()

            pattern = (
                str(
                    modes.iloc[0]
                )
                if not modes.empty
                else "구조불명"
            )

            # BASE cluster ID는
            # 원래 Pattern Cluster 기준으로 만든다.
            base_cluster_id = (
                make_base_cluster_id(
                    symbol,
                    pattern,
                    group["events"],
                )
            )

            cycle_clusters.append(
                {
                    "events":
                        group["events"],

                    "cycle_type":
                        group["cycle_type"],

                    "cycle_mode":
                        group["cycle_mode"],

                    "cycle_observations":
                        group[
                            "cycle_observations"
                        ],

                    "cycle_pairs":
                        group.get(
                            "cycle_pairs",
                            [],
                        ),

                    # ★ 핵심 수정
                    "centroid":
                        source_centroid,

                    "base_cluster_id":
                        base_cluster_id,
                }
            )
            )

    if not cycle_clusters:

        print(
            f"[SKIP] {symbol}: "
            "no cycle-validated Driver"
        )

        return [], [], []


    # ========================================================
    # Current
    # ========================================================

    current_window = usable.iloc[
        -WINDOW:
    ].copy()

    current_signature = (
        make_signature(
            current_window
        )
    )

    current_price = safe_float(
        usable[
            "Close"
        ].iloc[-1]
    )

    phase = current_phase(
        usable
    )


    driver_candidates = []


    # ========================================================
    # Analyze each cycle cluster
    # ========================================================

    for cluster in cycle_clusters:

        cluster_events = (
            cluster["events"]
        )

        similarities = [
            cosine_similarity(
                current_signature,
                event[
                    "signature"
                ],
            )
            for event
            in cluster_events
        ]

        current_similarity = (
            max(
                similarities
            )
            if similarities
            else 0.0
        )


        labels = [
            event[
                "label"
            ]
            for event
            in cluster_events
        ]

        modes = (
            pd.Series(
                labels
            ).mode()
        )

        pattern = (
            str(
                modes.iloc[0]
            )
            if not modes.empty
            else "구조불명"
        )


        cycle_type = (
            cluster[
                "cycle_type"
            ]
        )

        cycle_mode = safe_float(
            cluster[
                "cycle_mode"
            ]
        )


        cycle_median, cycle_mean, cycle_std, cycle_obs, cycle_stability = (
            cycle_statistics(
                cluster_events,
                cycle_mode,
            )
        )


        pair_stability = (
            cycle_stability_from_pairs(
                cluster.get(
                    "cycle_pairs",
                    [],
                ),
                cycle_mode,
            )
        )


        if np.isfinite(
            safe_float(
                pair_stability
            )
        ):

            final_cycle_stability = (
                pair_stability
            )

        else:

            final_cycle_stability = (
                cycle_stability
            )


        display_cycle = (
            cycle_mode
            if np.isfinite(
                cycle_mode
            )
            else cycle_median
        )


        base_cluster_id = (
            cluster[
                "base_cluster_id"
            ]
        )


        cluster_id = (
            make_cycle_cluster_id(
                symbol,
                base_cluster_id,
                cycle_type,
                display_cycle,
            )
        )


        ordered = ordered_stats(
            cluster_events
        )


        recent = recent_ordered_stats(
            cluster_events
        )


        occurrence_count = len(
            cluster_events
        )


        success_count = (
            ordered[
                "plus5_first_count"
            ]
        )


        success_rate = (
            ordered[
                "plus5_first_rate"
            ]
        )


        forward_returns = finite_values(
            event[
                "forward_return"
            ]
            for event
            in cluster_events
        )


        max_ups = finite_values(
            event[
                "max_up"
            ]
            for event
            in cluster_events
        )


        max_downs = finite_values(
            event[
                "max_down"
            ]
            for event
            in cluster_events
        )


        target1, target2, invalidation = (
            historical_levels(
                cluster_events,
                current_price,
            )
        )


        validation = validation_status(
            occurrence_count,
            final_cycle_stability,
            ordered[
                "plus5_valid_n"
            ],
        )


        state = driver_state(
            current_similarity,
            occurrence_count,
            history_length,
            validation,
        )


        driver_candidates.append(
            {

                "cluster":
                    cluster,

                "cluster_id":
                    cluster_id,

                "base_cluster_id":
                    base_cluster_id,

                "pattern":
                    pattern,

                "cycle_type":
                    cycle_type,

                "cycle_mode":
                    display_cycle,

                "cycle_observations":
                    (
                        cycle_obs
                        if cycle_obs > 0
                        else cluster[
                            "cycle_observations"
                        ]
                    ),

                "cycle_stability":
                    final_cycle_stability,

                "cycle_mean":
                    cycle_mean,

                "cycle_std":
                    cycle_std,

                "current_similarity":
                    current_similarity,

                "occurrences":
                    occurrence_count,

                "success_count":
                    success_count,

                "success_rate":
                    success_rate,

                "ordered":
                    ordered,

                "recent":
                    recent,

                "forward_returns":
                    forward_returns,

                "max_ups":
                    max_ups,

                "max_downs":
                    max_downs,

                "target1":
                    target1,

                "target2":
                    target2,

                "invalidation":
                    invalidation,

                "state":
                    state,

                "validation_status":
                    validation,
            }
        )


    # ========================================================
    # Candidate score
    # ========================================================

    def candidate_score(item):

        sim = max(
            item[
                "current_similarity"
            ],
            0.0,
        )


        occurrence = min(
            item[
                "occurrences"
            ],
            8,
        ) / 8.0


        success = safe_float(
            item[
                "success_rate"
            ]
        )

        if not np.isfinite(
            success
        ):
            success = 0.0


        cycle_stability = safe_float(
            item[
                "cycle_stability"
            ]
        )

        if not np.isfinite(
            cycle_stability
        ):
            cycle_stability = 0.0


        # 최근 재현성
        recent_success = safe_float(
            item[
                "recent"
            ][
                "recent_plus5_first_pct"
            ]
        )

        if not np.isfinite(
            recent_success
        ):
            recent_success = (
                success * 100.0
                if np.isfinite(
                    success
                )
                else 0.0
            )

        recent_success /= 100.0


        # cycle 안정성
        #
        # 현재 유사도 40%
        # 전체 반복 15%
        # +5 선후관계 15%
        # cycle 안정성 15%
        # 최근 재현성 15%
        #
        # 특정 한 숫자만으로 Driver를 결정하지 않는다.
        return (
            sim * 0.40
            + occurrence * 0.15
            + success * 0.15
            + cycle_stability * 0.15
            + recent_success * 0.15
        )


    driver_candidates.sort(
        key=candidate_score,
        reverse=True,
    )


    # ========================================================
    # 중복 Driver 제거
    # ========================================================

    selected = []

    max_allowed = min(
        allowed_driver_count(
            history_length
        ),
        MAX_DRIVERS,
    )


    for candidate in driver_candidates:

        if len(selected) >= max_allowed:
            break

        duplicate = False

        for existing in selected:

            same_base = (
                candidate[
                    "base_cluster_id"
                ]
                == existing[
                    "base_cluster_id"
                ]
            )

            same_cycle = (
                candidate[
                    "cycle_type"
                ]
                == existing[
                    "cycle_type"
                ]
            )

            if (
                same_base
                and same_cycle
            ):

                duplicate = True
                break

            sim = cosine_similarity(
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

            if sim >= 0.97:

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

                    if ratio < 1.35:

                        duplicate = True
                        break


        if not duplicate:

            selected.append(
                candidate
            )


    if not selected:

        return [], [], []


    profiles = []
    current_rows = []
    history_rows = []


    # ========================================================
    # Driver rank
    # ========================================================

    for rank, item in enumerate(
        selected,
        start=1,
    ):

        cluster_events = (
            item[
                "cluster"
            ]["events"]
        )


        driver_name = (
            f"DRIVER_{rank}"
        )


        pattern = item[
            "pattern"
        ]


        cycle_type = item[
            "cycle_type"
        ]


        display_pattern = (
            f"[{cycle_type}] "
            f"{pattern}"
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
                cycle_type,

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

            "history_days":
                history_length,

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

            "occurrences":
                item[
                    "occurrences"
                ],

            "success_count":
                item[
                    "success_count"
                ],

            "success_rate_pct":
                rnd(
                    item[
                        "success_rate"
                    ] * 100
                    if np.isfinite(
                        safe_float(
                            item[
                                "success_rate"
                            ]
                        )
                    )
                    else np.nan,
                    2,
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

            "median_cycle_days":
                rnd(
                    item[
                        "cycle_mode"
                    ],
                    1,
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

            "cycle_observations":
                item[
                    "cycle_observations"
                ],

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
                    pd.Timestamp(
                        event["date"]
                    )
                    .date()
                    .isoformat()
                    for event
                    in cluster_events
                ),

            "last_seen":
                max(
                    pd.Timestamp(
                        event["date"]
                    )
                    .date()
                    .isoformat()
                    for event
                    in cluster_events
                ),
        }


        profiles.append(
            profile
        )


        # ====================================================
        # History
        # ====================================================

        for event in cluster_events:

            history_rows.append(
                make_history_row(
                    symbol,
                    rank,
                    driver_name,
                    item[
                        "cluster_id"
                    ],
                    display_pattern,
                    cycle_type,
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
    # Current Driver
    # ========================================================

    if profiles:

        top = profiles[0]

        current_rows.append(
            {

                "symbol":
                    symbol,

                "as_of":
                    usable.index[-1]
                    .date()
                    .isoformat(),

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

                "similarity_pct":
                    top[
                        "current_similarity_pct"
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
                            ] >= 2
                        )
                        else
                        "INSUFFICIENT_REPEAT"
                    ),
            }
        )


    # ========================================================
    # Console
    # ========================================================

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
            f"cluster={profile['cluster_id']} | "
            f"sim={profile['current_similarity_pct']:.1f}% | "
            f"+5first={profile['plus5_first_rate_pct']:.1f}% | "
            f"+10first={profile['plus10_first_rate_pct']:.1f}% | "
            f"+20first={profile['plus20_first_rate_pct']:.1f}% | "
            f"-5first={profile['minus5_first_rate_pct']:.1f}% | "
            f"cycle={profile['cycle_mode_days']}d | "
            f"cycle_stab={profile['cycle_stability_pct']}% | "
            f"occ={profile['occurrences']} | "
            f"validation={profile['validation_status']} | "
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
# HISTORY ROW
# ============================================================

def make_history_row(
    symbol,
    driver_rank,
    driver_name,
    cluster_id,
    pattern,
    cycle_type,
    cycle_mode,
    event,
    base_cluster_id=None,
):

    return {

        "symbol":
            symbol,

        "driver_rank":
            driver_rank,

        "driver_name":
            driver_name,

        "cluster_id":
            cluster_id,

        "base_cluster_id":
            base_cluster_id,

        "pattern":
            pattern,

        "cycle_type":
            cycle_type,

        "cycle_mode_days":
            rnd(
                cycle_mode,
                1,
            ),

        "event_date":
            pd.Timestamp(
                event["date"]
            )
            .date()
            .isoformat(),

        "forward_return_pct":
            pct(
                event.get(
                    "forward_return",
                    np.nan,
                )
            ),

        "max_up_pct":
            pct(
                event.get(
                    "max_up",
                    np.nan,
                )
            ),

        "max_down_pct":
            pct(
                event.get(
                    "max_down",
                    np.nan,
                )
            ),

        "first_event":
            event.get(
                "first_event",
                "NONE",
            ),

        "first_day":
            event.get(
                "first_day",
                np.nan,
            ),

        "outcome_status":
            event.get(
                "outcome_status",
                "NO_HIT",
            ),

        "plus5_first":
            event.get(
                "plus5_first",
                np.nan,
            ),

        "plus10_first":
            event.get(
                "plus10_first",
                np.nan,
            ),

        "plus20_first":
            event.get(
                "plus20_first",
                np.nan,
            ),

        "minus5_first":
            event.get(
                "minus5_first",
                np.nan,
            ),

        "days_to_plus5":
            event.get(
                "days_to_plus5",
                np.nan,
            ),

        "days_to_plus10":
            event.get(
                "days_to_plus10",
                np.nan,
            ),

        "days_to_plus20":
            event.get(
                "days_to_plus20",
                np.nan,
            ),

        "days_to_minus5":
            event.get(
                "days_to_minus5",
                np.nan,
            ),
    }


# ============================================================
# CURRENT PHASE
# ============================================================

def current_phase(df):

    if len(df) < 60:
        return "INSUFFICIENT_HISTORY"

    last = df.iloc[-1]

    close = safe_float(
        last["Close"]
    )

    ma20 = safe_float(
        last["ma20"]
    )

    ma50 = safe_float(
        last["ma50"]
    )

    slope20 = safe_float(
        last.get(
            "slope20",
            0,
        ),
        0,
    )

    volume_ratio = safe_float(
        last.get(
            "vol_ratio",
            1,
        ),
        1,
    )

    recent_high = safe_float(
        df[
            "High"
        ].tail(20).max()
    )

    recent_low = safe_float(
        df[
            "Low"
        ].tail(20).min()
    )

    if (
        close
        >= recent_high * 0.995
        and volume_ratio >= 1.25
    ):
        return "BREAKOUT"

    if (
        close
        <= recent_low * 1.005
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
# SYMBOL LOADING
# ============================================================

def load_symbols_from_project():

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
            str(value)
            .strip()
            .upper()
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
        " UNIVERSAL DRIVER ENGINE v6"
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
        f"min_driver_occurrences : {MIN_DRIVER_OCCURRENCES}"
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


if __name__ == "__main__":

    symbols = (
        load_symbols_from_project()
    )

    run(symbols)


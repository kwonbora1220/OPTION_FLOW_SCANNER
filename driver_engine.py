```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNIVERSAL DRIVER ENGINE v5
==========================

목적
----
종목별 과거 OHLCV에서 반복되는 "운전수(Driver)"를 탐색한다.

핵심 개념
---------
1. 선후관계 적중률
   +5% 먼저 / -5% 먼저
   +10% 먼저 / -5% 먼저
   +20% 먼저 / -5% 먼저

   단순히 둘 다 도달했다고 성공으로 계산하지 않는다.

2. Driver 주기 분리
   [단기]
   [장기]

   예:
   UBER
   58일 반복
   581일 반복

   → 같은 Driver로 합치지 않는다.

3. cluster_id
   각 Driver cluster에 고유 ID를 부여한다.

   과거 발생 사건이 어느 Driver였는지
   driver_rank가 바뀌어도 추적할 수 있도록 한다.

4. 사건 중복 제거
   동일한 상승/하락 구간에서 40개 창이 생기는 문제를 줄인다.

5. 신규 상장
   반복 사건이 1개밖에 없으면 PROVISIONAL Driver로 표시한다.

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

# 동일한 움직임을 여러 번 사건으로 세지 않기 위한 최소 간격
EVENT_SPACING = int(
    os.getenv(
        "DRIVER_EVENT_SPACING",
        "40",
    )
)

# Cycle 계산 최소 기간
CYCLE_MIN_DAYS = int(
    os.getenv(
        "DRIVER_CYCLE_MIN_DAYS",
        "25",
    )
)

# 단기/장기 cycle 분리를 위한 최소 비율
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
```

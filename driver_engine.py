#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNIVERSAL DRIVER ENGINE v3
==========================

목적
- 종목별 과거 OHLCV에서 반복되는 "운전수(Driver)"를 자동 탐색
- 종목마다 실제 데이터에 따라 1~3개 Driver 후보를 자동 선택
- 겹치는 윈도우를 같은 사건으로 과대계상하지 않음
- 실제 반복 발생일 사이의 간격을 계산
- 현재 패턴과 과거 Driver 유사도 계산
- +5% / +10% / +20% 도달률과 -5% 선이탈률 계산
- 신규 상장 종목은 PROVISIONAL Driver를 허용
- 장기 사이클은 충분한 역사 데이터가 있을 때만 표시
- 기존 옵션 스캐너와 독립 실행

실행:
    python driver_engine.py
    python driver_engine.py RKLB UBER CBRS

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

# 이벤트끼리 최소한 이 정도 떨어져 있어야 별도 발생으로 인정
EVENT_SPACING = int(
    os.getenv(
        "DRIVER_EVENT_SPACING",
        "40",
    )
)

# 같은 Cluster 안에서 서로 다른 실제 발생으로 인정하기 위한 최소 거리
CYCLE_MIN_DAYS = int(
    os.getenv(
        "DRIVER_CYCLE_MIN_DAYS",
        "25",
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
# SIGNATURE
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

    # 가격 경로 20
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

    # 거래량 8
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

    # MA 구조
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

    # 변동성
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
# HUMAN PATTERN LABEL
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

    ma20_gap = float(
        last.get(
            "ma20_gap",
            0,
        )
    )

    ma50_gap = float(
        last.get(
            "ma50_gap",
            0,
        )
    )

    slope20 = float(
        last.get(
            "slope20",
            0,
        )
    )

    current_vol = float(
        window[
            "vol20"
        ].iloc[-1]
    )

    median_vol = float(
        window[
            "vol20"
        ].median()
    )

    # 급락 → 저점 → 회복
    if (
        trough_return <= -0.18
        and trough_pos
        < len(close) * 0.80
        and total > -0.10
    ):

        return (
            "급락→저점형성→회복"
        )

    # 급등 → 조정 → 재상승
    if (
        peak_return >= 0.20
        and trough_pos > peak_pos
        and second_move > 0.05
    ):

        return (
            "급등→조정→재상승"
        )

    # 압축 → 돌파 → 확장
    if (
        np.isfinite(
            current_vol
        )
        and np.isfinite(
            median_vol
        )
        and median_vol > 0
        and current_vol
        > median_vol * 1.15
        and total > 0.10
    ):

        return (
            "압축→돌파→확장"
        )

    # 추세 상승
    if (
        total > 0.12
        and ma20_gap > 0
        and ma50_gap > -0.02
        and slope20 > 0
    ):

        return (
            "추세상승→확장"
        )

    # 추세 하락
    if (
        total < -0.12
        and ma20_gap < 0
        and ma50_gap < 0
        and slope20 < 0
    ):

        return (
            "추세하락→반등시도"
        )

    # 박스
    if abs(total) < 0.10:

        return (
            "박스→반복회귀"
        )

    return "추세전환형"


# ============================================================
# FUTURE OUTCOME ENGINE
# ============================================================

def future_outcome(
    df,
    end,
):

    entry = float(
        df[
            "Close"
        ].iloc[
            end - 1
        ]
    )

    if entry <= 0:
        return {}

    future = df.iloc[
        end:
        min(
            end + FORWARD,
            len(df),
        )
    ]

    if future.empty:
        return {}

    high_ret = (
        future[
            "High"
        ]
        / entry
        - 1
    )

    low_ret = (
        future[
            "Low"
        ]
        / entry
        - 1
    )

    close_ret = (
        future[
            "Close"
        ]
        / entry
        - 1
    )

    result = {
        "forward_return": float(
            close_ret.iloc[-1]
        ),
        "max_up": float(
            high_ret.max()
        ),
        "max_down": float(
            low_ret.min()
        ),
    }

    # --------------------------------------------------------
    # +5 / +10 / +20 도달 여부
    # --------------------------------------------------------

    for target in [
        0.05,
        0.10,
        0.20,
    ]:

        reached = (
            high_ret >= target
        )

        if reached.any():

            first_position = (
                int(
                    np.argmax(
                        reached.to_numpy()
                    )
                )
                + 1
            )

            result[
                f"hit_{int(target * 100)}"
            ] = True

            result[
                f"days_to_{int(target * 100)}"
            ] = first_position

        else:

            result[
                f"hit_{int(target * 100)}"
            ] = False

            result[
                f"days_to_{int(target * 100)}"
            ] = np.nan

    # --------------------------------------------------------
    # -5 / -10 선이탈
    # --------------------------------------------------------

    for stop in [
        -0.05,
        -0.10,
    ]:

        reached = (
            low_ret <= stop
        )

        name = str(
            int(
                abs(stop)
                * 100
            )
        )

        if reached.any():

            first_position = (
                int(
                    np.argmax(
                        reached.to_numpy()
                    )
                )
                + 1
            )

            result[
                f"hit_down_{name}"
            ] = True

            result[
                f"days_to_down_{name}"
            ] = first_position

        else:

            result[
                f"hit_down_{name}"
            ] = False

            result[
                f"days_to_down_{name}"
            ] = np.nan

    return result


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

        outcome = future_outcome(
            df,
            end,
        )

        if not outcome:
            continue

        event = {
            "end_idx": end,
            "date": df.index[
                end - 1
            ],
            "signature": signature,
            "label": choose_pattern_label(
                window
            ),
            **outcome,
        }

        events.append(
            event
        )

    # --------------------------------------------------------
    # 중요:
    # 40일 패턴을 매일 평가하면 동일한 움직임이 수십 번
    # 중복된다. 여기서는 최소 40거래일 간격으로 이벤트를
    # 먼저 줄인다.
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
# CLUSTER
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

            similarity = (
                cosine_similarity(
                    signature,
                    cluster[
                        "centroid"
                    ],
                )
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
# REAL CYCLE STATISTICS
# ============================================================

def cycle_statistics(
    events,
):

    if len(events) < 2:

        return (
            np.nan,
            np.nan,
            np.nan,
            0,
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

        if gap >= CYCLE_MIN_DAYS:

            gaps.append(
                gap
            )

    if not gaps:

        return (
            np.nan,
            np.nan,
            np.nan,
            0,
        )

    return (
        float(
            np.median(gaps)
        ),
        float(
            np.mean(gaps)
        ),
        float(
            np.std(gaps)
        ),
        len(gaps),
    )


# ============================================================
# DRIVER STATE
# ============================================================

def driver_state(
    similarity,
    occurrences,
    history_length,
):

    if (
        occurrences < 2
    ):

        # 신규상장/데이터 부족 상태
        if history_length < 300:
            return "PROVISIONAL"

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
# TARGET / INVALIDATION
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

        # 과거 MAE의 25% 분위수 사용
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
# ANALYZE
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

    # --------------------------------------------------------
    # 신규 상장 예외
    # --------------------------------------------------------

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

        similarity = (
            cosine_similarity(
                current_signature,
                event["signature"],
            )
        )

        current_price = float(
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

        current = {
            "symbol": symbol,
            "as_of": (
                usable.index[-1]
                .date()
                .isoformat()
            ),
            "current_price": rnd(
                current_price,
                4,
            ),
            "active_driver": "DRIVER_1",
            "pattern": pattern,
            "state": "PROVISIONAL",
            "phase": current_phase(
                usable
            ),
            "similarity_pct": rnd(
                similarity * 100,
                2,
            ),
            "success_rate_pct": np.nan,
            "hit_5_pct": np.nan,
            "hit_10_pct": np.nan,
            "hit_20_pct": np.nan,
            "stop_5_pct": np.nan,
            "stop_10_pct": np.nan,
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
            "occurrences": 1,
            "cycle_status":
                "INSUFFICIENT_HISTORY",
            "median_cycle_days":
                np.nan,
        }

        profile = {
            "symbol": symbol,
            "driver_rank": 1,
            "driver_name": "DRIVER_1",
            "pattern": pattern,
            "state": "PROVISIONAL",
            "phase": current[
                "phase"
            ],
            "history_days": history_length,
            "current_price": rnd(
                current_price,
                4,
            ),
            "current_similarity_pct": rnd(
                similarity * 100,
                2,
            ),
            "occurrences": 1,
            "success_count": np.nan,
            "success_rate_pct": np.nan,
            "hit_5_rate_pct": pct(
                event.get(
                    "hit_5",
                    np.nan,
                )
                if event.get(
                    "hit_5",
                    False,
                )
                else np.nan
            ),
            "hit_10_rate_pct": np.nan,
            "hit_20_rate_pct": np.nan,
            "stop_5_rate_pct": np.nan,
            "stop_10_rate_pct": np.nan,
            "avg_forward_return_pct": pct(
                event.get(
                    "forward_return",
                    np.nan,
                )
            ),
            "median_forward_return_pct": pct(
                event.get(
                    "forward_return",
                    np.nan,
                )
            ),
            "median_max_up_pct": pct(
                event.get(
                    "max_up",
                    np.nan,
                )
            ),
            "median_max_down_pct": pct(
                event.get(
                    "max_down",
                    np.nan,
                )
            ),
            "median_cycle_days": np.nan,
            "mean_cycle_days": np.nan,
            "cycle_std_days": np.nan,
            "cycle_observations": 0,
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
                pd.Timestamp(
                    event["date"]
                )
                .date()
                .isoformat()
            ),
            "last_seen": (
                pd.Timestamp(
                    event["date"]
                )
                .date()
                .isoformat()
            ),
        }

        history = {
            "symbol": symbol,
            "driver_rank": 1,
            "driver_name": "DRIVER_1",
            "pattern": pattern,
            "event_date": (
                pd.Timestamp(
                    event["date"]
                )
                .date()
                .isoformat()
            ),
            "forward_return_pct": pct(
                event[
                    "forward_return"
                ]
            ),
            "max_up_pct": pct(
                event[
                    "max_up"
                ]
            ),
            "max_down_pct": pct(
                event[
                    "max_down"
                ]
            ),
            "hit_5": event.get(
                "hit_5",
                False,
            ),
            "hit_10": event.get(
                "hit_10",
                False,
            ),
            "hit_20": event.get(
                "hit_20",
                False,
            ),
            "hit_down_5": event.get(
                "hit_down_5",
                False,
            ),
            "hit_down_10": event.get(
                "hit_down_10",
                False,
            ),
            "days_to_5": event.get(
                "days_to_5",
                np.nan,
            ),
            "days_to_10": event.get(
                "days_to_10",
                np.nan,
            ),
            "days_to_20": event.get(
                "days_to_20",
                np.nan,
            ),
            "days_to_down_5": event.get(
                "days_to_down_5",
                np.nan,
            ),
            "days_to_down_10": event.get(
                "days_to_down_10",
                np.nan,
            ),
        }

        print(
            f"[OK] {symbol}: "
            f"history={history_length} "
            f"events=1 "
            f"drivers=1 "
            f"PROVISIONAL "
            f"phase={current['phase']} "
            f"price={current_price:.2f}"
        )

        print(
            "  DRIVER_1 | "
            f"{pattern} | "
            "PROVISIONAL | "
            f"sim={similarity * 100:.1f}% | "
            "success=N/A | "
            f"T1={target1:.2f} | "
            f"T2={target2:.2f} | "
            f"INV={invalidation:.2f}"
        )

        return (
            [profile],
            [current],
            [history],
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
        if len(
            cluster["events"]
        ) >= 2
    ]

    # 신규 종목은 1개,
    # 중기 2개,
    # 장기 3개
    max_allowed = min(
        allowed_driver_count(
            history_length
        ),
        MAX_DRIVERS,
    )

    valid_clusters.sort(
        key=lambda cluster: (
            len(
                cluster["events"]
            ),
            (
                float(
                    finite_values(
                        event[
                            "forward_return"
                        ]
                        for event
                        in cluster[
                            "events"
                        ]
                    ).mean()
                )
                if not finite_values(
                    event[
                        "forward_return"
                    ]
                    for event
                    in cluster[
                        "events"
                    ]
                ).empty
                else -999.0
            ),
        ),
        reverse=True,
    )

    selected = valid_clusters[
        :max_allowed
    ]

    if not selected:

        print(
            f"[SKIP] {symbol}: "
            "no recurring Driver found"
        )

        return [], [], []

    current_window = usable.iloc[
        -WINDOW:
    ].copy()

    current_signature = (
        make_signature(
            current_window
        )
    )

    current_price = float(
        usable[
            "Close"
        ].iloc[-1]
    )

    phase = current_phase(
        usable
    )

    profiles = []
    current_rows = []
    history_rows = []

    for cluster in selected:

        cluster_events = (
            cluster[
                "events"
            ]
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
            max(similarities)
            if similarities
            else 0.0
        )

        forward_returns = (
            finite_values(
                event[
                    "forward_return"
                ]
                for event
                in cluster_events
            )
        )

        max_ups = (
            finite_values(
                event[
                    "max_up"
                ]
                for event
                in cluster_events
            )
        )

        max_downs = (
            finite_values(
                event[
                    "max_down"
                ]
                for event
                in cluster_events
            )
        )

        occurrence_count = len(
            cluster_events
        )

        success_count = sum(
            bool(
                event.get(
                    "hit_5",
                    False,
                )
            )
            for event
            in cluster_events
        )

        hit_5_count = sum(
            bool(
                event.get(
                    "hit_5",
                    False,
                )
            )
            for event
            in cluster_events
        )

        hit_10_count = sum(
            bool(
                event.get(
                    "hit_10",
                    False,
                )
            )
            for event
            in cluster_events
        )

        hit_20_count = sum(
            bool(
                event.get(
                    "hit_20",
                    False,
                )
            )
            for event
            in cluster_events
        )

        stop_5_count = sum(
            bool(
                event.get(
                    "hit_down_5",
                    False,
                )
            )
            for event
            in cluster_events
        )

        stop_10_count = sum(
            bool(
                event.get(
                    "hit_down_10",
                    False,
                )
            )
            for event
            in cluster_events
        )

        success_rate = (
            success_count
            / occurrence_count
        )

        hit_5_rate = (
            hit_5_count
            / occurrence_count
        )

        hit_10_rate = (
            hit_10_count
            / occurrence_count
        )

        hit_20_rate = (
            hit_20_count
            / occurrence_count
        )

        stop_5_rate = (
            stop_5_count
            / occurrence_count
        )

        stop_10_rate = (
            stop_10_count
            / occurrence_count
        )

        (
            median_cycle_days,
            mean_cycle_days,
            cycle_std_days,
            cycle_observations,
        ) = cycle_statistics(
            cluster_events
        )

        target1, target2, invalidation = (
            historical_levels(
                cluster_events,
                current_price,
            )
        )

        state = driver_state(
            current_similarity,
            occurrence_count,
            history_length,
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

        dates = sorted(
            pd.Timestamp(
                event[
                    "date"
                ]
            )
            for event
            in cluster_events
        )

        profile = {
            "symbol": symbol,
            "driver_name": None,
            "driver_rank": None,
            "pattern": pattern,
            "state": state,
            "phase": phase,
            "history_days": history_length,
            "current_price": rnd(
                current_price,
                4,
            ),
            "current_similarity_pct": rnd(
                current_similarity
                * 100,
                2,
            ),
            "occurrences": occurrence_count,
            "success_count": success_count,
            "success_rate_pct": rnd(
                success_rate * 100,
                2,
            ),
            "hit_5_rate_pct": rnd(
                hit_5_rate * 100,
                2,
            ),
            "hit_10_rate_pct": rnd(
                hit_10_rate * 100,
                2,
            ),
            "hit_20_rate_pct": rnd(
                hit_20_rate * 100,
                2,
            ),
            "stop_5_rate_pct": rnd(
                stop_5_rate * 100,
                2,
            ),
            "stop_10_rate_pct": rnd(
                stop_10_rate * 100,
                2,
            ),
            "avg_forward_return_pct": rnd(
                forward_returns.mean()
                * 100
                if not forward_returns.empty
                else np.nan,
                2,
            ),
            "median_forward_return_pct": rnd(
                forward_returns.median()
                * 100
                if not forward_returns.empty
                else np.nan,
                2,
            ),
            "median_max_up_pct": rnd(
                max_ups.median()
                * 100
                if not max_ups.empty
                else np.nan,
                2,
            ),
            "median_max_down_pct": rnd(
                max_downs.median()
                * 100
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
            "cycle_observations":
                cycle_observations,
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
        }

        profiles.append(
            profile
        )

    # --------------------------------------------------------
    # 현재 유사도 + 상태 우선 정렬
    # --------------------------------------------------------

    profiles.sort(
        key=lambda item: (
            {
                "ACTIVE": 0,
                "WATCH": 1,
                "PROVISIONAL": 2,
                "INACTIVE": 3,
            }.get(
                item[
                    "state"
                ],
                9,
            ),
            -float(
                item[
                    "current_similarity_pct"
                ]
            ),
            -int(
                item[
                    "occurrences"
                ]
            ),
        )
    )

    # --------------------------------------------------------
    # 최종 Driver 번호
    # --------------------------------------------------------

    for rank, profile in enumerate(
        profiles,
        start=1,
    ):

        profile[
            "driver_rank"
        ] = rank

        profile[
            "driver_name"
        ] = (
            f"DRIVER_{rank}"
        )

    # --------------------------------------------------------
    # 각 profile에 연결된 history 생성
    # --------------------------------------------------------

    for profile in profiles:

        # profile의 pattern + occurrence 수에 맞는
        # 가장 유사한 cluster를 찾는다.
        candidates = [
            cluster
            for cluster in selected
            if len(
                cluster["events"]
            )
            == profile[
                "occurrences"
            ]
        ]

        if not candidates:

            candidates = selected

        best_cluster = min(
            candidates,
            key=lambda cluster: abs(
                len(
                    cluster["events"]
                )
                - profile[
                    "occurrences"
                ]
            ),
        )

        for event in (
            best_cluster[
                "events"
            ]
        ):

            history_rows.append(
                {
                    "symbol": symbol,
                    "driver_rank":
                        profile[
                            "driver_rank"
                        ],
                    "driver_name":
                        profile[
                            "driver_name"
                        ],
                    "pattern":
                        profile[
                            "pattern"
                        ],
                    "event_date": (
                        pd.Timestamp(
                            event[
                                "date"
                            ]
                        )
                        .date()
                        .isoformat()
                    ),
                    "forward_return_pct":
                        pct(
                            event[
                                "forward_return"
                            ]
                        ),
                    "max_up_pct":
                        pct(
                            event[
                                "max_up"
                            ]
                        ),
                    "max_down_pct":
                        pct(
                            event[
                                "max_down"
                            ]
                        ),
                    "hit_5":
                        event.get(
                            "hit_5",
                            False,
                        ),
                    "hit_10":
                        event.get(
                            "hit_10",
                            False,
                        ),
                    "hit_20":
                        event.get(
                            "hit_20",
                            False,
                        ),
                    "hit_down_5":
                        event.get(
                            "hit_down_5",
                            False,
                        ),
                    "hit_down_10":
                        event.get(
                            "hit_down_10",
                            False,
                        ),
                    "days_to_5":
                        event.get(
                            "days_to_5",
                            np.nan,
                        ),
                    "days_to_10":
                        event.get(
                            "days_to_10",
                            np.nan,
                        ),
                    "days_to_20":
                        event.get(
                            "days_to_20",
                            np.nan,
                        ),
                    "days_to_down_5":
                        event.get(
                            "days_to_down_5",
                            np.nan,
                        ),
                    "days_to_down_10":
                        event.get(
                            "days_to_down_10",
                            np.nan,
                        ),
                }
            )

    # --------------------------------------------------------
    # 현재 최상위 Driver
    # --------------------------------------------------------

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
                "current_price":
                    top[
                        "current_price"
                    ],
                "active_driver":
                    top[
                        "driver_name"
                    ],
                "pattern":
                    top[
                        "pattern"
                    ],
                "state":
                    top[
                        "state"
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
                "hit_5_pct":
                    top[
                        "hit_5_rate_pct"
                    ],
                "hit_10_pct":
                    top[
                        "hit_10_rate_pct"
                    ],
                "hit_20_pct":
                    top[
                        "hit_20_rate_pct"
                    ],
                "stop_5_pct":
                    top[
                        "stop_5_rate_pct"
                    ],
                "stop_10_pct":
                    top[
                        "stop_10_rate_pct"
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
                "cycle_status": (
                    "VALIDATED"
                    if top[
                        "cycle_observations"
                    ] >= 2
                    else
                    "INSUFFICIENT_REPEAT"
                ),
                "median_cycle_days":
                    top[
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
            f"+5={profile['hit_5_rate_pct']:.1f}% | "
            f"+10={profile['hit_10_rate_pct']:.1f}% | "
            f"+20={profile['hit_20_rate_pct']:.1f}% | "
            f"-5={profile['stop_5_rate_pct']:.1f}% | "
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
# CURRENT PHASE
# ============================================================

def current_phase(df):

    if len(df) < 60:
        return (
            "INSUFFICIENT_HISTORY"
        )

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
        df[
            "High"
        ].tail(20).max()
    )

    recent_low = float(
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
        " UNIVERSAL DRIVER ENGINE v3"
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
        f"event_spacing : {EVENT_SPACING}"
    )

    print(
        f"cycle_min     : {CYCLE_MIN_DAYS}"
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

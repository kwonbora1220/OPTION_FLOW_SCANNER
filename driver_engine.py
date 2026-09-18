#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OPTION_FLOW_SCANNER - Universal Driver Engine

목적
- 종목별 과거 가격/거래량/이평/변동성 구조에서 반복되는 "운전수(Driver)"를 자동 탐색
- 종목마다 실제 데이터가 허용하는 범위에서 1~3개의 Driver만 생성
- 현재 구간이 과거 어떤 Driver와 유사한지 계산
- 반복 패턴의 발생 횟수/성공률/평균수익/상승폭/하락폭/반복 간격을 기록
- 기존 옵션 스캐너와 독립적으로 동작하도록 설계

실행:
    python driver_engine.py
    python driver_engine.py RKLB UBER CBRS

환경변수:
    DRIVER_SYMBOLS="RKLB,UBER,CBRS"
    DRIVER_LOOKBACK="10y"
    DRIVER_WINDOW=40
    DRIVER_FORWARD=40
    DRIVER_MIN_HISTORY=90
    DRIVER_MAX_DRIVERS=3

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


OUTPUT_DIR = Path("03_RESULTS/daily")
PROFILE_FILE = OUTPUT_DIR / "driver_profile.csv"
CURRENT_FILE = OUTPUT_DIR / "driver_current.csv"
HISTORY_FILE = OUTPUT_DIR / "driver_history.csv"

LOOKBACK_PERIOD = os.getenv("DRIVER_LOOKBACK", "10y")
WINDOW = int(os.getenv("DRIVER_WINDOW", "40"))
FORWARD = int(os.getenv("DRIVER_FORWARD", "40"))
MIN_HISTORY = int(os.getenv("DRIVER_MIN_HISTORY", "90"))
MAX_DRIVERS = int(os.getenv("DRIVER_MAX_DRIVERS", "3"))

# 현재 패턴을 과거 Driver로 인정하는 유사도
CURRENT_ACTIVE_SIM = float(os.getenv("DRIVER_ACTIVE_SIM", "0.82"))
CURRENT_WATCH_SIM = float(os.getenv("DRIVER_WATCH_SIM", "0.72"))

# 과거 패턴끼리 같은 Driver로 묶는 기준
CLUSTER_SIM = float(os.getenv("DRIVER_CLUSTER_SIM", "0.90"))

# 너무 가까운 중복 이벤트 방지
EVENT_SPACING = int(os.getenv("DRIVER_EVENT_SPACING", "10"))


def clean_download(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 버전에 따라 발생하는 MultiIndex를 안전하게 정리."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        # 단일 종목 다운로드인 경우 첫 번째 레벨만 사용
        if len(out.columns.levels) >= 2:
            new_cols = []
            for col in out.columns:
                # OHLCV 이름을 우선 탐색
                found = None
                for x in col:
                    sx = str(x)
                    if sx in {"Open", "High", "Low", "Close", "Adj Close", "Volume"}:
                        found = sx
                        break
                new_cols.append(found if found else str(col[0]))
            out.columns = new_cols

    rename = {}
    for c in out.columns:
        s = str(c).strip().lower()
        if s == "open":
            rename[c] = "Open"
        elif s == "high":
            rename[c] = "High"
        elif s == "low":
            rename[c] = "Low"
        elif s == "close":
            rename[c] = "Close"
        elif s == "adj close":
            rename[c] = "Adj Close"
        elif s == "volume":
            rename[c] = "Volume"

    out = out.rename(columns=rename)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        return pd.DataFrame()

    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()

    return out


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
    except Exception as e:
        print(f"[WARN] {symbol}: download failed: {e}")
        return pd.DataFrame()


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

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr14"] = tr.rolling(14).mean()
    x["atr_pct"] = x["atr14"] / close.replace(0, np.nan)

    x["vol20"] = x["ret1"].rolling(20).std() * np.sqrt(252)
    x["vol_ratio"] = volume / volume.rolling(20).median().replace(0, np.nan)

    # 이평선 위치
    x["ma20_gap"] = close / x["ma20"] - 1
    x["ma50_gap"] = close / x["ma50"] - 1
    x["ma100_gap"] = close / x["ma100"] - 1
    x["ma200_gap"] = close / x["ma200"] - 1

    # 최근 고점/저점 대비 위치
    x["hh20"] = close / close.rolling(20).max() - 1
    x["ll20"] = close / close.rolling(20).min() - 1
    x["hh60"] = close / close.rolling(60).max() - 1
    x["ll60"] = close / close.rolling(60).min() - 1

    # 추세 기울기
    x["slope20"] = x["ma20"].pct_change(10)
    x["slope50"] = x["ma50"].pct_change(10)

    return x


def zscore(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))

    if not np.isfinite(mad) or mad < 1e-9:
        std = np.nanstd(arr)
        if not np.isfinite(std) or std < 1e-9:
            return np.zeros_like(arr)
        return (arr - np.nanmean(arr)) / std

    return (arr - med) / (1.4826 * mad)


def safe_series(values, length):
    a = np.asarray(values, dtype=float)
    if len(a) != length:
        a = np.resize(a, length)
    a = pd.Series(a).interpolate(limit_direction="both").fillna(0).to_numpy()
    return a


def make_signature(window: pd.DataFrame) -> np.ndarray:
    """
    가격 경로 + 거래량 + 이평 관계 + 변동성 + 추세/위치 정보를
    하나의 고정 길이 벡터로 압축.
    """

    n = len(window)
    if n < 10:
        return np.zeros(1)

    close = window["Close"].to_numpy(dtype=float)
    base = close[0] if close[0] != 0 else 1.0
    price_path = close / base - 1.0

    # 가격 경로는 16개 구간으로 리샘플
    p_idx = np.linspace(0, n - 1, 16)
    p = np.interp(p_idx, np.arange(n), price_path)

    # 거래량
    vr = np.log1p(
        np.clip(
            window["vol_ratio"].fillna(1).to_numpy(dtype=float),
            0,
            100,
        )
    )
    v_idx = np.linspace(0, n - 1, 8)
    v = np.interp(v_idx, np.arange(n), vr)

    # MA 관계
    ma_cols = ["ma20_gap", "ma50_gap", "ma100_gap"]
    ma_parts = []
    for c in ma_cols:
        a = window[c].to_numpy(dtype=float)
        a = safe_series(a, n)
        idx = np.linspace(0, n - 1, 4)
        ma_parts.extend(np.interp(idx, np.arange(n), a))

    # 변동성
    vol = safe_series(window["vol20"].to_numpy(dtype=float), n)
    vol_idx = np.linspace(0, n - 1, 4)
    vol_part = np.interp(vol_idx, np.arange(n), vol)

    # 최근 위치/추세 특징
    last = window.iloc[-1]
    scalar = np.array(
        [
            float(last.get("hh20", 0)),
            float(last.get("ll20", 0)),
            float(last.get("hh60", 0)),
            float(last.get("ll60", 0)),
            float(last.get("slope20", 0)),
            float(last.get("slope50", 0)),
            float(last.get("atr_pct", 0)),
        ],
        dtype=float,
    )

    # 각각 스케일 차이가 크므로 전체 벡터 정규화
    sig = np.concatenate([p, v, np.asarray(ma_parts), vol_part, scalar])
    sig = pd.Series(sig).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    sig = zscore(sig)

    norm = np.linalg.norm(sig)
    if norm < 1e-12:
        return np.zeros_like(sig)

    return sig / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        return 0.0

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-12 or nb < 1e-12:
        return 0.0

    return float(np.dot(a, b) / (na * nb))


def pattern_metrics(
    df: pd.DataFrame,
    start: int,
    end: int,
    forward: int,
) -> dict:
    """한 패턴 발생 후 미래 성과를 측정."""

    entry = float(df["Close"].iloc[end - 1])

    future = df.iloc[end : min(end + forward, len(df))]
    if future.empty or entry <= 0:
        return {
            "forward_return": np.nan,
            "max_up": np.nan,
            "max_down": np.nan,
            "duration_to_high": np.nan,
        }

    high_ret = future["High"] / entry - 1
    low_ret = future["Low"] / entry - 1
    close_ret = future["Close"] / entry - 1

    max_up = float(high_ret.max())
    max_down = float(low_ret.min())
    forward_return = float(close_ret.iloc[-1])

    high_pos = int(high_ret.argmax()) + 1

    return {
        "forward_return": forward_return,
        "max_up": max_up,
        "max_down": max_down,
        "duration_to_high": high_pos,
    }


def choose_pattern_label(window: pd.DataFrame) -> str:
    """
    숫자 클러스터를 사람이 읽을 수 있는 구조명으로 변환.
    이름 자체가 Driver를 미리 결정하는 것은 아니며,
    실제 과거 데이터의 형태를 요약하는 용도.
    """

    close = window["Close"]
    start = float(close.iloc[0])
    end = float(close.iloc[-1])

    total = end / start - 1

    first_half = close.iloc[: len(close) // 2]
    second_half = close.iloc[len(close) // 2 :]

    first_move = first_half.iloc[-1] / first_half.iloc[0] - 1
    second_move = second_half.iloc[-1] / second_half.iloc[0] - 1

    peak = close.max()
    trough = close.min()

    peak_pos = int(close.argmax())
    trough_pos = int(close.argmin())

    peak_from_start = peak / start - 1
    trough_from_start = trough / start - 1

    last = window.iloc[-1]

    ma20 = float(last.get("ma20_gap", 0))
    ma50 = float(last.get("ma50_gap", 0))
    slope20 = float(last.get("slope20", 0))

    # 급락 후 회복
    if trough_pos < len(close) * 0.75 and trough_from_start < -0.18 and total > -0.10:
        return "급락→저점형성→회복"

    # 급등 후 큰 조정 뒤 재상승
    if peak_from_start > 0.20 and second_move > 0.05 and trough_pos > peak_pos:
        return "급등→조정→재상승"

    # 압축 후 확장
    if (
        float(window["vol20"].iloc[-1]) > 0
        and float(window["vol20"].iloc[-1]) > float(window["vol20"].median()) * 1.15
        and total > 0.10
    ):
        return "압축→돌파→확장"

    # 추세 상승
    if total > 0.12 and ma20 > 0 and ma50 > -0.02 and slope20 > 0:
        return "추세상승→확장"

    # 추세 하락
    if total < -0.12 and ma20 < 0 and ma50 < 0 and slope20 < 0:
        return "추세하락→반등시도"

    # 박스/반복 회귀
    if abs(total) < 0.10:
        return "박스→반복회귀"

    return "추세전환형"


def cluster_patterns(events, threshold=0.90):
    """
    고정 K값 없이 greedy clustering.
    각 패턴은 가장 가까운 대표 centroid에 배정.
    """

    clusters = []

    for event in events:
        sig = event["signature"]

        best_idx = None
        best_sim = -1.0

        for i, cluster in enumerate(clusters):
            sim = cosine_similarity(sig, cluster["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx is not None and best_sim >= threshold:
            cluster = clusters[best_idx]
            cluster["events"].append(event)

            # centroid를 단순 평균 후 재정규화
            arr = np.mean([x["signature"] for x in cluster["events"]], axis=0)
            norm = np.linalg.norm(arr)
            cluster["centroid"] = arr / norm if norm > 1e-12 else arr
        else:
            clusters.append(
                {
                    "centroid": sig.copy(),
                    "events": [event],
                }
            )

    return clusters


def select_non_overlapping_events(df: pd.DataFrame):
    """
    모든 5일 단위 샘플을 쓰되,
    비슷한 시점의 이벤트가 과도하게 중복되는 것을 방지.
    """

    events = []

    start_min = max(0, MIN_HISTORY - WINDOW)

    for end in range(start_min + WINDOW, len(df) - 2):
        if end < WINDOW:
            continue

        window = df.iloc[end - WINDOW : end].copy()

        if window["Close"].isna().any():
            continue

        sig = make_signature(window)

        if not np.isfinite(sig).all():
            continue

        metrics = pattern_metrics(df, end - WINDOW, end, FORWARD)

        event = {
            "end_idx": end,
            "date": df.index[end - 1],
            "signature": sig,
            **metrics,
            "label": choose_pattern_label(window),
        }

        events.append(event)

    # 날짜 순으로 greedy thinning
    selected = []
    last_idx = -10**9

    for e in events:
        if e["end_idx"] - last_idx >= EVENT_SPACING:
            selected.append(e)
            last_idx = e["end_idx"]

    return selected


def cycle_statistics(events) -> tuple:
    if len(events) < 2:
        return np.nan, np.nan

    dates = sorted([pd.Timestamp(e["date"]) for e in events])
    gaps = []

    for a, b in zip(dates[:-1], dates[1:]):
        gaps.append((b - a).days)

    if not gaps:
        return np.nan, np.nan

    return float(np.median(gaps)), float(np.mean(gaps))


def classify_state(similarity: float, success_rate: float, occurrences: int) -> str:
    if occurrences < 2:
        return "INACTIVE"

    if similarity >= CURRENT_ACTIVE_SIM:
        return "ACTIVE"

    if similarity >= CURRENT_WATCH_SIM:
        return "WATCH"

    return "INACTIVE"


def current_phase(df: pd.DataFrame) -> str:
    """현재 시장 구조를 간단한 단계로 표현."""

    if len(df) < 60:
        return "INSUFFICIENT_HISTORY"

    x = df.iloc[-1]

    close = float(x["Close"])
    ma20 = float(x["ma20"])
    ma50 = float(x["ma50"])

    slope20 = float(x.get("slope20", 0))
    vol_ratio = float(x.get("vol_ratio", 1))

    high20 = float(df["High"].tail(20).max())
    low20 = float(df["Low"].tail(20).min())

    range20 = max(high20 - low20, 1e-9)

    if close > high20 * 0.995 and vol_ratio >= 1.25:
        return "BREAKOUT"

    if close < low20 * 1.005:
        return "BREAKDOWN"

    if close > ma20 > ma50 and slope20 > 0:
        return "UPTREND"

    if close < ma20 < ma50 and slope20 < 0:
        return "DOWNTREND"

    if range20 / close < 0.12:
        return "COMPRESSION"

    return "TRANSITION"


def historical_levels(cluster_events, current_price):
    """
    실제 과거 Driver 발생 당시 가격 움직임을 이용한
    확률적 목표/무효화 범위.

    고정 숫자를 강제로 넣지 않는다.
    """

    ups = pd.Series(
        [e["max_up"] for e in cluster_events],
        dtype=float,
    ).dropna()

    downs = pd.Series(
        [e["max_down"] for e in cluster_events],
        dtype=float,
    ).dropna()

    if ups.empty:
        target1 = current_price * 1.05
        target2 = current_price * 1.08
    else:
        # 중앙값보다 약간 보수적인 50% 지점
        up50 = float(ups.quantile(0.50))
        up75 = float(ups.quantile(0.75))

        target1 = current_price * (1 + max(0.03, up50 * 0.60))
        target2 = current_price * (1 + max(0.05, up75 * 0.70))

    if downs.empty:
        invalidation = current_price * 0.95
    else:
        down25 = float(downs.quantile(0.25))
        invalidation = current_price * (1 + min(-0.03, down25 * 0.75))

    return target1, target2, invalidation


def analyze_symbol(symbol: str):
    symbol = symbol.upper().strip()

    print(f"\n===== DRIVER: {symbol} =====")

    raw = download_price(symbol)

    if raw.empty:
        print(f"[SKIP] {symbol}: no price data")
        return [], [], []

    df = add_features(raw)

    usable = df.dropna(subset=["Close"]).copy()

    if len(usable) < MIN_HISTORY:
        print(
            f"[SKIP] {symbol}: history {len(usable)} < minimum {MIN_HISTORY}"
        )
        return [], [], []

    events = select_non_overlapping_events(usable)

    if len(events) < 3:
        print(f"[SKIP] {symbol}: not enough pattern events ({len(events)})")
        return [], [], []

    clusters = cluster_patterns(events, threshold=CLUSTER_SIM)

    # 발생 횟수가 많은 Driver를 우선.
    # 단, 너무 작은 cluster는 제외.
    valid_clusters = [
        c for c in clusters
        if len(c["events"]) >= 2
    ]

    valid_clusters.sort(
        key=lambda c: (
            len(c["events"]),
            np.nanmedian(
                [
                    x["forward_return"]
                    for x in c["events"]
                    if np.isfinite(x["forward_return"])
                ]
            )
            if any(
                np.isfinite(x["forward_return"])
                for x in c["events"]
            )
            else -999,
        ),
        reverse=True,
    )

    # 실제 데이터가 1개 Driver만 안정적으로 보여주면 1개만 출력.
    selected = valid_clusters[:MAX_DRIVERS]

    current_window = usable.iloc[-WINDOW:].copy()
    current_sig = make_signature(current_window)
    current_price = float(usable["Close"].iloc[-1])
    phase = current_phase(usable)

    profiles = []
    history_rows = []

    for rank, cluster in enumerate(selected, start=1):
        cluster_events = cluster["events"]

        sims = [
            cosine_similarity(current_sig, e["signature"])
            for e in cluster_events
        ]

        current_similarity = max(sims) if sims else 0.0

        forward_returns = pd.Series(
            [e["forward_return"] for e in cluster_events],
            dtype=float,
        ).dropna()

        max_ups = pd.Series(
            [e["max_up"] for e in cluster_events],
            dtype=float,
        ).dropna()

        max_downs = pd.Series(
            [e["max_down"] for e in cluster_events],
            dtype=float,
        ).dropna()

        # 성공 정의:
        # 40거래일 내 +5% 이상 상승이 한 번이라도 있었는가.
        success_flags = [
            bool(np.isfinite(e["max_up"]) and e["max_up"] >= 0.05)
            for e in cluster_events
        ]

        success_count = int(sum(success_flags))
        occurrence_count = len(cluster_events)
        success_rate = success_count / occurrence_count

        median_cycle_days, mean_cycle_days = cycle_statistics(
            cluster_events
        )

        target1, target2, invalidation = historical_levels(
            cluster_events,
            current_price,
        )

        state = classify_state(
            current_similarity,
            success_rate,
            occurrence_count,
        )

        labels = [e["label"] for e in cluster_events]
        label = pd.Series(labels).mode().iloc[0]

        dates = sorted([pd.Timestamp(e["date"]) for e in cluster_events])

        # 현재 상태와 가장 가까운 과거 발생일
        nearest_event = max(
            cluster_events,
            key=lambda e: cosine_similarity(
                current_sig,
                e["signature"],
            ),
        )

        nearest_similarity = cosine_similarity(
            current_sig,
            nearest_event["signature"],
        )

        profiles.append(
            {
                "symbol": symbol,
                "driver_rank": rank,
                "driver_name": f"DRIVER_{rank}",
                "pattern": label,
                "state": state,
                "phase": phase,
                "current_price": round(current_price, 4),
                "current_similarity_pct": round(current_similarity * 100, 2),
                "occurrences": occurrence_count,
                "success_count": success_count,
                "success_rate_pct": round(success_rate * 100, 2),
                "avg_forward_return_pct": round(
                    forward_returns.mean() * 100, 2
                    if not forward_returns.empty
                    else np.nan,
                    2,
                ),
                "median_forward_return_pct": round(
                    forward_returns.median() * 100
                    if not forward_returns.empty
                    else np.nan,
                    2,
                ),
                "median_max_up_pct": round(
                    max_ups.median() * 100
                    if not max_ups.empty
                    else np.nan,
                    2,
                ),
                "median_max_down_pct": round(
                    max_downs.median() * 100
                    if not max_downs.empty
                    else np.nan,
                    2,
                ),
                "median_cycle_days": round(median_cycle_days, 1)
                if np.isfinite(median_cycle_days)
                else np.nan,
                "mean_cycle_days": round(mean_cycle_days, 1)
                if np.isfinite(mean_cycle_days)
                else np.nan,
                "target1": round(target1, 4),
                "target2": round(target2, 4),
                "invalidation": round(invalidation, 4),
                "first_seen": dates[0].date().isoformat(),
                "last_seen": dates[-1].date().isoformat(),
                "nearest_historical_similarity_pct": round(
                    nearest_similarity * 100,
                    2,
                ),
            }
        )

        for e in cluster_events:
            history_rows.append(
                {
                    "symbol": symbol,
                    "driver_rank": rank,
                    "driver_name": f"DRIVER_{rank}",
                    "pattern": label,
                    "event_date": pd.Timestamp(e["date"]).date().isoformat(),
                    "forward_return_pct": round(
                        e["forward_return"] * 100
                        if np.isfinite(e["forward_return"])
                        else np.nan,
                        2,
                    ),
                    "max_up_pct": round(
                        e["max_up"] * 100
                        if np.isfinite(e["max_up"])
                        else np.nan,
                        2,
                    ),
                    "max_down_pct": round(
                        e["max_down"] * 100
                        if np.isfinite(e["max_down"])
                        else np.nan,
                        2,
                    ),
                    "duration_to_high_days": e["duration_to_high"],
                    "label_at_event": e["label"],
                }
            )

    # 현재 가장 가까운 Driver를 위로
    profiles.sort(
        key=lambda x: (
            {"ACTIVE": 0, "WATCH": 1, "INACTIVE": 2}.get(
                x["state"], 9
            ),
            -x["current_similarity_pct"],
        )
    )

    # rank는 최종 표시 순서에 맞춰 재부여
    for i, p in enumerate(profiles, start=1):
        old_rank = p["driver_rank"]
        p["driver_rank"] = i
        p["driver_name"] = f"DRIVER_{i}"

        for h in history_rows:
            if h["driver_rank"] == old_rank:
                h["driver_rank"] = i
                h["driver_name"] = f"DRIVER_{i}"

    current_rows = []

    if profiles:
        top = profiles[0]

        current_rows.append(
            {
                "symbol": symbol,
                "as_of": usable.index[-1].date().isoformat(),
                "current_price": top["current_price"],
                "active_driver": top["driver_name"],
                "pattern": top["pattern"],
                "state": top["state"],
                "phase": top["phase"],
                "similarity_pct": top["current_similarity_pct"],
                "success_rate_pct": top["success_rate_pct"],
                "target1": top["target1"],
                "target2": top["target2"],
                "invalidation": top["invalidation"],
                "occurrences": top["occurrences"],
                "median_cycle_days": top["median_cycle_days"],
            }
        )

    print(
        f"[OK] {symbol}: "
        f"drivers={len(profiles)} "
        f"phase={phase} "
        f"price={current_price:.2f}"
    )

    for p in profiles:
        print(
            f"  {p['driver_name']} | "
            f"{p['pattern']} | "
            f"{p['state']} | "
            f"sim={p['current_similarity_pct']:.1f}% | "
            f"success={p['success_rate_pct']:.1f}% | "
            f"T1={p['target1']:.2f} | "
            f"T2={p['target2']:.2f} | "
            f"INV={p['invalidation']:.2f}"
        )

    return profiles, current_rows, history_rows


def load_symbols_from_project():
    # 1) CLI
    cli = [
        x.strip().upper()
        for x in sys.argv[1:]
        if x.strip()
    ]
    if cli:
        return list(dict.fromkeys(cli))

    # 2) 환경변수
    env = os.getenv("DRIVER_SYMBOLS", "")
    if env.strip():
        return list(
            dict.fromkeys(
                x.strip().upper()
                for x in env.split(",")
                if x.strip()
            )
        )

    # 3) 기존 selected_symbols.py
    try:
        from selected_symbols import SELECTED_SYMBOLS

        symbols = [
            str(x).strip().upper()
            for x in SELECTED_SYMBOLS
            if str(x).strip()
        ]

        if symbols:
            return list(dict.fromkeys(symbols))
    except Exception:
        pass

    # 4) 최후 fallback
    # 프로젝트가 별도의 symbol 파일을 아직 연결하지 않은 경우
    fallback = [
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

    return fallback


def save_outputs(profiles, current_rows, history_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    profile_df = pd.DataFrame(profiles)
    current_df = pd.DataFrame(current_rows)
    history_df = pd.DataFrame(history_rows)

    # 기존 파일이 있다면 현재 실행 결과로 교체.
    # Driver는 매 실행마다 최신 전체 히스토리를 다시 계산하는 구조.
    profile_df.to_csv(PROFILE_FILE, index=False, encoding="utf-8-sig")
    current_df.to_csv(CURRENT_FILE, index=False, encoding="utf-8-sig")
    history_df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")

    print("\n===== OUTPUT =====")
    print(f"profile : {PROFILE_FILE}")
    print(f"current : {CURRENT_FILE}")
    print(f"history : {HISTORY_FILE}")
    print(
        f"rows    : "
        f"profile={len(profile_df)}, "
        f"current={len(current_df)}, "
        f"history={len(history_df)}"
    )


def run(symbols):
    all_profiles = []
    all_current = []
    all_history = []

    symbols = list(dict.fromkeys(
        str(s).strip().upper()
        for s in symbols
        if str(s).strip()
    ))

    print("==============================================")
    print(" UNIVERSAL DRIVER ENGINE")
    print("==============================================")
    print(f"symbols       : {len(symbols)}")
    print(f"lookback      : {LOOKBACK_PERIOD}")
    print(f"window        : {WINDOW}")
    print(f"forward       : {FORWARD}")
    print(f"min_history   : {MIN_HISTORY}")
    print(f"max_drivers   : {MAX_DRIVERS}")
    print(f"cluster_sim   : {CLUSTER_SIM}")
    print("==============================================")

    for symbol in symbols:
        try:
            profiles, current, history = analyze_symbol(symbol)

            all_profiles.extend(profiles)
            all_current.extend(current)
            all_history.extend(history)

        except Exception as e:
            print(f"[ERROR] {symbol}: {type(e).__name__}: {e}")

    save_outputs(
        all_profiles,
        all_current,
        all_history,
    )

    # 화면에서 빠르게 검증할 수 있는 요약
    if all_current:
        print("\n===== CURRENT DRIVER SUMMARY =====")
        print(
            pd.DataFrame(all_current).to_string(index=False)
        )
    else:
        print("\n[WARN] No current driver detected.")


if __name__ == "__main__":
    symbols = load_symbols_from_project()
    run(symbols)

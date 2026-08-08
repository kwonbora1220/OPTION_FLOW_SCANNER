import pandas as pd


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

    df["mid_price"] = (
        df["bid"] + df["ask"]
    ) / 2

    df["mid_price"] = df["mid_price"].where(
        df["mid_price"] > 0,
        df["lastPrice"]
    )

    df["premium_flow"] = (
        df["volume"]
        * df["mid_price"]
        * 100
    )

    return df


def calculate_dte(df, min_dte=0, max_dte=180):
    df = df.copy()

    today = pd.Timestamp.now().normalize()

    df["expiration_date"] = pd.to_datetime(
        df["expiration"],
        errors="coerce"
    )

    df["DTE"] = (
        df["expiration_date"] - today
    ).dt.days

    df = df[
        (df["DTE"] >= min_dte)
        & (df["DTE"] <= max_dte)
    ]

    return df


def calculate_basic_flow(df):
    calls = df[
        df["option_type"] == "CALL"
    ].copy()

    puts = df[
        df["option_type"] == "PUT"
    ].copy()

    if calls.empty and puts.empty:
        return {
            "flow_score": 50.0,
            "call_premium": 0,
            "put_premium": 0,
            "call_volume": 0,
            "put_volume": 0,
            "call_oi": 0,
            "put_oi": 0
        }

    call_premium = calls["premium_flow"].sum()
    put_premium = puts["premium_flow"].sum()

    call_volume = calls["volume"].sum()
    put_volume = puts["volume"].sum()

    call_oi = calls["openInterest"].sum()
    put_oi = puts["openInterest"].sum()

    total_premium = call_premium + put_premium
    total_volume = call_volume + put_volume
    total_oi = call_oi + put_oi

    call_premium_ratio = (
        call_premium / total_premium
        if total_premium > 0 else 0.5
    )

    call_volume_ratio = (
        call_volume / total_volume
        if total_volume > 0 else 0.5
    )

    call_oi_ratio = (
        call_oi / total_oi
        if total_oi > 0 else 0.5
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

    score = max(0, min(100, score))

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


def dte_quality(df):
    score = 0
    reasons = []

    ranges = [
        (0, 7, 2, "0~7DTE"),
        (8, 30, 5, "8~30DTE"),
        (31, 60, 5, "31~60DTE"),
        (61, 180, 5, "61~180DTE")
    ]

    for low, high, points, label in ranges:

        subset = df[
            (df["DTE"] >= low)
            & (df["DTE"] <= high)
        ]

        if not subset.empty:
            score += points
            reasons.append(f"{label} 구조")

    return score, reasons


def classify(score):
    if score >= 70:
        return "🟢 오늘 진입 후보"

    if score >= 40:
        return "🟡 감시"

    return "🔴 회피"

import os
import sys
import time
import math
import re
import glob
import requests
import yfinance as yf
import pandas as pd

from datetime import datetime, date
from zoneinfo import ZoneInfo

def get_us_market_date():

    return datetime.now(
        ZoneInfo("America/New_York")
    ).date()


# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    ""
)

CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID",
    ""
)

MAX_DTE = 180

RISK_FREE_RATE = 0.04

CONTRACT_MULTIPLIER = 100

WALL_RANGE_PCT = 30

ATM_RANGE_PCT = 5

ENTRY_RESISTANCE_PCT = 3.0

HOLDING_SUPPORT_BUFFER_PCT = 2.0


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "03_RESULTS",
    "daily"
)

if os.path.basename(BASE_DIR).upper() == "02_PROGRAM":

    RESULT_DIR = os.path.abspath(
        os.path.join(
            BASE_DIR,
            "..",
            "03_RESULTS",
            "daily"
        )
    )

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# OI HISTORY
# ============================================================

OI_HISTORY_DIR = os.path.join(
    RESULT_DIR,
    "oi_history"
)

os.makedirs(
    OI_HISTORY_DIR,
    exist_ok=True
)


# ============================================================
# FORMAT
# ============================================================

def format_money(x):

    try:
        x = float(x)
    except Exception:
        return "$0"

    sign = ""

    if x < 0:
        sign = "-"
        x = abs(x)

    if x >= 1_000_000_000:
        return f"{sign}${x / 1_000_000_000:.2f}B"

    if x >= 1_000_000:
        return f"{sign}${x / 1_000_000:.2f}M"

    if x >= 1_000:
        return f"{sign}${x / 1_000:.1f}K"

    return f"{sign}${x:.0f}"


def format_signed_money(x):

    try:
        x = float(x)
    except Exception:
        return "$0"

    if x > 0:
        return f"+{format_money(x)}"

    return format_money(x)


# ============================================================
# MARKET PRICE
# ============================================================

def get_market_price_context(ticker):

    for attempt in range(1, 4):

        try:

            print(
                f"💰 {ticker} 가격 조회 ({attempt}/3)"
            )

            t = yf.Ticker(ticker)

            hist = t.history(
                period="5d",
                interval="1d",
                auto_adjust=False
            )

            regular_close = None

            if not hist.empty:

                close = hist["Close"].dropna()

                if not close.empty:
                    regular_close = float(
                        close.iloc[-1]
                    )

            if regular_close is None:
                raise ValueError(
                    "정규장 종가를 가져오지 못했습니다."
                )

            market_state = None

            try:

                info = t.get_info()

                market_state = info.get(
                    "marketState"
                )

            except Exception:

                market_state = None

            after_hours_price = None

            try:

                intraday = t.history(
                    period="1d",
                    interval="1m",
                    prepost=True,
                    auto_adjust=False
                )

                if not intraday.empty:

                    close_1m = (
                        intraday["Close"]
                        .dropna()
                    )

                    if not close_1m.empty:

                        latest_price = float(
                            close_1m.iloc[-1]
                        )

                        if market_state in {
                            "POST",
                            "POSTPOST"
                        }:

                            if abs(
                                latest_price
                                - regular_close
                            ) > 0.000001:

                                after_hours_price = (
                                    latest_price
                                )

            except Exception as e:

                print(
                    f"⚠️ 시간외 가격 조회 실패: {e}"
                )

            after_hours_change_pct = None

            if (
                after_hours_price is not None
                and regular_close != 0
            ):

                after_hours_change_pct = (
                    (
                        after_hours_price
                        / regular_close
                    ) - 1
                ) * 100

            print(
                f"💰 {ticker} 정규장 종가: "
                f"${regular_close:.2f}"
            )

            if after_hours_price is not None:

                print(
                    f"🌙 {ticker} 시간외 현재가: "
                    f"${after_hours_price:.2f}"
                    + (
                        f" ({after_hours_change_pct:+.2f}%)"
                        if after_hours_change_pct is not None
                        else ""
                    )
                )

            else:

                print(
                    f"🌙 {ticker} 시간외 현재가: N/A"
                )

            return {
                "regular_close": regular_close,
                "after_hours_price": after_hours_price,
                "after_hours_change_pct": after_hours_change_pct,
                "market_state": market_state,
                "option_analysis_price": regular_close
            }

        except Exception as e:

            print(
                f"⚠️ 가격 조회 실패: {e}"
            )

        if attempt < 3:
            time.sleep(2)

    return None


def get_current_price(ticker):

    context = get_market_price_context(
        ticker
    )

    if context is None:
        return None

    return context[
        "option_analysis_price"
    ]


# ============================================================
# OI SNAPSHOT PATH
# ============================================================

def get_oi_snapshot_path(
    ticker,
    snapshot_date=None
):

    if snapshot_date is None:
        snapshot_date = get_us_market_date()

    return os.path.join(
        OI_HISTORY_DIR,
        f"{ticker.upper().strip()}_OI_"
        f"{snapshot_date.strftime('%Y%m%d')}.csv"
    )


# ============================================================
# BUILD OI SNAPSHOT
# ============================================================

def build_oi_snapshot(df):

    required = [
        "expiration",
        "DTE",
        "strike",
        "option_type",
        "openInterest"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"OI snapshot 필수 컬럼 누락: {missing}"
        )

    snapshot = df[
        required
    ].copy()

    snapshot["expiration"] = (
        snapshot["expiration"]
        .astype(str)
        .str.strip()
    )

    snapshot["option_type"] = (
        snapshot["option_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    snapshot["strike"] = pd.to_numeric(
        snapshot["strike"],
        errors="coerce"
    )

    snapshot["DTE"] = pd.to_numeric(
        snapshot["DTE"],
        errors="coerce"
    )

    snapshot["openInterest"] = pd.to_numeric(
        snapshot["openInterest"],
        errors="coerce"
    ).fillna(0)

    snapshot["openInterest"] = (
        snapshot["openInterest"]
        .clip(lower=0)
    )

    snapshot = snapshot.dropna(
        subset=[
            "expiration",
            "strike"
        ]
    )

    snapshot = snapshot[
        snapshot["option_type"].isin(
            ["CALL", "PUT"]
        )
    ]

    keys = [
        "expiration",
        "strike",
        "option_type"
    ]

    snapshot = (
        snapshot
        .groupby(keys, as_index=False)
        .agg(
            DTE=("DTE", "max"),
            openInterest=(
                "openInterest",
                "max"
            )
        )
    )

    return snapshot


# ============================================================
# FIND PREVIOUS OI SNAPSHOT
# ============================================================

def find_previous_oi_snapshot(
    ticker,
    current_date=None
):

    if current_date is None:
        current_date = get_us_market_date()

    pattern = os.path.join(
        OI_HISTORY_DIR,
        f"{ticker.upper().strip()}_OI_*.csv"
    )

    candidates = []

    for path in glob.glob(pattern):

        filename = os.path.basename(path)

        match = re.search(
            r"_OI_(\d{8})\.csv$",
            filename
        )

        if not match:
            continue

        try:

            snapshot_date = datetime.strptime(
                match.group(1),
                "%Y%m%d"
            ).date()

        except Exception:

            continue

        if snapshot_date < current_date:

            candidates.append(
                (
                    snapshot_date,
                    path
                )
            )

    if not candidates:

        return None, None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return candidates[0]


# ============================================================
# CALCULATE OI DELTA
# ============================================================

def calculate_oi_delta(
    current_df,
    ticker
):

    df = current_df.copy()

    df["previous_openInterest"] = 0.0
    df["oi_delta"] = 0.0
    df["oi_delta_available"] = False
    df["oi_delta_status"] = (
        "NO_PREVIOUS_SNAPSHOT"
    )

    previous_date, previous_path = (
        find_previous_oi_snapshot(
            ticker
        )
    )

    if previous_path is None:

        print(
            "⚠️ 이전 OI snapshot 없음"
        )

        return df, None

    print(
        f"📂 이전 OI snapshot: "
        f"{previous_path}"
    )

    try:

        previous = pd.read_csv(
            previous_path,
            encoding="utf-8-sig"
        )

        previous = build_oi_snapshot(
            previous
        )

        current = build_oi_snapshot(
            df
        )

        keys = [
            "expiration",
            "strike",
            "option_type"
        ]

        previous = previous[
            keys + ["openInterest"]
        ].rename(
            columns={
                "openInterest":
                "previous_openInterest"
            }
        )

        current = current[
            keys + ["openInterest"]
        ]

        # ----------------------------------------------------
        # UNION MERGE
        # ----------------------------------------------------

        merged = pd.merge(
            current,
            previous,
            on=keys,
            how="outer"
        )

        merged["openInterest"] = pd.to_numeric(
            merged["openInterest"],
            errors="coerce"
        ).fillna(0)

        merged["previous_openInterest"] = pd.to_numeric(
            merged["previous_openInterest"],
            errors="coerce"
        ).fillna(0)

        merged["oi_delta"] = (
            merged["openInterest"]
            - merged["previous_openInterest"]
        )

        # ----------------------------------------------------
        # CURRENT DATA + PREVIOUS OI
        # ----------------------------------------------------


        current_df = df.copy()

        # calculate_oi_delta() 시작부에서 초기화한
        # helper column이 merge 대상과 충돌하지 않도록 제거한다.
        current_df = current_df.drop(
            columns=[
                "previous_openInterest",
                "oi_delta"
            ],
            errors="ignore"
        )

        current_df["expiration"] = (
            current_df["expiration"]
            .astype(str)
            .str.strip()
        )


        
        current_df["option_type"] = (
            current_df["option_type"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        current_df["strike"] = pd.to_numeric(
            current_df["strike"],
            errors="coerce"
        )

        current_df = current_df.merge(
            merged[
                keys
                + [
                    "previous_openInterest",
                    "oi_delta"
                ]
            ],
            on=keys,
            how="left"
        )

        current_df["previous_openInterest"] = (
            pd.to_numeric(
                current_df[
                    "previous_openInterest"
                ],
                errors="coerce"
            )
            .fillna(0)
        )

        current_df["oi_delta"] = (
            pd.to_numeric(
                current_df["oi_delta"],
                errors="coerce"
            )
            .fillna(0)
        )

        current_df[
            "oi_delta_available"
        ] = True

        current_df[
            "oi_delta_status"
        ] = (
            f"COMPARED_WITH_"
            f"{previous_date.strftime('%Y%m%d')}"
        )

        total_delta = float(
            current_df["oi_delta"].sum()
        )

        positive = int(
            (
                current_df["oi_delta"] > 0
            ).sum()
        )

        negative = int(
            (
                current_df["oi_delta"] < 0
            ).sum()
        )

        print(
            f"📈 OI 비교 완료: "
            f"{previous_date.strftime('%Y-%m-%d')}"
        )

        print(
            f"📈 OI 증가 계약: {positive:,}"
        )

        print(
            f"📉 OI 감소 계약: {negative:,}"
        )

        print(
            f"📊 전체 OI Δ: {total_delta:+,.0f}"
        )

        if abs(total_delta) < 0.000001:

            print(
                "ℹ️ 전체 OI 변화: 0"
            )

            current_df[
                "oi_delta_status"
            ] = (
                f"NO_OI_CHANGE_"
                f"{previous_date.strftime('%Y%m%d')}"
            )

        return (
            current_df,
            previous_date
        )

    except Exception as e:

        print(
            f"⚠️ OI snapshot 비교 실패: {e}"
        )

        return df, None


# ============================================================
# SAVE OI SNAPSHOT
# ============================================================

def save_oi_snapshot(
    ticker,
    df
):

    filename = get_oi_snapshot_path(
        ticker
    )

    snapshot = build_oi_snapshot(
        df
    )

    snapshot.to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"💾 OI snapshot 저장: "
        f"{filename}"
    )

    print(
        f"   계약 수: {len(snapshot):,}"
    )

    print(
        f"   총 OI: "
        f"{snapshot['openInterest'].sum():,.0f}"
    )

    return filename


# ============================================================
# TOP OI DELTA
# ============================================================

def find_top_oi_delta(
    df,
    current_price,
    top_n=5
):

    empty = {
        "call_increase": pd.DataFrame(),
        "call_decrease": pd.DataFrame(),
        "put_increase": pd.DataFrame(),
        "put_decrease": pd.DataFrame()
    }

    if "oi_delta" not in df.columns:
        return empty

    if "oi_delta_available" not in df.columns:
        return empty

    active = df[
        df["oi_delta_available"]
        & (
            df["oi_delta"] != 0
        )
    ].copy()

    if active.empty:
        return empty

    active["distance_pct"] = (
        abs(
            active["strike"]
            - current_price
        )
        / current_price
        * 100
    )

    active = active[
        active["distance_pct"]
        <= WALL_RANGE_PCT
    ]

    if active.empty:
        return empty

    calls = active[
        active["option_type"] == "CALL"
    ]

    puts = active[
        active["option_type"] == "PUT"
    ]

    return {

        "call_increase":
        calls[
            calls["oi_delta"] > 0
        ]
        .sort_values(
            [
                "oi_delta",
                "volume"
            ],
            ascending=False
        )
        .head(top_n),

        "call_decrease":
        calls[
            calls["oi_delta"] < 0
        ]
        .sort_values(
            [
                "oi_delta",
                "volume"
            ],
            ascending=True
        )
        .head(top_n),

        "put_increase":
        puts[
            puts["oi_delta"] > 0
        ]
        .sort_values(
            [
                "oi_delta",
                "volume"
            ],
            ascending=False
        )
        .head(top_n),

        "put_decrease":
        puts[
            puts["oi_delta"] < 0
        ]
        .sort_values(
            [
                "oi_delta",
                "volume"
            ],
            ascending=True
        )
        .head(top_n)
    }


# ============================================================
# OPTION DATA
# ============================================================

def get_option_data(ticker):

    t = yf.Ticker(ticker)

    expirations = t.options

    if not expirations:

        raise Exception(
            "옵션 만기 데이터가 없습니다."
        )

    today = date.today()

    valid_expirations = []

    for exp in expirations:

        try:

            exp_date = datetime.strptime(
                exp,
                "%Y-%m-%d"
            ).date()

            dte = (
                exp_date - today
            ).days

            if 0 <= dte <= MAX_DTE:

                valid_expirations.append(
                    (
                        exp,
                        dte
                    )
                )

        except Exception:

            continue

    print(
        f"📅 전체 만기: "
        f"{len(expirations)}개"
    )

    print(
        f"📅 0~180 DTE 만기: "
        f"{len(valid_expirations)}개"
    )

    if not valid_expirations:

        raise Exception(
            "0~180 DTE 옵션 만기가 없습니다."
        )

    rows = []

    for exp, dte in valid_expirations:

        print(
            f"   수집: {exp} | DTE {dte}"
        )

        try:

            chain = t.option_chain(
                exp
            )

            calls = chain.calls.copy()
            puts = chain.puts.copy()

            calls["option_type"] = "CALL"
            puts["option_type"] = "PUT"

            calls["expiration"] = exp
            puts["expiration"] = exp

            calls["DTE"] = dte
            puts["DTE"] = dte

            rows.append(calls)
            rows.append(puts)

        except Exception as e:

            print(
                f"⚠️ {exp} 오류: {e}"
            )

    if not rows:

        raise Exception(
            "옵션 데이터를 가져오지 못했습니다."
        )

    return pd.concat(
        rows,
        ignore_index=True
    )


# ============================================================
# NORMALIZE
# ============================================================

def normalize_option_data(df):

    df = df.copy()

    numeric_cols = [
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "volume",
        "openInterest",
        "impliedVolatility",
        "DTE"
    ]

    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    df["volume"] = (
        df["volume"]
        .clip(lower=0)
    )

    df["openInterest"] = (
        df["openInterest"]
        .clip(lower=0)
    )

    return df


# ============================================================
# NORMAL DISTRIBUTION
# ============================================================

def norm_pdf(x):

    return (
        math.exp(
            -0.5 * x * x
        )
        / math.sqrt(
            2 * math.pi
        )
    )


def norm_cdf(x):

    return (
        0.5
        * (
            1
            + math.erf(
                x / math.sqrt(2)
            )
        )
    )


# ============================================================
# IV
# ============================================================

def safe_iv(iv):

    try:

        iv = float(iv)

        if iv <= 0:
            return 0.0001

        if iv > 5:
            iv = iv / 100

        return max(
            0.0001,
            min(iv, 5.0)
        )

    except Exception:

        return 0.0001


# ============================================================
# BLACK-SCHOLES GREEKS
# ============================================================

def calculate_greeks(
    spot,
    strike,
    iv,
    dte,
    option_type
):

    try:

        spot = float(spot)
        strike = float(strike)
        iv = safe_iv(iv)
        dte = float(dte)

        if spot <= 0:
            raise ValueError

        if strike <= 0:
            raise ValueError

        T = max(
            dte / 365.0,
            1 / 365.0
        )

        r = RISK_FREE_RATE

        sqrt_T = math.sqrt(T)

        d1 = (
            math.log(
                spot / strike
            )
            + (
                r
                + 0.5 * iv * iv
            ) * T
        ) / (
            iv * sqrt_T
        )

        d2 = (
            d1
            - iv * sqrt_T
        )

        pdf = norm_pdf(d1)

        if option_type == "CALL":

            delta = norm_cdf(d1)

        else:

            delta = (
                norm_cdf(d1) - 1
            )

        gamma = (
            pdf
            / (
                spot
                * iv
                * sqrt_T
            )
        )

        vega = (
            spot
            * pdf
            * sqrt_T
        )

        vanna = (
            -pdf
            * d2
            / iv
        )

        charm_common = (
            pdf
            * (
                2 * r * T
                - d2 * iv * sqrt_T
            )
            / (
                2 * T
            )
        )

        charm = (
            -charm_common
            / (
                spot
                * iv
                * sqrt_T
            )
        )

        if option_type == "CALL":

            theta = (
                -(
                    spot
                    * pdf
                    * iv
                    / (
                        2 * sqrt_T
                    )
                )
                - (
                    r
                    * strike
                    * math.exp(
                        -r * T
                    )
                    * norm_cdf(d2)
                )
            )

        else:

            theta = (
                -(
                    spot
                    * pdf
                    * iv
                    / (
                        2 * sqrt_T
                    )
                )
                + (
                    r
                    * strike
                    * math.exp(
                        -r * T
                    )
                    * norm_cdf(-d2)
                )
            )

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "vanna": vanna,
            "charm": charm,
            "theta": theta,
            "iv": iv
        }

    except Exception:

        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "vanna": 0.0,
            "charm": 0.0,
            "theta": 0.0,
            "iv": safe_iv(iv)
        }


# ============================================================
# OPTION METRICS
# ============================================================

def calculate_option_metrics(
    df,
    current_price
):

    df = normalize_option_data(
        df
    )

    values = {
        "delta": [],
        "gamma": [],
        "vega": [],
        "vanna": [],
        "charm": [],
        "theta": [],
        "IV": []
    }

    for _, row in df.iterrows():

        g = calculate_greeks(
            current_price,
            row["strike"],
            row["impliedVolatility"],
            row["DTE"],
            row["option_type"]
        )

        values["delta"].append(
            g["delta"]
        )

        values["gamma"].append(
            g["gamma"]
        )

        values["vega"].append(
            g["vega"]
        )

        values["vanna"].append(
            g["vanna"]
        )

        values["charm"].append(
            g["charm"]
        )

        values["theta"].append(
            g["theta"]
        )

        values["IV"].append(
            g["iv"]
        )

    df["delta"] = values["delta"]
    df["gamma"] = values["gamma"]
    df["vega"] = values["vega"]
    df["vanna"] = values["vanna"]
    df["charm"] = values["charm"]
    df["theta"] = values["theta"]
    df["IV"] = values["IV"]

    df["traded_premium_proxy"] = (
        df["lastPrice"]
        * df["volume"]
        * CONTRACT_MULTIPLIER
    )

    df["premium_flow"] = (
        df["traded_premium_proxy"]
    )

    df["mid_price"] = (
        df["bid"]
        + df["ask"]
    ) / 2

    df["spread_pct"] = 0.0

    valid_mid = (
        df["mid_price"] > 0
    )

    df.loc[
        valid_mid,
        "spread_pct"
    ] = (
        (
            df.loc[
                valid_mid,
                "ask"
            ]
            -
            df.loc[
                valid_mid,
                "bid"
            ]
        )
        /
        df.loc[
            valid_mid,
            "mid_price"
        ]
        * 100
    )

    df["GEX"] = (
        df["gamma"]
        * df["openInterest"]
        * CONTRACT_MULTIPLIER
        * current_price
        * current_price
        * 0.01
    )

    df.loc[
        df["option_type"] == "PUT",
        "GEX"
    ] *= -1

    df["delta_exposure"] = (
        df["delta"]
        * df["openInterest"]
        * CONTRACT_MULTIPLIER
        * current_price
    )

    df["vanna_exposure"] = (
        df["vanna"]
        * df["openInterest"]
        * CONTRACT_MULTIPLIER
        * current_price
    )

    df["charm_exposure"] = (
        df["charm"]
        * df["openInterest"]
        * CONTRACT_MULTIPLIER
        * current_price
    )

    df["vega_exposure"] = (
        df["vega"]
        * df["openInterest"]
        * CONTRACT_MULTIPLIER
    )

    def estimate_trade_side(row):

        try:

            last = float(
                row["lastPrice"]
            )

            bid = float(
                row["bid"]
            )

            ask = float(
                row["ask"]
            )

            if (
                ask > 0
                and bid > 0
                and ask >= bid
            ):

                if last >= (
                    ask
                    - (
                        ask - bid
                    ) * 0.20
                ):

                    return 1

                if last <= (
                    bid
                    + (
                        ask - bid
                    ) * 0.20
                ):

                    return -1

            return 0

        except Exception:

            return 0

    df["trade_side_proxy"] = df.apply(
        estimate_trade_side,
        axis=1
    )

    df["trade_side"] = (
        df["trade_side_proxy"]
    )

    df["HIRO_proxy"] = (
        df["delta"]
        * df["volume"]
        * CONTRACT_MULTIPLIER
        * current_price
        * df["trade_side_proxy"]
    )

    return df


# ============================================================
# AGGREGATE GREEKS
# ============================================================

def calculate_aggregate_greeks(df):

    result = {}

    valid_iv = df[
        (df["IV"] > 0)
        & (df["IV"] < 5)
        & (df["volume"] > 0)
    ]

    if not valid_iv.empty:

        volume_sum = (
            valid_iv["volume"].sum()
        )

        if volume_sum > 0:

            result["IV"] = (
                (
                    valid_iv["IV"]
                    * valid_iv["volume"]
                ).sum()
                / volume_sum
            )

        else:

            result["IV"] = (
                valid_iv["IV"].mean()
            )

    else:

        result["IV"] = 0.0

    result["Delta"] = float(
        df["delta_exposure"].sum()
    )

    result["GEX"] = float(
        df["GEX"].sum()
    )

    result["Vanna"] = float(
        df["vanna_exposure"].sum()
    )

    result["Charm"] = float(
        df["charm_exposure"].sum()
    )

    result["Vega"] = float(
        df["vega_exposure"].sum()
    )

    result["HIRO"] = float(
        df["HIRO_proxy"].sum()
    )

    return result


# ============================================================
# FLOW SUMMARY
# ============================================================

def calculate_flow_summary(
    df,
    current_price
):

    calls = df[
        df["option_type"] == "CALL"
    ]

    puts = df[
        df["option_type"] == "PUT"
    ]

    call_volume = float(
        calls["volume"].sum()
    )

    put_volume = float(
        puts["volume"].sum()
    )

    call_oi = float(
        calls["openInterest"].sum()
    )

    put_oi = float(
        puts["openInterest"].sum()
    )

    call_premium = float(
        calls[
            "traded_premium_proxy"
        ].sum()
    )

    put_premium = float(
        puts[
            "traded_premium_proxy"
        ].sum()
    )

    total_volume = (
        call_volume
        + put_volume
    )

    total_oi = (
        call_oi
        + put_oi
    )

    total_premium = (
        call_premium
        + put_premium
    )

    call_volume_ratio = (
        call_volume / total_volume
        if total_volume > 0
        else 0.5
    )

    call_oi_ratio = (
        call_oi / total_oi
        if total_oi > 0
        else 0.5
    )

    call_premium_ratio = (
        call_premium / total_premium
        if total_premium > 0
        else 0.5
    )

    atm = df[
        (
            abs(
                df["strike"]
                - current_price
            )
            / current_price
            * 100
        )
        <= ATM_RANGE_PCT
    ]

    atm = atm[
        (atm["IV"] > 0)
        & (atm["volume"] > 0)
    ]

    if not atm.empty:

        volume_sum = atm["volume"].sum()

        atm_iv = (
            (
                atm["IV"]
                * atm["volume"]
            ).sum()
            / volume_sum
            if volume_sum > 0
            else atm["IV"].mean()
        )

    else:

        atm_iv = 0.0

    dte_buckets = {

        "0_7": int(
            (
                (df["DTE"] >= 0)
                &
                (df["DTE"] <= 7)
            ).sum()
        ),

        "8_30": int(
            (
                (df["DTE"] >= 8)
                &
                (df["DTE"] <= 30)
            ).sum()
        ),

        "31_60": int(
            (
                (df["DTE"] >= 31)
                &
                (df["DTE"] <= 60)
            ).sum()
        ),

        "61_180": int(
            (
                (df["DTE"] >= 61)
                &
                (df["DTE"] <= 180)
            ).sum()
        )
    }

    if "oi_delta" in df.columns:

        call_oi_delta = float(
            df[
                df["option_type"] == "CALL"
            ]["oi_delta"].sum()
        )

        put_oi_delta = float(
            df[
                df["option_type"] == "PUT"
            ]["oi_delta"].sum()
        )

        total_oi_delta = float(
            df["oi_delta"].sum()
        )

    else:

        call_oi_delta = 0.0
        put_oi_delta = 0.0
        total_oi_delta = 0.0

    if "oi_delta_available" in df.columns:

        oi_delta_available = bool(
            df[
                "oi_delta_available"
            ].any()
        )

    else:

        oi_delta_available = False

    if (
        "oi_delta_status" in df.columns
        and not df.empty
    ):

        oi_delta_status = str(
            df[
                "oi_delta_status"
            ].iloc[0]
        )

    else:

        oi_delta_status = (
            "NO_PREVIOUS_SNAPSHOT"
        )

    return {

        "call_volume": call_volume,

        "put_volume": put_volume,

        "call_oi": call_oi,

        "put_oi": put_oi,

        "call_premium": call_premium,

        "put_premium": put_premium,

        "call_volume_ratio":
        call_volume_ratio,

        "call_oi_ratio":
        call_oi_ratio,

        "call_premium_ratio":
        call_premium_ratio,

        "atm_iv":
        atm_iv,

        "dte_buckets":
        dte_buckets,

        "call_oi_delta":
        call_oi_delta,

        "put_oi_delta":
        put_oi_delta,

        "total_oi_delta":
        total_oi_delta,

        "oi_delta_available":
        oi_delta_available,

        "oi_delta_status":
        oi_delta_status
    }


# ============================================================
# WALLS
# ============================================================

def find_walls(
    df,
    current_price
):

    active = df[
        df["openInterest"] > 0
    ].copy()

    if active.empty:

        return {
            "call_wall": None,
            "put_wall": None,
            "call_wall_gex": 0.0,
            "put_wall_gex": 0.0,
            "call_wall_oi": 0.0,
            "put_wall_oi": 0.0
        }

    active["distance_pct"] = (
        abs(
            active["strike"]
            - current_price
        )
        / current_price
        * 100
    )

    active = active[
        active["distance_pct"]
        <= WALL_RANGE_PCT
    ]

    calls = active[
        (
            active["option_type"]
            == "CALL"
        )
        &
        (
            active["strike"]
            > current_price
        )
    ]

    puts = active[
        (
            active["option_type"]
            == "PUT"
        )
        &
        (
            active["strike"]
            < current_price
        )
    ]

    call_wall = None
    call_wall_gex = 0.0
    call_wall_oi = 0.0

    if not calls.empty:

        grouped = (
            calls
            .groupby("strike")
            .agg(
                gex=("GEX", "sum"),
                oi=("openInterest", "sum")
            )
        )

        if not grouped.empty:

            idx = grouped["gex"].idxmax()

            call_wall = float(idx)

            call_wall_gex = float(
                grouped.loc[
                    idx,
                    "gex"
                ]
            )

            call_wall_oi = float(
                grouped.loc[
                    idx,
                    "oi"
                ]
            )

    put_wall = None
    put_wall_gex = 0.0
    put_wall_oi = 0.0

    if not puts.empty:

        grouped = (
            puts
            .groupby("strike")
            .agg(
                gex=("GEX", "sum"),
                oi=("openInterest", "sum")
            )
        )

        if not grouped.empty:

            idx = grouped["gex"].idxmin()

            put_wall = float(idx)

            put_wall_gex = float(
                grouped.loc[
                    idx,
                    "gex"
                ]
            )

            put_wall_oi = float(
                grouped.loc[
                    idx,
                    "oi"
                ]
            )

    return {

        "call_wall": call_wall,

        "put_wall": put_wall,

        "call_wall_gex":
        call_wall_gex,

        "put_wall_gex":
        put_wall_gex,

        "call_wall_oi":
        call_wall_oi,

        "put_wall_oi":
        put_wall_oi
    }


# ============================================================
# TOP FLOW
# ============================================================

def find_top_flow(
    df,
    current_price
):

    active = df[
        df["volume"] > 0
    ].copy()

    if active.empty:

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    active["distance_pct"] = (
        abs(
            active["strike"]
            - current_price
        )
        / current_price
        * 100
    )

    active = active[
        active["distance_pct"]
        <= WALL_RANGE_PCT
    ]

    calls = (
        active[
            active["option_type"]
            == "CALL"
        ]
        .sort_values(
            [
                "traded_premium_proxy",
                "volume"
            ],
            ascending=False
        )
        .head(5)
    )

    puts = (
        active[
            active["option_type"]
            == "PUT"
        ]
        .sort_values(
            [
                "traded_premium_proxy",
                "volume"
            ],
            ascending=False
        )
        .head(5)
    )

    return calls, puts


# ============================================================
# DATA QUALITY
# ============================================================

def calculate_data_quality(df):

    total = len(df)

    if total == 0:

        return {
            "score": 0,
            "label": "LOW"
        }

    valid_bidask = (
        (
            df["bid"] > 0
        )
        &
        (
            df["ask"] > 0
        )
        &
        (
            df["ask"] >= df["bid"]
        )
    ).mean()

    active_volume = (
        df["volume"] > 0
    ).mean()

    score = (
        valid_bidask * 60
        + active_volume * 40
    )

    if score >= 75:

        label = "HIGH"

    elif score >= 45:

        label = "MEDIUM"

    else:

        label = "LOW"

    return {
        "score": score,
        "label": label
    }


# ============================================================
# STRUCTURE
# ============================================================

def get_structure_location(
    current_price,
    walls
):

    put_wall = walls.get(
        "put_wall"
    )

    call_wall = walls.get(
        "call_wall"
    )

    if (
        put_wall is not None
        and call_wall is not None
    ):

        if current_price < put_wall:
            return "BELOW_PUT_WALL"

        if current_price > call_wall:
            return "ABOVE_CALL_WALL"

        return "BETWEEN_WALLS"

    if put_wall is not None:

        if current_price < put_wall:
            return "BELOW_PUT_WALL"

        return "ABOVE_PUT_WALL_ONLY"

    if call_wall is not None:

        if current_price > call_wall:
            return "ABOVE_CALL_WALL"

        return "BELOW_CALL_WALL_ONLY"

    return "NO_WALL"


def build_structure_interpretation(
    current_price,
    walls,
    greeks
):

    put_wall = walls.get(
        "put_wall"
    )

    call_wall = walls.get(
        "call_wall"
    )

    gex = float(
        greeks.get(
            "GEX",
            0
        )
    )

    delta = float(
        greeks.get(
            "Delta",
            0
        )
    )

    hiro = float(
        greeks.get(
            "HIRO",
            0
        )
    )

    location = get_structure_location(
        current_price,
        walls
    )

    if location == "BELOW_PUT_WALL":

        price_text = (
            f"현재가는 Put Wall "
            f"${put_wall:g} 아래입니다."
        )

    elif location == "ABOVE_CALL_WALL":

        price_text = (
            f"현재가는 Call Wall "
            f"${call_wall:g} 위입니다."
        )

    elif location == "BETWEEN_WALLS":

        price_text = (
            f"현재가는 Put Wall "
            f"${put_wall:g}과 Call Wall "
            f"${call_wall:g} 사이입니다."
        )

    elif location == "ABOVE_PUT_WALL_ONLY":

        price_text = (
            f"현재가는 Put Wall "
            f"${put_wall:g} 위입니다."
        )

    elif location == "BELOW_CALL_WALL_ONLY":

        price_text = (
            f"현재가는 Call Wall "
            f"${call_wall:g} 아래입니다."
        )

    else:

        price_text = (
            "현재 옵션 핵심 가격대가 "
            "뚜렷하지 않습니다."
        )

    if gex > 0:

        gex_text = (
            "GEX는 양(+)으로 "
            "가격 안정화 성격이 강합니다."
        )

    elif gex < 0:

        gex_text = (
            "GEX는 음(-)으로 "
            "변동성 확대 가능성이 있습니다."
        )

    else:

        gex_text = (
            "GEX는 중립에 가깝습니다."
        )

    if delta > 0:

        delta_text = (
            "Delta는 양(+)입니다."
        )

    elif delta < 0:

        delta_text = (
            "Delta는 음(-)입니다."
        )

    else:

        delta_text = (
            "Delta는 중립입니다."
        )

    if hiro > 0:

        hiro_text = (
            "HIRO Proxy는 양(+)입니다."
        )

    elif hiro < 0:

        hiro_text = (
            "HIRO Proxy는 음(-)입니다."
        )

    else:

        hiro_text = (
            "HIRO Proxy는 중립입니다."
        )

    return {

        "location": location,

        "price_text": price_text,

        "gex_text": gex_text,

        "delta_text": delta_text,

        "hiro_text": hiro_text,

        "text": " ".join(
            [
                price_text,
                gex_text,
                delta_text,
                hiro_text
            ]
        )
    }


# ============================================================
# ENTRY
# ============================================================

def judge_new_entry(
    current_price,
    walls,
    greeks
):

    put_wall = walls.get(
        "put_wall"
    )

    call_wall = walls.get(
        "call_wall"
    )

    delta = float(
        greeks.get(
            "Delta",
            0
        )
    )

    hiro = float(
        greeks.get(
            "HIRO",
            0
        )
    )

    gex = float(
        greeks.get(
            "GEX",
            0
        )
    )

    if (
        put_wall is None
        and call_wall is None
    ):

        return {
            "label": "🔴 진입 금지",
            "reason":
            "핵심 옵션 가격대가 확인되지 않아 "
            "신규 진입 우위가 부족합니다."
        }

    if (
        put_wall is not None
        and current_price < put_wall
    ):

        return {
            "label": "🔴 진입 금지",
            "reason":
            f"현재가가 Put Wall "
            f"${put_wall:g} 아래에 있습니다."
        }

    if (
        call_wall is not None
        and current_price > call_wall
    ):

        if (
            delta > 0
            and hiro > 0
            and gex >= 0
        ):

            return {
                "label": "🟢 진입 가능",
                "reason":
                "Call Wall 상단에서 "
                "Delta와 HIRO가 긍정적이고 "
                "GEX도 안정적입니다."
            }

        return {
            "label": "🟡 확인 후 진입",
            "reason":
            "Call Wall 상단이지만 "
            "방향성 확인이 필요합니다."
        }

    if call_wall is not None:

        distance = (
            (
                call_wall
                - current_price
            )
            / current_price
            * 100
        )

        if (
            0 <= distance
            <= ENTRY_RESISTANCE_PCT
        ):

            return {
                "label": "🟡 확인 후 진입",
                "reason":
                f"Call Wall "
                f"${call_wall:g} 돌파 확인이 필요합니다."
            }

    if (
        put_wall is not None
        and current_price >= put_wall
        and delta > 0
        and hiro > 0
    ):

        return {
            "label": "🟡 확인 후 진입",
            "reason":
            "Put Wall 위에서 "
            "Delta와 HIRO가 긍정적입니다."
        }

    return {
        "label": "🔴 진입 금지",
        "reason":
        "현재 옵션 구조만으로는 "
        "신규 진입 우위가 충분하지 않습니다."
    }


# ============================================================
# HOLDING
# ============================================================

def judge_holding(
    current_price,
    walls,
    greeks
):

    put_wall = walls.get(
        "put_wall"
    )

    gex = float(
        greeks.get(
            "GEX",
            0
        )
    )

    delta = float(
        greeks.get(
            "Delta",
            0
        )
    )

    hiro = float(
        greeks.get(
            "HIRO",
            0
        )
    )

    if (
        put_wall is not None
        and current_price < put_wall
    ):

        return {
            "label": "🔴 이탈검토",
            "reason":
            f"현재가가 Put Wall "
            f"${put_wall:g} 아래입니다."
        }

    if put_wall is not None:

        distance = (
            (
                current_price
                - put_wall
            )
            / current_price
            * 100
        )

        if (
            0 <= distance
            <= HOLDING_SUPPORT_BUFFER_PCT
        ):

            return {
                "label": "🟡 주의",
                "reason":
                f"Put Wall "
                f"${put_wall:g} 부근입니다."
            }

    if (
        delta < 0
        and hiro < 0
    ):

        return {
            "label": "🟠 축소검토",
            "reason":
            "Delta와 HIRO가 모두 약합니다."
        }

    if (
        gex < 0
        and (
            delta < 0
            or hiro < 0
        )
    ):

        return {
            "label": "🟠 축소검토",
            "reason":
            "음의 GEX와 약한 방향성이 "
            "동시에 나타납니다."
        }

    if (
        put_wall is not None
        and current_price > put_wall
        and delta > 0
        and hiro > 0
    ):

        return {
            "label": "🟢 유지",
            "reason":
            "Put Wall 위에 있고 "
            "Delta/HIRO가 긍정적입니다."
        }

    if (
        put_wall is not None
        and current_price >= put_wall
        and gex > 0
    ):

        return {
            "label": "🟢 유지",
            "reason":
            "Put Wall 위이며 "
            "GEX가 안정적입니다."
        }

    return {
        "label": "🟡 주의",
        "reason":
        "추가 방향성 확인이 필요합니다."
    }


# ============================================================
# SUMMARY
# ============================================================

def build_my_summary(
    current_price,
    walls,
    greeks
):

    return {

        "structure":
        build_structure_interpretation(
            current_price,
            walls,
            greeks
        ),

        "new_entry":
        judge_new_entry(
            current_price,
            walls,
            greeks
        ),

        "holding":
        judge_holding(
            current_price,
            walls,
            greeks
        )
    }


# ============================================================
# REPORT
# ============================================================

def build_report(
    ticker,
    current_price,
    df,
    greeks,
    flow,
    walls,
    quality,
    price_context=None
):

    summary = build_my_summary(
        current_price,
        walls,
        greeks
    )

    structure = summary["structure"]
    new_entry = summary["new_entry"]
    holding = summary["holding"]

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"🔥 {ticker} OPTION SEARCH"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    lines.append(
        f"💰 정규장 종가: "
        f"${current_price:.2f}"
    )

    if price_context:

        ah = price_context.get(
            "after_hours_price"
        )

        ah_pct = price_context.get(
            "after_hours_change_pct"
        )

        if ah is not None:

            lines.append(
                f"🌙 시간외 현재가: "
                f"${ah:.2f}"
                + (
                    f" ({ah_pct:+.2f}%)"
                    if ah_pct is not None
                    else ""
                )
            )

        else:

            lines.append(
                "🌙 시간외 현재가: N/A"
            )

        lines.append(
            f"📊 옵션 계산 기준가: "
            f"${current_price:.2f}"
        )

    lines.append(
        f"📊 옵션 행수: "
        f"{len(df):,}"
    )

    lines.append(
        f"📅 DTE: "
        f"{int(df['DTE'].min())} ~ "
        f"{int(df['DTE'].max())}"
    )

    lines.append(
        f"🧪 데이터 품질: "
        f"{quality['label']} "
        f"({quality['score']:.0f}/100)"
    )

    lines.append("")

    # --------------------------------------------------------
    # FLOW
    # --------------------------------------------------------

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📊 1. OPTION FLOW"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"CALL Volume: "
        f"{flow['call_volume']:,.0f}"
    )

    lines.append(
        f"PUT Volume: "
        f"{flow['put_volume']:,.0f}"
    )

    lines.append(
        f"CALL Volume Ratio: "
        f"{flow['call_volume_ratio'] * 100:.1f}%"
    )

    lines.append(
        f"CALL OI: "
        f"{flow['call_oi']:,.0f}"
    )

    lines.append(
        f"PUT OI: "
        f"{flow['put_oi']:,.0f}"
    )

    lines.append(
        f"CALL OI Ratio: "
        f"{flow['call_oi_ratio'] * 100:.1f}%"
    )

    lines.append(
        f"CALL OI Δ: "
        f"{flow['call_oi_delta']:+,.0f}"
    )

    lines.append(
        f"PUT OI Δ: "
        f"{flow['put_oi_delta']:+,.0f}"
    )

    if flow["oi_delta_available"]:

        status = (
            flow["oi_delta_status"]
            .replace(
                "COMPARED_WITH_",
                ""
            )
        )

        lines.append(
            f"OI 비교 기준: {status}"
        )

    else:

        lines.append(
            "OI 비교 기준: "
            "전일 snapshot 없음"
        )

    lines.append(
        f"CALL 거래대금 Proxy: "
        f"{format_money(flow['call_premium'])}"
    )

    lines.append(
        f"PUT 거래대금 Proxy: "
        f"{format_money(flow['put_premium'])}"
    )

    lines.append(
        f"CALL Premium Ratio: "
        f"{flow['call_premium_ratio'] * 100:.1f}%"
    )

    lines.append("")

    # --------------------------------------------------------
    # GREEKS
    # --------------------------------------------------------

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🧮 2. AGGREGATE GREEKS"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"Delta Exposure: "
        f"{format_signed_money(greeks['Delta'])}"
    )

    lines.append(
        f"GEX / 1% Move: "
        f"{format_signed_money(greeks['GEX'])}"
    )

    lines.append(
        f"Vanna Exposure: "
        f"{format_signed_money(greeks['Vanna'])}"
    )

    lines.append(
        f"Charm Exposure: "
        f"{format_signed_money(greeks['Charm'])}"
    )

    lines.append(
        f"Vega Exposure: "
        f"{format_signed_money(greeks['Vega'])}"
    )

    lines.append(
        f"HIRO-like Flow: "
        f"{format_signed_money(greeks['HIRO'])}"
    )

    lines.append(
        f"ATM IV: "
        f"{flow['atm_iv'] * 100:.1f}%"
    )

    lines.append("")

    # --------------------------------------------------------
    # WALL
    # --------------------------------------------------------

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🧱 3. OPTION STRUCTURE"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    if walls["call_wall"] is not None:

        distance = (
            (
                walls["call_wall"]
                - current_price
            )
            / current_price
            * 100
        )

        lines.append(
            f"📈 Call Wall: "
            f"${walls['call_wall']:g} "
            f"({distance:+.1f}%)"
        )

        # FIXED: nested f-string syntax error
        call_wall_gex_text = format_signed_money(
            walls["call_wall_gex"]
        )

        lines.append(
            f"   GEX: {call_wall_gex_text}"
        )

        lines.append(
            f"   OI: "
            f"{walls['call_wall_oi']:,.0f}"
        )

    else:

        lines.append(
            "📈 Call Wall: N/A"
        )

    if walls["put_wall"] is not None:

        distance = (
            (
                current_price
                - walls["put_wall"]
            )
            / current_price
            * 100
        )

        lines.append(
            f"📉 Put Wall: "
            f"${walls['put_wall']:g} "
            f"(-{distance:.1f}%)"
        )

        # FIXED: nested f-string syntax error
        put_wall_gex_text = format_signed_money(
            walls["put_wall_gex"]
        )

        lines.append(
            f"   GEX: {put_wall_gex_text}"
        )

        lines.append(
            f"   OI: "
            f"{walls['put_wall_oi']:,.0f}"
        )

    else:

        lines.append(
            "📉 Put Wall: N/A"
        )

    lines.append("")

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🏗️ 4. STRUCTURE"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"📍 가격 위치: "
        f"{structure['location']}"
    )

    lines.append(
        f"• {structure['price_text']}"
    )

    lines.append(
        f"• {structure['gex_text']}"
    )

    lines.append(
        f"• {structure['delta_text']}"
    )

    lines.append(
        f"• {structure['hiro_text']}"
    )

    lines.append("")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🧠 5. MY SUMMARY"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"🆕 신규 진입: "
        f"{new_entry['label']}"
    )

    lines.append(
        f"   └ {new_entry['reason']}"
    )

    lines.append("")

    lines.append(
        f"📦 보유 판단: "
        f"{holding['label']}"
    )

    lines.append(
        f"   └ {holding['reason']}"
    )

    lines.append("")

    # --------------------------------------------------------
    # TOP FLOW
    # --------------------------------------------------------

    top_calls, top_puts = find_top_flow(
        df,
        current_price
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🔥 6. TOP CALL FLOW"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    if top_calls.empty:

        lines.append("없음")

    else:

        for _, row in top_calls.iterrows():

            lines.append(
                f"• ${row['strike']:g} "
                f"| DTE {int(row['DTE'])} "
                f"| Vol {int(row['volume']):,} "
                f"| "
                f"{format_money(row['traded_premium_proxy'])}"
            )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🔻 7. TOP PUT FLOW"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    if top_puts.empty:

        lines.append("없음")

    else:

        for _, row in top_puts.iterrows():

            lines.append(
                f"• ${row['strike']:g} "
                f"| DTE {int(row['DTE'])} "
                f"| Vol {int(row['volume']):,} "
                f"| "
                f"{format_money(row['traded_premium_proxy'])}"
            )

    lines.append("")

    # --------------------------------------------------------
    # OI DELTA
    # --------------------------------------------------------

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📈 8. OI DELTA"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    if flow["oi_delta_available"]:

        oi = find_top_oi_delta(
            df,
            current_price,
            5
        )

        def append_oi_group(
            title,
            data
        ):

            lines.append("")
            lines.append(title)

            if data.empty:

                lines.append("없음")
                return

            for _, row in data.iterrows():

                delta = float(
                    row["oi_delta"]
                )

                icon = (
                    "🔺"
                    if delta > 0
                    else "🔻"
                )

                lines.append(
                    f"• ${row['strike']:g} "
                    f"| DTE {int(row['DTE'])} "
                    f"| OI {int(row['openInterest']):,} "
                    f"| Δ {delta:+,.0f} {icon} "
                    f"| Vol {int(row['volume']):,}"
                )

        append_oi_group(
            "🟢 CALL OI 증가 TOP 5",
            oi["call_increase"]
        )

        append_oi_group(
            "🔻 CALL OI 감소 TOP 5",
            oi["call_decrease"]
        )

        append_oi_group(
            "🟢 PUT OI 증가 TOP 5",
            oi["put_increase"]
        )

        append_oi_group(
            "🔻 PUT OI 감소 TOP 5",
            oi["put_decrease"]
        )

    else:

        lines.append(
            "⚠️ 이전 OI snapshot이 없어 "
            "비교할 수 없습니다."
        )

        lines.append(
            "현재 데이터를 snapshot으로 저장했습니다."
        )

    lines.append("")

    # --------------------------------------------------------
    # DTE
    # --------------------------------------------------------

    dte = flow["dte_buckets"]

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📅 9. DTE DISTRIBUTION"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"0~7 DTE: {dte['0_7']:,}"
    )

    lines.append(
        f"8~30 DTE: {dte['8_30']:,}"
    )

    lines.append(
        f"31~60 DTE: {dte['31_60']:,}"
    )

    lines.append(
        f"61~180 DTE: {dte['61_180']:,}"
    )

    lines.append("")

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "⚠️ DATA LIMITATIONS"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "• 거래대금 Proxy는 실제 Buy/Sell 체결 방향이 아닙니다."
    )

    lines.append(
        "• GEX는 OI 기반 Dealer Positioning Proxy입니다."
    )

    lines.append(
        "• HIRO-like Flow는 실제 HIRO 데이터가 아닙니다."
    )

    lines.append(
        "• OI만으로 실제 Long/Short 포지션을 확정할 수 없습니다."
    )

    lines.append(
        "• 무료 Yahoo Finance 옵션 데이터 기반 분석입니다."
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return "\n".join(lines)


# ============================================================
# SAVE CSV
# ============================================================

def save_option_csv(
    ticker,
    df
):

    filename = os.path.join(
        RESULT_DIR,
        f"{ticker}_OPTION_SEARCH.csv"
    )

    df.to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"💾 CSV 저장: {filename}"
    )

    return filename


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(text):

    if not BOT_TOKEN:

        print(
            "⚠️ TELEGRAM_BOT_TOKEN이 없습니다."
        )

        return False

    if not CHAT_ID:

        print(
            "⚠️ TELEGRAM_CHAT_ID가 없습니다."
        )

        return False

    if not text:

        return False

    MAX_LENGTH = 3800

    chunks = []

    if len(text) <= MAX_LENGTH:

        chunks = [text]

    else:

        current = ""

        for line in text.splitlines(
            True
        ):

            if (
                len(current)
                + len(line)
                <= MAX_LENGTH
            ):

                current += line

            else:

                if current:
                    chunks.append(
                        current
                    )

                while len(line) > MAX_LENGTH:

                    chunks.append(
                        line[:MAX_LENGTH]
                    )

                    line = line[
                        MAX_LENGTH:
                    ]

                current = line

        if current:
            chunks.append(
                current
            )

    url = (
        "https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )

    success = True

    total = len(chunks)

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        if total > 1:

            chunk = (
                f"📨 PART {i}/{total}\n\n"
                + chunk
            )

        data = {
            "chat_id": CHAT_ID,
            "text": chunk,
            "parse_mode": "HTML"
        }

        try:

            response = requests.post(
                url,
                data=data,
                timeout=30
            )

            print(
                f"📨 Telegram "
                f"{i}/{total}: "
                f"{response.status_code}"
            )

            if not response.ok:

                print(
                    response.text[:1000]
                )

                success = False

            time.sleep(0.3)

        except Exception as e:

            print(
                f"❌ Telegram 오류: {e}"
            )

            success = False

    return success


# ============================================================
# ANALYZE
# ============================================================

def analyze_ticker(ticker):

    ticker = (
        ticker
        .upper()
        .strip()
    )

    print("")
    print("=" * 70)
    print(
        f"🔥 OPTION SEARCH : {ticker}"
    )
    print("=" * 70)

    try:

        price_context = (
            get_market_price_context(
                ticker
            )
        )

        if price_context is None:
            return None

        current_price = (
            price_context[
                "option_analysis_price"
            ]
        )

        print("")

        df = get_option_data(
            ticker
        )

        print(
            f"📊 옵션 행수: "
            f"{len(df):,}"
        )

        print(
            f"📅 DTE: "
            f"{df['DTE'].min()} ~ "
            f"{df['DTE'].max()}"
        )

        print("")

        print(
            "📊 Greeks / GEX 계산..."
        )

        df = calculate_option_metrics(
            df,
            current_price
        )

        # ====================================================
        # OI DELTA
        # ====================================================

        df, previous_oi_date = (
            calculate_oi_delta(
                df,
                ticker
            )
        )

        if previous_oi_date is not None:

            print(
                f"📈 OI 비교 기준일: "
                f"{previous_oi_date.strftime('%Y-%m-%d')}"
            )

        else:

            print(
                "📈 OI 비교 기준일: 없음"
            )

        greeks = (
            calculate_aggregate_greeks(
                df
            )
        )

        flow = (
            calculate_flow_summary(
                df,
                current_price
            )
        )

        walls = (
            find_walls(
                df,
                current_price
            )
        )

        quality = (
            calculate_data_quality(
                df
            )
        )

        report = build_report(
            ticker,
            current_price,
            df,
            greeks,
            flow,
            walls,
            quality,
            price_context
        )

        print("")
        print(report)
        print("")

        csv_file = save_option_csv(
            ticker,
            df
        )

        # ====================================================
        # SAVE OI AFTER COMPARISON
        # ====================================================

        oi_snapshot_file = (
            save_oi_snapshot(
                ticker,
                df
            )
        )

        telegram_ok = send_telegram(
            report
        )

        return {

            "ticker": ticker,

            "current_price":
            current_price,

            "price_context":
            price_context,

            "df": df,

            "greeks":
            greeks,

            "flow":
            flow,

            "walls":
            walls,

            "quality":
            quality,

            "report":
            report,

            "csv_file":
            csv_file,

            "oi_snapshot_file":
            oi_snapshot_file,

            "previous_oi_date":
            previous_oi_date,

            "telegram_ok":
            telegram_ok
        }

    except Exception as e:

        print("")
        print(
            f"❌ {ticker} OPTION SEARCH 실패"
        )

        print(
            f"❌ 오류: {e}"
        )

        import traceback

        traceback.print_exc()

        return None


# ============================================================
# RUN
# ============================================================

def run(ticker):

    result = analyze_ticker(
        ticker
    )

    return result is not None


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "사용법:"
        )

        print(
            "python option_search.py TICKER"
        )

        sys.exit(1)

    result = analyze_ticker(
        sys.argv[1]
    )

    if result is None:

        sys.exit(1)

    print("")
    print(
        "🔥 OPTION SEARCH 완료"
    )

    print(
        f"📌 Ticker: "
        f"{result['ticker']}"
    )

    print(
        f"📌 CSV: "
        f"{result['csv_file']}"
    )

    print(
        f"📌 OI Snapshot: "
        f"{result['oi_snapshot_file']}"
    )

    print(
        f"📌 OI 비교: "
        f"{result['previous_oi_date']}"
    )

    print(
        f"📌 Telegram: "
        f"{'SUCCESS' if result['telegram_ok'] else 'SKIPPED/FAILED'}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

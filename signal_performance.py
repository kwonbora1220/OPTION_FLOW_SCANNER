from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

ET = ZoneInfo(
    "America/New_York"
)

ROOT_DIR = Path(
    __file__
).resolve().parent

RESULT_DIR = (
    ROOT_DIR
    / "03_RESULTS"
    / "daily"
)

HISTORY_DIR = (
    ROOT_DIR
    / "03_RESULTS"
    / "history"
)

HISTORY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ARCHIVE
# ============================================================

def archive_ranking(
    ranking_file
):

    src = Path(
        ranking_file
    )


    if not src.exists():

        return ""


    now = datetime.now(
        ET
    )


    stamp = now.strftime(
        "%Y%m%d_%H%M%S"
    )


    dest = (
        HISTORY_DIR
        /
        f"OPTION_FINAL_RANKING_{stamp}.csv"
    )


    df = pd.read_csv(
        src
    )


    df[
        "scan_date_et"
    ] = now.strftime(
        "%Y-%m-%d"
    )


    df[
        "scan_time_et"
    ] = now.strftime(
        "%H:%M:%S"
    )


    df.to_csv(
        dest,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        f"💾 Ranking archive: {dest}"
    )


    return str(dest)


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():

    files = sorted(
        HISTORY_DIR.glob(
            "OPTION_FINAL_RANKING_*.csv"
        )
    )


    if not files:

        return pd.DataFrame()


    frames = []


    for file in files:

        try:

            frames.append(
                pd.read_csv(
                    file
                )
            )

        except Exception:

            pass


    if not frames:

        return pd.DataFrame()


    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================
# FLOW DIRECTION
# ============================================================

def flow_direction(
    row
):

    values = []


    for column in [
        "call_volume_ratio",
        "call_premium_ratio"
    ]:

        try:

            values.append(
                float(
                    row.get(
                        column
                    )
                )
            )

        except Exception:

            pass


    if not values:

        return "NEUTRAL"


    value = (
        sum(values)
        / len(values)
    )


    if value >= 0.55:

        return "BULLISH"


    if value <= 0.45:

        return "BEARISH"


    return "NEUTRAL"


# ============================================================
# OI DIRECTION
# ============================================================

def oi_direction(
    row
):

    try:

        call_change = float(
            row.get(
                "call_oi_change"
            )
        )

        put_change = float(
            row.get(
                "put_oi_change"
            )
        )

    except Exception:

        return "NEUTRAL"


    if call_change > put_change:

        return "BULLISH"


    if put_change > call_change:

        return "BEARISH"


    return "NEUTRAL"


# ============================================================
# PRICE RESULT
# ============================================================

def get_price_after_days(
    ticker,
    signal_date,
    days
):

    try:

        date_value = (
            pd.Timestamp(
                signal_date
            )
            .normalize()
        )


        today = (
            pd.Timestamp
            .now()
            .normalize()
        )


        history = (
            yf.Ticker(
                ticker
            )
            .history(

                start=(
                    date_value
                    -
                    pd.Timedelta(
                        days=2
                    )
                ).strftime(
                    "%Y-%m-%d"
                ),

                end=(
                    today
                    +
                    pd.Timedelta(
                        days=2
                    )
                ).strftime(
                    "%Y-%m-%d"
                ),

                auto_adjust=False
            )
        )


        if history.empty:

            return None


        try:

            if (
                history.index.tz
                is not None
            ):

                history.index = (
                    history.index
                    .tz_localize(
                        None
                    )
                )

        except Exception:

            pass


        future = history[
            history.index.normalize()
            >
            date_value
        ]


        if len(future) < days:

            return None


        return float(
            future[
                "Close"
            ].iloc[
                days - 1
            ]
        )


    except Exception:

        return None


# ============================================================
# UPDATE OUTCOMES
# ============================================================

def update_outcomes(
    df
):

    if df.empty:

        return df


    for column in [

        "price_1d",
        "price_3d",
        "price_5d",

        "return_1d",
        "return_3d",
        "return_5d",

        "win_1d",
        "win_3d",
        "win_5d"

    ]:

        if column not in df.columns:

            df[column] = None


    cache = {}


    for index, row in df.iterrows():

        ticker = str(
            row.get(
                "ticker",
                ""
            )
        ).upper().strip()


        category = str(
            row.get(
                "category",
                ""
            )
        )


        # 관망은 승/패 계산에서 제외
        if (
            not ticker
            or
            category
            == "🟡 관망"
        ):

            continue


        try:

            entry_price = float(
                row.get(
                    "current_price"
                )
            )

        except Exception:

            continue


        if entry_price <= 0:

            continue


        signal_date = (
            row.get(
                "scan_date_et"
            )
        )


        if pd.isna(
            signal_date
        ):

            continue


        for days in [
            1,
            3,
            5
        ]:

            price_column = (
                f"price_{days}d"
            )

            return_column = (
                f"return_{days}d"
            )

            win_column = (
                f"win_{days}d"
            )


            if pd.notna(
                row.get(
                    price_column
                )
            ):

                continue


            key = (
                ticker,
                str(signal_date),
                days
            )


            if key not in cache:

                cache[key] = (
                    get_price_after_days(
                        ticker,
                        signal_date,
                        days
                    )
                )


            price = cache[key]


            if price is None:

                continue


            return_pct = (
                (
                    price
                    /
                    entry_price
                )
                - 1
            ) * 100


            # 진입 = 상승하면 WIN
            # 회피 = 하락하면 WIN

            if (
                category
                ==
                "🟢 오늘 진입 후보"
            ):

                win = (
                    1
                    if return_pct > 0
                    else 0
                )

            else:

                win = (
                    1
                    if return_pct < 0
                    else 0
                )


            df.at[
                index,
                price_column
            ] = price


            df.at[
                index,
                return_column
            ] = return_pct


            df.at[
                index,
                win_column
            ] = win


    return df


# ============================================================
# FORMAT
# ============================================================

def format_pct(
    value
):

    if value is None:

        return "N/A"


    return (
        f"{value:.1f}%"
    )


# ============================================================
# PERFORMANCE MESSAGE
# ============================================================

def build_performance_message(
    limit=30
):

    df = load_history()


    if df.empty:

        return (
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🧠 <b>SIGNAL PERFORMANCE</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "아직 누적 신호가 없습니다."
        )


    df = update_outcomes(
        df
    )


    df = df.sort_values(

        [
            "scan_date_et",
            "scan_time_et"
        ],

        ascending=False

    ).head(
        limit
    ).copy()


    # ========================================================
    # CATEGORY COUNT
    # ========================================================

    entry_count = int(
        (
            df["category"]
            ==
            "🟢 오늘 진입 후보"
        ).sum()
    )


    watch_count = int(
        (
            df["category"]
            ==
            "🟡 관망"
        ).sum()
    )


    avoid_count = int(
        (
            df["category"]
            ==
            "🔴 회피"
        ).sum()
    )


    # ========================================================
    # HIT RATE
    # ========================================================

    entry_results = pd.to_numeric(

        df.loc[
            df["category"]
            ==
            "🟢 오늘 진입 후보",
            "win_1d"
        ],

        errors="coerce"

    ).dropna()


    avoid_results = pd.to_numeric(

        df.loc[
            df["category"]
            ==
            "🔴 회피",
            "win_1d"
        ],

        errors="coerce"

    ).dropna()


    resolved = pd.concat(
        [
            entry_results,
            avoid_results
        ],
        ignore_index=True
    )


    if len(resolved):

        hit_rate = (
            resolved.mean()
            * 100
        )

    else:

        hit_rate = None


    # ========================================================
    # AM AGREEMENT
    # ========================================================

    am_matches = []


    for _, row in df.iterrows():

        am = str(
            row.get(
                "am_direction",
                "NEUTRAL"
            )
        )


        option = str(
            row.get(
                "direction",
                "NEUTRAL"
            )
        )


        if (
            am
            not in
            (
                "BULLISH",
                "BEARISH"
            )
        ):

            continue


        if (
            option
            not in
            (
                "BULLISH",
                "BEARISH"
            )
        ):

            continue


        am_matches.append(

            1
            if am == option
            else 0

        )


    if am_matches:

        am_rate = (
            sum(am_matches)
            /
            len(am_matches)
            * 100
        )

    else:

        am_rate = None


    # ========================================================
    # OI AGREEMENT
    # ========================================================

    oi_matches = []


    for _, row in df.iterrows():

        option = str(
            row.get(
                "direction",
                "NEUTRAL"
            )
        )


        oi = oi_direction(
            row
        )


        if (
            option
            in
            (
                "BULLISH",
                "BEARISH"
            )
            and
            oi
            in
            (
                "BULLISH",
                "BEARISH"
            )
        ):

            oi_matches.append(

                1
                if option == oi
                else 0

            )


    if oi_matches:

        oi_rate = (
            sum(oi_matches)
            /
            len(oi_matches)
            * 100
        )

    else:

        oi_rate = None


    # ========================================================
    # FLOW AGREEMENT
    # ========================================================

    flow_matches = []


    for _, row in df.iterrows():

        option = str(
            row.get(
                "direction",
                "NEUTRAL"
            )
        )


        flow = flow_direction(
            row
        )


        if (
            option
            in
            (
                "BULLISH",
                "BEARISH"
            )
            and
            flow
            in
            (
                "BULLISH",
                "BEARISH"
            )
        ):

            flow_matches.append(

                1
                if option == flow
                else 0

            )


    if flow_matches:

        flow_rate = (
            sum(flow_matches)
            /
            len(flow_matches)
            * 100
        )

    else:

        flow_rate = None


    # ========================================================
    # MESSAGE
    # ========================================================

    return "\n".join(

        [

            "━━━━━━━━━━━━━━━━━━━━",

            "🧠 <b>SIGNAL PERFORMANCE</b>",

            "━━━━━━━━━━━━━━━━━━━━",

            "",

            f"최근 {len(df)}개 신호",

            "",

            f"🟢 진입: {entry_count}",

            f"🟡 관망: {watch_count}",

            f"🔴 회피: {avoid_count}",

            "",

            (
                f"적중률: "
                f"<b>{format_pct(hit_rate)}</b>"
            ),

            "",

            (
                f"AM Context 일치: "
                f"{format_pct(am_rate)}"
            ),

            (
                f"OI Delta 일치: "
                f"{format_pct(oi_rate)}"
            ),

            (
                f"Flow 일치: "
                f"{format_pct(flow_rate)}"
            ),

            "",

            "※ 적중률은 진입/회피의 "
            "1거래일 결과 기준.",

            "※ 관망은 승/패 계산에서 제외.",

            "※ 표본이 쌓일수록 "
            "통계 신뢰도가 올라갑니다.",

        ]

    )

import os
from datetime import date, datetime

import pandas as pd
import requests

from selected_symbols import SELECTED_SYMBOLS
from option_search import analyze_ticker


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "03_RESULTS",
    "daily"
)

OI_HISTORY_DIR = os.path.join(
    RESULT_DIR,
    "oi_history"
)

os.makedirs(
    OI_HISTORY_DIR,
    exist_ok=True
)


BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    ""
)

CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID",
    ""
)


# 현재가 주변 몇 %의 행사가를 보여줄지
STRIKE_WINDOW_PCT = 15.0

# 증가/감소 TOP
TOP_N = 5


# ============================================================
# SNAPSHOT PATH
# ============================================================

def snapshot_path(
    ticker,
    snapshot_date
):

    return os.path.join(

        OI_HISTORY_DIR,

        f"{ticker.upper().strip()}_OI_"
        f"{snapshot_date.strftime('%Y%m%d')}.csv"

    )


# ============================================================
# BUILD SNAPSHOT
# ============================================================

def build_snapshot(df):

    required = [
        "expiration",
        "DTE",
        "strike",
        "option_type",
        "openInterest"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"OI snapshot 필수 컬럼 없음: {missing}"
        )

    snapshot = df[
        required
    ].copy()

    snapshot["expiration"] = (
        snapshot["expiration"]
        .astype(str)
    )

    snapshot["option_type"] = (
        snapshot["option_type"]
        .astype(str)
        .str.upper()
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
    ).fillna(0).clip(lower=0)

    snapshot = snapshot.dropna(
        subset=[
            "expiration",
            "strike",
            "option_type"
        ]
    )

    snapshot = snapshot.drop_duplicates(
        subset=[
            "expiration",
            "strike",
            "option_type"
        ],
        keep="last"
    )

    return snapshot


# ============================================================
# FIND PREVIOUS SNAPSHOT
# ============================================================

def previous_snapshot(
    ticker,
    today
):

    prefix = (
        f"{ticker.upper().strip()}_OI_"
    )

    candidates = []

    if not os.path.exists(
        OI_HISTORY_DIR
    ):
        return None, None

    for filename in os.listdir(
        OI_HISTORY_DIR
    ):

        if not (
            filename.startswith(prefix)
            and filename.endswith(".csv")
        ):
            continue

        raw_date = filename[
            len(prefix):-4
        ]

        try:

            snapshot_date = (
                datetime.strptime(
                    raw_date,
                    "%Y%m%d"
                ).date()
            )

        except ValueError:

            continue

        if snapshot_date < today:

            candidates.append(
                (
                    snapshot_date,
                    os.path.join(
                        OI_HISTORY_DIR,
                        filename
                    )
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

def compare_oi(
    df,
    ticker,
    today
):

    current = build_snapshot(df)

    previous_date, previous_path = (
        previous_snapshot(
            ticker,
            today
        )
    )

    # --------------------------------------------------------
    # NO PREVIOUS SNAPSHOT
    # --------------------------------------------------------

    if previous_path is None:

        current[
            "previous_openInterest"
        ] = pd.NA

        current[
            "oi_delta"
        ] = pd.NA

        current[
            "oi_delta_available"
        ] = False

        return (
            current,
            None
        )

    # --------------------------------------------------------
    # LOAD PREVIOUS
    # --------------------------------------------------------

    previous = pd.read_csv(
        previous_path,
        encoding="utf-8-sig"
    )

    previous["expiration"] = (
        previous["expiration"]
        .astype(str)
    )

    previous["option_type"] = (
        previous["option_type"]
        .astype(str)
        .str.upper()
    )

    previous["strike"] = pd.to_numeric(
        previous["strike"],
        errors="coerce"
    )

    previous["openInterest"] = pd.to_numeric(
        previous["openInterest"],
        errors="coerce"
    ).fillna(0).clip(lower=0)

    previous = previous.dropna(
        subset=[
            "expiration",
            "strike",
            "option_type"
        ]
    )

    previous = previous.drop_duplicates(
        subset=[
            "expiration",
            "strike",
            "option_type"
        ],
        keep="last"
    )

    previous = previous.rename(
        columns={
            "openInterest":
                "previous_openInterest"
        }
    )

    keys = [
        "expiration",
        "strike",
        "option_type"
    ]

    merged = current.merge(
        previous[
            keys
            + ["previous_openInterest"]
        ],
        on=keys,
        how="left"
    )

    merged[
        "previous_openInterest"
    ] = pd.to_numeric(
        merged[
            "previous_openInterest"
        ],
        errors="coerce"
    )

    # 기존에 없던 계약은
    # 과거 OI 0으로 처리
    merged[
        "oi_delta_available"
    ] = merged[
        "previous_openInterest"
    ].notna()

    merged[
        "previous_openInterest"
    ] = merged[
        "previous_openInterest"
    ].fillna(0)

    merged["oi_delta"] = (
        merged["openInterest"]
        - merged["previous_openInterest"]
    )

    return (
        merged,
        previous_date
    )


# ============================================================
# FORMAT DELTA
# ============================================================

def format_delta(
    value
):

    value = int(
        round(
            float(value)
        )
    )

    if value > 0:

        return f"+{value:,}"

    return f"{value:,}"


# ============================================================
# SEND TELEGRAM
# ============================================================

def send_telegram(
    text
):

    if not BOT_TOKEN:

        print(
            "⚠️ TELEGRAM_BOT_TOKEN 없음"
        )

        return

    if not CHAT_ID:

        print(
            "⚠️ TELEGRAM_CHAT_ID 없음"
        )

        return

    url = (
        "https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )

    response = requests.post(

        url,

        data={
            "chat_id": CHAT_ID,
            "text": text
        },

        timeout=30

    )

    response.raise_for_status()

    print(
        "📨 Telegram OI Delta 전송 완료"
    )


# ============================================================
# BUILD TELEGRAM MESSAGE
# ============================================================

def build_message(
    results
):

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📈 OI DELTA FIX"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    for item in results:

        ticker = item[
            "ticker"
        ]

        price = item[
            "price"
        ]

        previous_date = item[
            "previous_date"
        ]

        df = item[
            "df"
        ]

        lines.append("")

        lines.append(
            f"🔥 {ticker} | "
            f"${price:.2f}"
        )

        # ----------------------------------------------------
        # FIRST SNAPSHOT
        # ----------------------------------------------------

        if previous_date is None:

            lines.append(
                "⚠️ 전일 snapshot 없음"
            )

            lines.append(
                "💾 오늘 snapshot 저장 완료"
            )

            continue

        lines.append(
            "📅 비교 기준: "
            f"{previous_date.strftime('%Y-%m-%d')}"
        )

        # ----------------------------------------------------
        # AVAILABLE
        # ----------------------------------------------------

        active = df[
            df["oi_delta_available"]
        ].copy()

        if active.empty:

            lines.append(
                "⚠️ 비교 가능한 OI 없음"
            )

            continue

        # ----------------------------------------------------
        # CURRENT PRICE RANGE
        # ----------------------------------------------------

        low = price * (
            1
            - STRIKE_WINDOW_PCT / 100
        )

        high = price * (
            1
            + STRIKE_WINDOW_PCT / 100
        )

        active = active[
            active["strike"].between(
                low,
                high
            )
        ]

        if active.empty:

            lines.append(
                "⚠️ 현재가 주변 OI 변화 없음"
            )

            continue

        # ----------------------------------------------------
        # CALL / PUT
        # ----------------------------------------------------

        calls = active[
            active["option_type"]
            == "CALL"
        ]

        puts = active[
            active["option_type"]
            == "PUT"
        ]

        # ----------------------------------------------------
        # CALL INCREASE
        # ----------------------------------------------------

        call_up = calls[
            calls["oi_delta"] > 0
        ].nlargest(
            TOP_N,
            "oi_delta"
        )

        if not call_up.empty:

            lines.append(
                "🟢 CALL OI 증가"
            )

            for _, row in (
                call_up.iterrows()
            ):

                lines.append(

                    f"• ${row['strike']:.0f}"
                    f" | DTE {int(row['DTE'])}"
                    f" | ΔOI "
                    f"{format_delta(row['oi_delta'])}"

                )

        # ----------------------------------------------------
        # CALL DECREASE
        # ----------------------------------------------------

        call_down = calls[
            calls["oi_delta"] < 0
        ].nsmallest(
            TOP_N,
            "oi_delta"
        )

        if not call_down.empty:

            lines.append(
                "🔴 CALL OI 감소"
            )

            for _, row in (
                call_down.iterrows()
            ):

                lines.append(

                    f"• ${row['strike']:.0f}"
                    f" | DTE {int(row['DTE'])}"
                    f" | ΔOI "
                    f"{format_delta(row['oi_delta'])}"

                )

        # ----------------------------------------------------
        # PUT INCREASE
        # ----------------------------------------------------

        put_up = puts[
            puts["oi_delta"] > 0
        ].nlargest(
            TOP_N,
            "oi_delta"
        )

        if not put_up.empty:

            lines.append(
                "🟢 PUT OI 증가"
            )

            for _, row in (
                put_up.iterrows()
            ):

                lines.append(

                    f"• ${row['strike']:.0f}"
                    f" | DTE {int(row['DTE'])}"
                    f" | ΔOI "
                    f"{format_delta(row['oi_delta'])}"

                )

        # ----------------------------------------------------
        # PUT DECREASE
        # ----------------------------------------------------

        put_down = puts[
            puts["oi_delta"] < 0
        ].nsmallest(
            TOP_N,
            "oi_delta"
        )

        if not put_down.empty:

            lines.append(
                "🔴 PUT OI 감소"
            )

            for _, row in (
                put_down.iterrows()
            ):

                lines.append(

                    f"• ${row['strike']:.0f}"
                    f" | DTE {int(row['DTE'])}"
                    f" | ΔOI "
                    f"{format_delta(row['oi_delta'])}"

                )

    return "\n".join(
        lines
    )


# ============================================================
# MAIN
# ============================================================

def main():

    today = date.today()

    results = []

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        "📈 OI DELTA MONITOR"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        f"📅 Snapshot date: "
        f"{today}"
    )

    # --------------------------------------------------------
    # EACH TICKER
    # --------------------------------------------------------

    for ticker in SELECTED_SYMBOLS:

        try:

            print("")
            print(
                f"===== {ticker} ====="
            )

            analysis = (
                analyze_ticker(
                    ticker
                )
            )

            df = analysis[
                "df"
            ]

            price = float(
                analysis[
                    "current_price"
                ]
            )

            # ------------------------------------------------
            # COMPARE BEFORE SAVE
            # ------------------------------------------------

            compared_df, previous_date = (
                compare_oi(
                    df,
                    ticker,
                    today
                )
            )

            # ------------------------------------------------
            # SAVE CURRENT SNAPSHOT
            # ------------------------------------------------

            path = snapshot_path(
                ticker,
                today
            )

            snapshot = build_snapshot(
                df
            )

            snapshot.to_csv(

                path,

                index=False,

                encoding="utf-8-sig"

            )

            print(
                f"💾 OI snapshot saved:"
                f" {path}"
            )

            if previous_date:

                print(
                    "🟢 OI comparison:"
                    f" {previous_date}"
                )

            else:

                print(
                    "⚠️ previous snapshot 없음"
                )

            results.append({

                "ticker":
                    ticker,

                "price":
                    price,

                "previous_date":
                    previous_date,

                "df":
                    compared_df

            })

        except Exception as e:

            print(
                f"❌ {ticker} "
                f"OI 처리 실패: {e}"
            )

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    if results:

        message = build_message(
            results
        )

        print("")
        print(message)

        try:

            send_telegram(
                message
            )

        except Exception as e:

            print(
                f"❌ Telegram 실패: {e}"
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

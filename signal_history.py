import os
from datetime import date

import pandas as pd
import yfinance as yf


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


SIGNAL_FILE = os.path.join(
    RESULT_DIR,
    "SIGNAL_HISTORY.csv"
)


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value):

    try:

        if pd.isna(value):
            return None

        return float(value)

    except Exception:

        return None


def _normalize_date(value):

    try:

        ts = pd.Timestamp(
            value
        )

        if pd.isna(ts):
            return None

        return ts.normalize()

    except Exception:

        return None


# ============================================================
# RECORD SIGNAL
# ============================================================

def record_signal(
    ticker,
    score,
    direction,
    category,
    current_price,
    signal_date=None
):

    # --------------------------------------------------------
    # 진입 신호만 기록
    # --------------------------------------------------------

    if category != "🟢 오늘 진입 후보":

        return False

    ticker = str(
        ticker
    ).upper()

    # --------------------------------------------------------
    # 날짜
    # --------------------------------------------------------

    if signal_date is None:

        signal_date = date.today()

    signal_date = _normalize_date(
        signal_date
    )

    if signal_date is None:

        signal_date = pd.Timestamp(
            date.today()
        )

    signal_date_text = (
        signal_date.strftime(
            "%Y-%m-%d"
        )
    )

    # --------------------------------------------------------
    # ENTRY PRICE
    # --------------------------------------------------------

    entry_price = _safe_float(
        current_price
    )

    if (
        entry_price is None
        or
        entry_price <= 0
    ):

        print(
            f"⚠️ {ticker} "
            "잘못된 진입가격"
        )

        return False

    # --------------------------------------------------------
    # 기존 HISTORY
    # --------------------------------------------------------

    if os.path.exists(
        SIGNAL_FILE
    ):

        try:

            history = pd.read_csv(
                SIGNAL_FILE
            )

        except Exception:

            history = pd.DataFrame()

    else:

        history = pd.DataFrame()

    # --------------------------------------------------------
    # 컬럼 보정
    # --------------------------------------------------------

    required_columns = [

        "signal_date",
        "ticker",
        "score",
        "direction",
        "category",
        "entry_price",

        "price_1d",
        "price_3d",
        "price_5d",

        "return_1d",
        "return_3d",
        "return_5d",

        "win_1d",
        "win_3d",
        "win_5d"
    ]

    for column in required_columns:

        if column not in history.columns:

            history[column] = None

    # --------------------------------------------------------
    # 날짜 / ticker 정규화
    # --------------------------------------------------------

    if not history.empty:

        history["signal_date"] = (
            history["signal_date"]
            .astype(str)
        )

        history["ticker"] = (
            history["ticker"]
            .astype(str)
            .str.upper()
        )

        # ----------------------------------------------------
        # 같은 날 같은 종목 제거
        # ----------------------------------------------------

        history = history[
            ~(
                (
                    history[
                        "signal_date"
                    ]
                    == signal_date_text
                )
                &
                (
                    history[
                        "ticker"
                    ]
                    == ticker
                )
            )
        ]

    # --------------------------------------------------------
    # NEW ROW
    # --------------------------------------------------------

    row = {

        "signal_date":
            signal_date_text,

        "ticker":
            ticker,

        "score":
            _safe_float(score),

        "direction":
            direction,

        "category":
            category,

        "entry_price":
            entry_price,

        "price_1d":
            None,

        "price_3d":
            None,

        "price_5d":
            None,

        "return_1d":
            None,

        "return_3d":
            None,

        "return_5d":
            None,

        "win_1d":
            None,

        "win_3d":
            None,

        "win_5d":
            None
    }

    history = pd.concat(
        [
            history,
            pd.DataFrame(
                [row]
            )
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    history.to_csv(
        SIGNAL_FILE,
        index=False
    )

    print(
        f"💾 SIGNAL 저장: "
        f"{ticker} "
        f"{signal_date_text}"
    )

    return True


# ============================================================
# UPDATE SIGNAL RESULTS
# ============================================================

def update_signal_results():

    if not os.path.exists(
        SIGNAL_FILE
    ):

        return

    try:

        history = pd.read_csv(
            SIGNAL_FILE
        )

    except Exception as e:

        print(
            f"⚠️ SIGNAL_HISTORY 읽기 실패: {e}"
        )

        return

    if history.empty:
        return

    # --------------------------------------------------------
    # 컬럼 보정
    # --------------------------------------------------------

    required_columns = [

        "signal_date",
        "ticker",
        "score",
        "direction",
        "category",
        "entry_price",

        "price_1d",
        "price_3d",
        "price_5d",

        "return_1d",
        "return_3d",
        "return_5d",

        "win_1d",
        "win_3d",
        "win_5d"
    ]

    for column in required_columns:

        if column not in history.columns:

            history[column] = None

    history["signal_date"] = pd.to_datetime(
        history["signal_date"],
        errors="coerce"
    )

    today = pd.Timestamp(
        date.today()
    )

    # --------------------------------------------------------
    # 각 신호 업데이트
    # --------------------------------------------------------

    for idx, row in history.iterrows():

        ticker = str(
            row["ticker"]
        ).upper()

        signal_date = _normalize_date(
            row["signal_date"]
        )

        if signal_date is None:
            continue

        entry_price = _safe_float(
            row["entry_price"]
        )

        if (
            entry_price is None
            or
            entry_price <= 0
        ):

            continue

        # ----------------------------------------------------
        # 이미 +1/+3/+5가 모두 계산됐으면 skip
        # ----------------------------------------------------

        if (
            pd.notna(
                row["price_1d"]
            )
            and
            pd.notna(
                row["price_3d"]
            )
            and
            pd.notna(
                row["price_5d"]
            )
        ):

            continue

        try:

            # ------------------------------------------------
            # yfinance
            # ------------------------------------------------

            start = (
                signal_date
                - pd.Timedelta(
                    days=3
                )
            )

            end = (
                today
                + pd.Timedelta(
                    days=1
                )
            )

            hist = yf.Ticker(
                ticker
            ).history(
                start=start.strftime(
                    "%Y-%m-%d"
                ),
                end=end.strftime(
                    "%Y-%m-%d"
                ),
                auto_adjust=False
            )

            if hist.empty:

                print(
                    f"⚠️ {ticker}: "
                    "가격 데이터 없음"
                )

                continue

            closes = (
                hist["Close"]
                .dropna()
            )

            if closes.empty:
                continue

            # ------------------------------------------------
            # timezone 제거
            # ------------------------------------------------

            try:

                if closes.index.tz is not None:

                    closes.index = (
                        closes.index
                        .tz_localize(None)
                    )

            except Exception:

                pass

            # ------------------------------------------------
            # signal date 이후 거래일만
            # ------------------------------------------------

            future = closes[
                closes.index.normalize()
                > signal_date
            ]

            if future.empty:
                continue

            # ------------------------------------------------
            # +1D / +3D / +5D
            # ------------------------------------------------

            for days in [
                1,
                3,
                5
            ]:

                target_col = (
                    f"price_{days}d"
                )

                return_col = (
                    f"return_{days}d"
                )

                win_col = (
                    f"win_{days}d"
                )

                # 이미 계산됐으면 skip
                if pd.notna(
                    row[target_col]
                ):

                    continue

                # 필요한 거래일 수 부족
                if len(future) < days:

                    continue

                price = _safe_float(
                    future.iloc[
                        days - 1
                    ]
                )

                if (
                    price is None
                    or
                    price <= 0
                ):

                    continue

                ret = (
                    (
                        price
                        / entry_price
                    ) - 1
                ) * 100

                history.at[
                    idx,
                    target_col
                ] = price

                history.at[
                    idx,
                    return_col
                ] = ret

                history.at[
                    idx,
                    win_col
                ] = (
                    1
                    if ret > 0
                    else 0
                )

        except Exception as e:

            print(
                f"⚠️ {ticker} "
                f"signal 업데이트 실패: "
                f"{e}"
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    history["signal_date"] = (
        pd.to_datetime(
            history["signal_date"],
            errors="coerce"
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    history.to_csv(
        SIGNAL_FILE,
        index=False
    )


# ============================================================
# SIGNAL STATS
# ============================================================

def get_signal_stats(
    ticker,
    score,
    direction,
    current_signal_date=None
):

    if not os.path.exists(
        SIGNAL_FILE
    ):

        return None

    try:

        history = pd.read_csv(
            SIGNAL_FILE
        )

    except Exception:

        return None

    if history.empty:
        return None

    required = {

        "signal_date",
        "ticker",
        "score",
        "direction",

        "win_1d",
        "win_3d",
        "win_5d"
    }

    if not required.issubset(
        history.columns
    ):

        return None

    # --------------------------------------------------------
    # 타입 정리
    # --------------------------------------------------------

    history["signal_date"] = pd.to_datetime(
        history["signal_date"],
        errors="coerce"
    )

    history["ticker"] = (
        history["ticker"]
        .astype(str)
        .str.upper()
    )

    history["score"] = pd.to_numeric(
        history["score"],
        errors="coerce"
    )

    ticker = str(
        ticker
    ).upper()

    score = _safe_float(
        score
    )

    if score is None:
        return None

    # --------------------------------------------------------
    # 현재 신호 날짜
    # --------------------------------------------------------

    if current_signal_date is None:

        current_signal_date = pd.Timestamp(
            date.today()
        )

    else:

        current_signal_date = (
            _normalize_date(
                current_signal_date
            )
        )

    # --------------------------------------------------------
    # 과거 유사 신호
    #
    # 같은 종목
    # 같은 방향
    # Score ±5
    # 현재 신호 이전
    # --------------------------------------------------------

    similar = history[
        (
            history["ticker"]
            == ticker
        )
        &
        (
            history["direction"]
            == direction
        )
        &
        (
            history["score"]
            >= score - 5
        )
        &
        (
            history["score"]
            <= score + 5
        )
        &
        (
            history["signal_date"]
            < current_signal_date
        )
    ].copy()

    # --------------------------------------------------------
    # 통계
    # --------------------------------------------------------

    stats = {}

    for days in [
        1,
        3,
        5
    ]:

        col = (
            f"win_{days}d"
        )

        valid = similar[
            similar[col].notna()
        ].copy()

        count = len(
            valid
        )

        if count < 5:

            stats[days] = {

                "count":
                    count,

                "win_rate":
                    None
            }

        else:

            win_rate = (
                pd.to_numeric(
                    valid[col],
                    errors="coerce"
                )
                .dropna()
                .mean()
                * 100
            )

            stats[days] = {

                "count":
                    count,

                "win_rate":
                    float(
                        win_rate
                    )
            }

    total_samples = max(
        [
            stats[d]["count"]
            for d in stats
        ],
        default=0
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if total_samples >= 30:

        confidence = "HIGH"

    elif total_samples >= 10:

        confidence = "MEDIUM"

    elif total_samples >= 5:

        confidence = "LOW"

    else:

        confidence = "INSUFFICIENT"

    return {

        "samples":
            total_samples,

        "stats":
            stats,

        "confidence":
            confidence
    }


# ============================================================
# FORMAT SIGNAL STATS
# ============================================================

def format_signal_stats(
    stats
):

    if not stats:

        return (
            "🧠 <b>SIGNAL BACKTEST</b>\n"
            "아직 과거 유사 신호 데이터가 부족합니다."
        )

    lines = [

        "🧠 <b>SIGNAL BACKTEST</b>"
    ]

    for days in [
        1,
        3,
        5
    ]:

        item = stats[
            "stats"
        ][days]

        if item[
            "win_rate"
        ] is None:

            lines.append(
                f"{days}D 승률: "
                f"데이터 부족 "
                f"({item['count']}회)"
            )

        else:

            lines.append(
                f"{days}D 승률: "
                f"{item['win_rate']:.1f}% "
                f"({item['count']}회)"
            )

    lines.append(
        f"유사 신호 최대 표본: "
        f"{stats['samples']}회"
    )

    lines.append(
        f"신뢰도: "
        f"{stats['confidence']}"
    )

    lines.append(
        "⚠️ 승률은 과거 유사 신호의 "
        "실제 가격 결과 기반"
    )

    return "\n".join(
        lines
    )

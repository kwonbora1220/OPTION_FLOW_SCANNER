"""Compatibility adapter for the unified OI snapshot system.

The authoritative OI storage/calculation lives in option_search.py:
03_RESULTS/daily/oi_history/TICKER_OI_YYYYMMDD.csv

This module intentionally does NOT create or read OI_HISTORY.csv.
It only preserves the existing batch_option_search.py interface while
using option_search.py as the single OI source of truth.
"""

import pandas as pd

from option_search import (
    build_oi_snapshot,
    save_oi_snapshot as _save_oi_snapshot,
)


def _safe_float(value):
    try:
        if pd.isna(value):
            return 0.0

        return float(value)

    except Exception:
        return 0.0


def get_oi_summary(df):

    if df is None:
        return None

    try:
        snapshot = build_oi_snapshot(df)

    except Exception:
        return None

    if snapshot.empty:

        return {
            "call_oi": 0.0,
            "put_oi": 0.0,
            "total_oi": 0.0,
            "call_ratio": 0.0,
            "put_ratio": 0.0
        }

    call_oi = _safe_float(
        snapshot.loc[
            snapshot["option_type"] == "CALL",
            "openInterest"
        ].sum()
    )

    put_oi = _safe_float(
        snapshot.loc[
            snapshot["option_type"] == "PUT",
            "openInterest"
        ].sum()
    )

    total_oi = (
        call_oi
        + put_oi
    )

    call_ratio = (
        call_oi / total_oi
        if total_oi > 0
        else 0.0
    )

    put_ratio = (
        put_oi / total_oi
        if total_oi > 0
        else 0.0
    )

    return {

        "call_oi":
            call_oi,

        "put_oi":
            put_oi,

        "total_oi":
            total_oi,

        "call_ratio":
            call_ratio,

        "put_ratio":
            put_ratio
    }


def save_oi_snapshot(
    ticker,
    df,
    snapshot_date=None
):
    """
    Compatibility wrapper.

    IMPORTANT:
    The actual snapshot is saved by option_search.py.
    This function never writes OI_HISTORY.csv.
    """

    return _save_oi_snapshot(
        ticker,
        df
    )


def _pct_change(
    current_value,
    previous_value
):

    current_value = _safe_float(
        current_value
    )

    previous_value = _safe_float(
        previous_value
    )

    if previous_value == 0:
        return None

    return (
        (
            current_value
            / previous_value
        ) - 1
    ) * 100


def _summary_from_delta_df(df):

    if df is None:
        return None

    if "oi_delta" not in df.columns:
        return None

    try:
        snapshot = build_oi_snapshot(
            df
        )

    except Exception:
        return None

    if snapshot.empty:

        return {
            "call_oi": 0.0,
            "put_oi": 0.0,
            "total_oi": 0.0,
            "call_ratio": 0.0,
            "put_ratio": 0.0
        }

    return get_oi_summary(
        snapshot
    )


def calculate_oi_change(
    ticker,
    current_df,
    current_date=None
):
    """
    Legacy interface used by batch_option_search.py.

    The actual OI comparison is already performed by
    option_search.calculate_oi_delta().

    This function only converts the already-calculated
    dataframe columns into the legacy summary structure.

    It does NOT read OI_HISTORY.csv.
    """

    current = _summary_from_delta_df(
        current_df
    )

    if current is None:
        return None

    available = False

    if (
        "oi_delta_available"
        in current_df.columns
    ):

        try:

            available = bool(
                current_df[
                    "oi_delta_available"
                ]
                .fillna(False)
                .astype(bool)
                .any()
            )

        except Exception:

            available = False

    if not available:

        return {

            "available":
                False,

            "current":
                current,

            "previous":
                None
        }

    try:

        work = current_df.copy()

        work["openInterest"] = (
            pd.to_numeric(
                work["openInterest"],
                errors="coerce"
            )
            .fillna(0)
        )

        work["previous_openInterest"] = (
            pd.to_numeric(
                work[
                    "previous_openInterest"
                ],
                errors="coerce"
            )
            .fillna(0)
        )

        work["oi_delta"] = (
            pd.to_numeric(
                work["oi_delta"],
                errors="coerce"
            )
            .fillna(0)
        )

        work["option_type"] = (
            work["option_type"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        call_oi = float(
            work.loc[
                work["option_type"]
                == "CALL",
                "openInterest"
            ].sum()
        )

        put_oi = float(
            work.loc[
                work["option_type"]
                == "PUT",
                "openInterest"
            ].sum()
        )

        previous_call = float(
            work.loc[
                work["option_type"]
                == "CALL",
                "previous_openInterest"
            ].sum()
        )

        previous_put = float(
            work.loc[
                work["option_type"]
                == "PUT",
                "previous_openInterest"
            ].sum()
        )

        previous_total = (
            previous_call
            + previous_put
        )

        previous_call_ratio = (
            previous_call
            / previous_total
            if previous_total > 0
            else 0.0
        )

        previous_put_ratio = (
            previous_put
            / previous_total
            if previous_total > 0
            else 0.0
        )

        call_change = (
            call_oi
            - previous_call
        )

        put_change = (
            put_oi
            - previous_put
        )

        total_change = (
            call_oi
            + put_oi
            - previous_total
        )

        call_ratio_change = (
            current["call_ratio"]
            - previous_call_ratio
        )

        put_ratio_change = (
            current["put_ratio"]
            - previous_put_ratio
        )

        if (
            call_ratio_change > 0.01
            and put_change < 0
        ):

            structure = (
                "🟢 BULLISH 강화"
            )

        elif (
            call_ratio_change < -0.01
            and put_change > 0
        ):

            structure = (
                "🔴 BEARISH 강화"
            )

        elif (
            call_change > 0
            and put_change > 0
        ):

            structure = (
                "🟡 양쪽 OI 증가"
            )

        elif (
            call_change < 0
            and put_change < 0
        ):

            structure = (
                "🟡 양쪽 OI 감소"
            )

        else:

            structure = (
                "🟡 OI 구조 큰 변화 없음"
            )

        previous = {

            "call_oi":
                previous_call,

            "put_oi":
                previous_put,

            "total_oi":
                previous_total,

            "call_ratio":
                previous_call_ratio,

            "put_ratio":
                previous_put_ratio
        }

        previous_date = None

        if (
            "oi_delta_status"
            in current_df.columns
        ):

            statuses = (
                current_df[
                    "oi_delta_status"
                ]
                .dropna()
                .astype(str)
            )

            if not statuses.empty:

                status = statuses.iloc[0]

                if status.startswith(
                    "COMPARED_WITH_"
                ):

                    previous_date = (
                        status.replace(
                            "COMPARED_WITH_",
                            ""
                        )
                    )

                elif status.startswith(
                    "NO_OI_CHANGE_"
                ):

                    previous_date = (
                        status.replace(
                            "NO_OI_CHANGE_",
                            ""
                        )
                    )

        return {

            "available":
                True,

            "current":
                current,

            "previous":
                previous,

            "previous_date":
                previous_date,

            "call_change":
                call_change,

            "put_change":
                put_change,

            "total_change":
                total_change,

            "call_change_pct":
                _pct_change(
                    current["call_oi"],
                    previous_call
                ),

            "put_change_pct":
                _pct_change(
                    current["put_oi"],
                    previous_put
                ),

            "total_change_pct":
                _pct_change(
                    current["total_oi"],
                    previous_total
                ),

            "call_ratio_change":
                call_ratio_change,

            "put_ratio_change":
                put_ratio_change,

            "structure":
                structure
        }

    except Exception as e:

        print(
            f"⚠️ OI summary 변환 실패: {e}"
        )

        return {

            "available":
                False,

            "current":
                current,

            "previous":
                None
        }


def format_oi_change(
    ticker,
    oi_change
):

    if not oi_change:
        return ""

    if not oi_change.get(
        "available",
        False
    ):

        return (
            "📊 <b>OI STRUCTURE</b>\n"
            "전일 데이터 없음\n"
            "→ 오늘부터 비교 데이터 축적"
        )

    current = (
        oi_change["current"]
    )

    previous = (
        oi_change["previous"]
    )

    call_pct = (
        oi_change.get(
            "call_change_pct"
        )
    )

    put_pct = (
        oi_change.get(
            "put_change_pct"
        )
    )

    total_pct = (
        oi_change.get(
            "total_change_pct"
        )
    )

    call_ratio_delta = (
        oi_change.get(
            "call_ratio_change",
            0
        ) * 100
    )

    structure = (
        oi_change.get(
            "structure",
            "🟡 OI 구조 큰 변화 없음"
        )
    )

    call_pct_text = (
        f"{call_pct:+.1f}%"
        if call_pct is not None
        else "N/A"
    )

    put_pct_text = (
        f"{put_pct:+.1f}%"
        if put_pct is not None
        else "N/A"
    )

    total_pct_text = (
        f"{total_pct:+.1f}%"
        if total_pct is not None
        else "N/A"
    )

    return (
        "📊 <b>OI STRUCTURE</b>\n"

        f"CALL OI "
        f"{previous['call_oi']:,.0f}"
        f" → "
        f"{current['call_oi']:,.0f}"
        f" ({call_pct_text})\n"

        f"PUT OI "
        f"{previous['put_oi']:,.0f}"
        f" → "
        f"{current['put_oi']:,.0f}"
        f" ({put_pct_text})\n"

        f"TOTAL OI "
        f"{previous['total_oi']:,.0f}"
        f" → "
        f"{current['total_oi']:,.0f}"
        f" ({total_pct_text})\n"

        f"Call 비중 변화 "
        f"{call_ratio_delta:+.1f}%p\n"

        f"→ {structure}"
    )

# ============================================================
# OPTION VISUALIZER
# ============================================================
#
# 실제 option_search.py의 analysis["df"]를 이용해서
# TOP 5 종목의 옵션 구조를 PNG로 생성한다.
#
# 표시:
#   - Current Price
#   - Call Wall
#   - Put Wall
#   - Strike별 CALL OI
#   - Strike별 PUT OI
#   - Strike별 GEX
#   - 각 막대의 실제 숫자
#   - IV / GEX / Delta / Vanna
#
# 기존 OPTION SEARCH 계산 로직은 변경하지 않는다.
# ============================================================

import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# SAFE NUMBER
# ============================================================

def _safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


# ============================================================
# COLUMN FINDER
# ============================================================

def _find_column(df, candidates):

    for column in candidates:

        if column in df.columns:

            return column

    return None


# ============================================================
# MONEY FORMAT
# ============================================================

def _format_money(value):

    value = _safe_float(value)

    sign = ""

    if value < 0:

        sign = "-"

        value = abs(value)

    if value >= 1_000_000_000:

        return f"{sign}${value / 1_000_000_000:.2f}B"

    if value >= 1_000_000:

        return f"{sign}${value / 1_000_000:.2f}M"

    if value >= 1_000:

        return f"{sign}${value / 1_000:.1f}K"

    return f"{sign}${value:.0f}"


# ============================================================
# NUMBER FORMAT
# ============================================================

def _format_oi(value):

    value = _safe_float(value)

    if value >= 1_000_000:

        return f"{value / 1_000_000:.2f}M"

    if value >= 1_000:

        return f"{value / 1_000:.1f}K"

    return f"{value:.0f}"


# ============================================================
# BUILD STRIKE DATA
# ============================================================

def _build_strike_data(
    df,
    current_price
):

    if df is None:

        return pd.DataFrame()

    if df.empty:

        return pd.DataFrame()


    data = df.copy()


    # --------------------------------------------------------
    # STRIKE
    # --------------------------------------------------------

    strike_col = _find_column(
        data,
        [
            "strike",
            "Strike",
            "STRIKE"
        ]
    )

    if strike_col is None:

        return pd.DataFrame()


    # --------------------------------------------------------
    # OPTION TYPE
    # --------------------------------------------------------

    type_col = _find_column(
        data,
        [
            "option_type",
            "OptionType",
            "optionType",
            "type",
            "Type",
            "contractType"
        ]
    )


    # --------------------------------------------------------
    # OPEN INTEREST
    # --------------------------------------------------------

    oi_col = _find_column(
        data,
        [
            "openInterest",
            "OpenInterest",
            "open_interest",
            "oi",
            "OI"
        ]
    )


    if oi_col is None:

        return pd.DataFrame()


    # --------------------------------------------------------
    # GEX
    # --------------------------------------------------------

    gex_col = _find_column(
        data,
        [
            "gex",
            "GEX",
            "gex_exposure",
            "GEX_EXPOSURE"
        ]
    )


    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    data["_strike"] = pd.to_numeric(
        data[strike_col],
        errors="coerce"
    )

    data["_oi"] = pd.to_numeric(
        data[oi_col],
        errors="coerce"
    ).fillna(0)


    if gex_col is not None:

        data["_gex"] = pd.to_numeric(
            data[gex_col],
            errors="coerce"
        ).fillna(0)

    else:

        data["_gex"] = 0.0


    data = data[
        data["_strike"].notna()
    ].copy()


    data = data[
        data["_oi"] >= 0
    ].copy()


    if data.empty:

        return pd.DataFrame()


    # --------------------------------------------------------
    # NORMALIZE OPTION TYPE
    # --------------------------------------------------------

    if type_col is not None:

        data["_type"] = (
            data[type_col]
            .astype(str)
            .str.upper()
            .str.strip()
        )

    else:

        # ----------------------------------------------------
        # 일부 데이터에서는 contractSymbol로 판별 가능
        # ----------------------------------------------------

        symbol_col = _find_column(
            data,
            [
                "contractSymbol",
                "ContractSymbol"
            ]
        )

        if symbol_col is not None:

            data["_type"] = (
                data[symbol_col]
                .astype(str)
                .str.upper()
                .str.strip()
                .apply(
                    lambda x:
                    "PUT"
                    if "P" in x[-10:]
                    else "CALL"
                )
            )

        else:

            data["_type"] = ""


    # --------------------------------------------------------
    # TYPE NORMALIZATION
    # --------------------------------------------------------

    data["_is_call"] = (
        data["_type"]
        .str.contains(
            "CALL|^C$",
            regex=True
        )
    )

    data["_is_put"] = (
        data["_type"]
        .str.contains(
            "PUT|^P$",
            regex=True
        )
    )


    # --------------------------------------------------------
    # STRIKE AGGREGATION
    # --------------------------------------------------------

    rows = []


    for strike, group in data.groupby(
        "_strike"
    ):

        call_group = group[
            group["_is_call"]
        ]

        put_group = group[
            group["_is_put"]
        ]


        call_oi = (
            call_group["_oi"]
            .sum()
        )

        put_oi = (
            put_group["_oi"]
            .sum()
        )


        call_gex = (
            call_group["_gex"]
            .sum()
        )

        put_gex = (
            put_group["_gex"]
            .sum()
        )


        rows.append(
            {
                "strike": float(strike),

                "call_oi":
                    float(call_oi),

                "put_oi":
                    float(put_oi),

                "total_oi":
                    float(
                        call_oi
                        + put_oi
                    ),

                "call_gex":
                    float(call_gex),

                "put_gex":
                    float(put_gex),

                "total_gex":
                    float(
                        call_gex
                        + put_gex
                    )
            }
        )


    result = pd.DataFrame(
        rows
    )


    if result.empty:

        return result


    result = result.sort_values(
        "strike"
    ).reset_index(
        drop=True
    )


    # ========================================================
    # RANGE
    # ========================================================

    current_price = _safe_float(
        current_price
    )


    if current_price > 0:

        low = current_price * 0.70

        high = current_price * 1.30

        result = result[
            (
                result["strike"]
                >= low
            )
            &
            (
                result["strike"]
                <= high
            )
        ].copy()


    # --------------------------------------------------------
    # If range removed everything, use all data
    # --------------------------------------------------------

    if result.empty:

        result = pd.DataFrame(
            rows
        )

        result = result.sort_values(
            "strike"
        ).reset_index(
            drop=True
        )


    # ========================================================
    # LIMIT STRIKES
    #
    # 너무 많은 strike가 있으면
    # OI가 가장 큰 구간을 우선한다.
    # ========================================================

    MAX_STRIKES = 18


    if len(result) > MAX_STRIKES:

        result = (
            result
            .sort_values(
                "total_oi",
                ascending=False
            )
            .head(
                MAX_STRIKES
            )
            .sort_values(
                "strike"
            )
            .reset_index(
                drop=True
            )
        )


    return result


# ============================================================
# PRICE MAP
# ============================================================

def _draw_price_map(
    ax,
    current_price,
    call_wall,
    put_wall,
    ticker
):

    current_price = _safe_float(
        current_price
    )


    values = [
        x
        for x in [
            current_price,
            _safe_float(
                call_wall
            )
            if call_wall is not None
            else None,
            _safe_float(
                put_wall
            )
            if put_wall is not None
            else None
        ]
        if x is not None
        and x > 0
    ]


    if not values:

        values = [1]


    min_price = min(values)

    max_price = max(values)


    if min_price == max_price:

        min_price *= 0.9

        max_price *= 1.1


    span = (
        max_price
        - min_price
    )


    padding = max(
        span * 0.18,
        max_price * 0.03
    )


    left = (
        min_price
        - padding
    )

    right = (
        max_price
        + padding
    )


    ax.set_xlim(
        left,
        right
    )

    ax.set_ylim(
        0,
        1
    )


    ax.axis("off")


    # --------------------------------------------------------
    # baseline
    # --------------------------------------------------------

    ax.plot(
        [left, right],
        [0.50, 0.50],
        linewidth=2
    )


    # --------------------------------------------------------
    # PUT WALL
    # --------------------------------------------------------

    if put_wall is not None:

        pw = _safe_float(
            put_wall
        )

        ax.axvline(
            pw,
            ymin=0.25,
            ymax=0.75,
            linewidth=2,
            linestyle="--"
        )

        ax.scatter(
            [pw],
            [0.50],
            s=90,
            zorder=5
        )

        ax.text(
            pw,
            0.82,
            f"PUT WALL\n${pw:g}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold"
        )


    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    if current_price > 0:

        ax.axvline(
            current_price,
            ymin=0.10,
            ymax=0.90,
            linewidth=3
        )

        ax.scatter(
            [current_price],
            [0.50],
            s=130,
            zorder=6
        )

        ax.text(
            current_price,
            0.08,
            f"CURRENT\n${current_price:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold"
        )


    # --------------------------------------------------------
    # CALL WALL
    # --------------------------------------------------------

    if call_wall is not None:

        cw = _safe_float(
            call_wall
        )

        ax.axvline(
            cw,
            ymin=0.25,
            ymax=0.75,
            linewidth=2,
            linestyle="--"
        )

        ax.scatter(
            [cw],
            [0.50],
            s=90,
            zorder=5
        )

        ax.text(
            cw,
            0.82,
            f"CALL WALL\n${cw:g}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold"
        )


    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    ax.text(
        left,
        0.50,
        "PUT",
        ha="right",
        va="center",
        fontsize=9,
        fontweight="bold"
    )

    ax.text(
        right,
        0.50,
        "CALL",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold"
    )


# ============================================================
# OI CHART
# ============================================================

def _draw_oi_chart(
    ax,
    strike_df,
    current_price,
    call_wall,
    put_wall
):

    if strike_df.empty:

        ax.text(
            0.5,
            0.5,
            "OI DATA N/A",
            ha="center",
            va="center"
        )

        ax.axis("off")

        return


    data = strike_df.copy()


    y = np.arange(
        len(data)
    )


    call_values = (
        data["call_oi"]
        .values
    )

    put_values = (
        data["put_oi"]
        .values
    )


    max_oi = max(
        call_values.max()
        if len(call_values)
        else 0,

        put_values.max()
        if len(put_values)
        else 0,

        1
    )


    # --------------------------------------------------------
    # CALL
    # --------------------------------------------------------

    ax.barh(
        y,
        call_values,
        height=0.36,
        alpha=0.85,
        label="CALL OI"
    )


    # --------------------------------------------------------
    # PUT
    # --------------------------------------------------------

    ax.barh(
        y,
        -put_values,
        height=0.36,
        alpha=0.65,
        label="PUT OI"
    )


    # --------------------------------------------------------
    # Y LABEL
    # --------------------------------------------------------

    labels = [
        f"${x:g}"
        for x in data["strike"]
    ]


    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        labels,
        fontsize=8
    )


    # --------------------------------------------------------
    # ZERO
    # --------------------------------------------------------

    ax.axvline(
        0,
        linewidth=1
    )


    # --------------------------------------------------------
    # CURRENT PRICE LINE
    # --------------------------------------------------------

    closest_idx = (
        np.abs(
            data["strike"]
            - current_price
        )
        .argmin()
    )


    ax.axhline(
        closest_idx,
        linewidth=1,
        linestyle=":"
    )


    # --------------------------------------------------------
    # NUMERIC LABELS
    # --------------------------------------------------------

    for idx, value in enumerate(
        call_values
    ):

        if value <= 0:

            continue

        ax.text(
            value
            + max_oi * 0.015,
            idx,
            _format_oi(value),
            va="center",
            ha="left",
            fontsize=7
        )


    for idx, value in enumerate(
        put_values
    ):

        if value <= 0:

            continue

        ax.text(
            -value
            - max_oi * 0.015,
            idx,
            _format_oi(value),
            va="center",
            ha="right",
            fontsize=7
        )


    # --------------------------------------------------------
    # WALL HIGHLIGHT
    # --------------------------------------------------------

    wall_values = []

    if call_wall is not None:

        wall_values.append(
            _safe_float(
                call_wall
            )
        )

    if put_wall is not None:

        wall_values.append(
            _safe_float(
                put_wall
            )
        )


    for wall in wall_values:

        idxs = np.where(
            np.isclose(
                data["strike"].values,
                wall
            )
        )[0]

        for idx in idxs:

            ax.axhline(
                idx,
                linewidth=2,
                linestyle="--"
            )


    ax.set_title(
        "STRIKE OI  |  CALL (+) / PUT (-)",
        fontsize=10,
        fontweight="bold"
    )


    ax.legend(
        loc="lower right",
        fontsize=7,
        frameon=False
    )


    ax.grid(
        axis="x",
        alpha=0.15
    )


    ax.tick_params(
        axis="x",
        labelsize=7
    )


# ============================================================
# GEX CHART
# ============================================================

def _draw_gex_chart(
    ax,
    strike_df
):

    if strike_df.empty:

        ax.text(
            0.5,
            0.5,
            "GEX DATA N/A",
            ha="center",
            va="center"
        )

        ax.axis("off")

        return


    data = strike_df.copy()


    y = np.arange(
        len(data)
    )


    values = (
        data["total_gex"]
        .values
    )


    ax.barh(
        y,
        values,
        height=0.48,
        alpha=0.85
    )


    ax.axvline(
        0,
        linewidth=1
    )


    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        [
            f"${x:g}"
            for x in data["strike"]
        ],
        fontsize=8
    )


    max_abs = max(
        np.max(
            np.abs(values)
        )
        if len(values)
        else 0,

        1
    )


    # --------------------------------------------------------
    # GEX NUMBERS
    # --------------------------------------------------------

    for idx, value in enumerate(
        values
    ):

        if value >= 0:

            x = (
                value
                + max_abs * 0.015
            )

            ha = "left"

        else:

            x = (
                value
                - max_abs * 0.015
            )

            ha = "right"


        ax.text(
            x,
            idx,
            _format_money(value),
            va="center",
            ha=ha,
            fontsize=7
        )


    ax.set_title(
        "GEX BY STRIKE",
        fontsize=10,
        fontweight="bold"
    )


    ax.grid(
        axis="x",
        alpha=0.15
    )


    ax.tick_params(
        axis="x",
        labelsize=7
    )


# ============================================================
# ONE TICKER PANEL
# ============================================================

def _draw_ticker_panel(
    fig,
    outer_gs,
    ticker,
    analysis,
    result,
    rank
):

    current_price = _safe_float(
        analysis.get(
            "current_price",
            0
        )
    )


    df = analysis.get(
        "df"
    )


    greeks = analysis.get(
        "greeks",
        {}
    )


    flow = analysis.get(
        "flow",
        {}
    )


    walls = analysis.get(
        "walls",
        {}
    )


    strike_df = _build_strike_data(
        df,
        current_price
    )


    call_wall = walls.get(
        "call_wall"
    )

    put_wall = walls.get(
        "put_wall"
    )


    score = _safe_float(
        result.get(
            "score",
            0
        )
    )


    direction = result.get(
        "direction",
        "N/A"
    )


    structure = result.get(
        "structure",
        "N/A"
    )


    iv = (
        _safe_float(
            flow.get(
                "atm_iv",
                0
            )
        )
        * 100
    )


    gex = _safe_float(
        greeks.get(
            "GEX",
            0
        )
    )


    delta = _safe_float(
        greeks.get(
            "Delta",
            0
        )
    )


    vanna = _safe_float(
        greeks.get(
            "Vanna",
            0
        )
    )


    # ========================================================
    # GRID
    # ========================================================

    gs = outer_gs.subgridspec(
        3,
        1,
        height_ratios=[
            0.85,
            0.85,
            2.8
        ],
        hspace=0.20
    )


    # ========================================================
    # TITLE
    # ========================================================

    ax_title = fig.add_subplot(
        gs[0]
    )

    ax_title.axis(
        "off"
    )


    ax_title.text(
        0.00,
        0.72,
        f"#{rank}  {ticker}",
        fontsize=15,
        fontweight="bold",
        transform=ax_title.transAxes
    )


    ax_title.text(
        0.25,
        0.72,
        f"${current_price:.2f}",
        fontsize=12,
        transform=ax_title.transAxes
    )


    ax_title.text(
        0.40,
        0.72,
        f"SCORE {score:.1f}",
        fontsize=11,
        fontweight="bold",
        transform=ax_title.transAxes
    )


    ax_title.text(
        0.57,
        0.72,
        direction,
        fontsize=11,
        fontweight="bold",
        transform=ax_title.transAxes
    )


    ax_title.text(
        0.72,
        0.72,
        structure,
        fontsize=9,
        transform=ax_title.transAxes
    )


    ax_title.text(
        0.00,
        0.20,
        (
            f"IV {iv:.1f}%   |   "
            f"GEX {_format_money(gex)}   |   "
            f"Delta {_format_money(delta)}   |   "
            f"Vanna {_format_money(vanna)}"
        ),
        fontsize=9,
        transform=ax_title.transAxes
    )


    # ========================================================
    # PRICE MAP
    # ========================================================

    ax_price = fig.add_subplot(
        gs[1]
    )


    _draw_price_map(
        ax_price,
        current_price,
        call_wall,
        put_wall,
        ticker
    )


    # ========================================================
    # BOTTOM
    # ========================================================

    bottom = gs[2].subgridspec(
        1,
        2,
        width_ratios=[
            1.25,
            1
        ],
        wspace=0.20
    )


    # ========================================================
    # OI
    # ========================================================

    ax_oi = fig.add_subplot(
        bottom[0]
    )


    _draw_oi_chart(
        ax_oi,
        strike_df,
        current_price,
        call_wall,
        put_wall
    )


    # ========================================================
    # GEX
    # ========================================================

    ax_gex = fig.add_subplot(
        bottom[1]
    )


    _draw_gex_chart(
        ax_gex,
        strike_df
    )


    # ========================================================
    # PANEL BORDER
    # ========================================================

    rect = Rectangle(
        (
            0,
            0
        ),
        1,
        1,
        transform=fig.transFigure,
        fill=False,
        linewidth=0.8,
        alpha=0.25
    )


# ============================================================
# MAIN VISUALIZER
# ============================================================

def create_final_top5_option_image(
    analysis_results,
    top_results,
    output_path,
    top_n=5
):

    if not top_results:

        raise ValueError(
            "top_results is empty"
        )


    selected = top_results[
        :top_n
    ]


    selected = [
        r
        for r in selected
        if r.get("ticker")
        in analysis_results
    ]


    if not selected:

        raise ValueError(
            "시각화 가능한 analysis 결과가 없습니다."
        )


    # ========================================================
    # FIGURE
    # ========================================================

    rows = len(
        selected
    )


    fig = plt.figure(
        figsize=(
            18,
            5.6 * rows
        )
    )


    outer = fig.add_gridspec(
        rows,
        1,
        hspace=0.42
    )


    # ========================================================
    # MAIN TITLE
    # ========================================================

    fig.suptitle(
        "🔥 FINAL TOP OPTION MAP",
        fontsize=20,
        fontweight="bold",
        y=0.995
    )


    # ========================================================
    # PANELS
    # ========================================================

    for rank, result in enumerate(
        selected,
        1
    ):

        ticker = result[
            "ticker"
        ]


        analysis = analysis_results[
            ticker
        ]


        _draw_ticker_panel(
            fig=fig,
            outer_gs=outer[rank - 1],
            ticker=ticker,
            analysis=analysis,
            result=result,
            rank=rank
        )


    # ========================================================
    # FOOTER
    # ========================================================

    fig.text(
        0.50,
        0.006,
        (
            "⚠️ OI = Yahoo Finance snapshot | "
            "GEX = OI 기반 Dealer Positioning Proxy | "
            "실제 Buy/Sell 방향은 확인할 수 없음"
        ),
        ha="center",
        fontsize=8
    )


    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True
    )


    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white"
    )


    plt.close(
        fig
    )


    print("")
    print(
        "🖼️ OPTION VISUALIZER 완료"
    )

    print(
        f"📁 {output_path}"
    )


    return output_path

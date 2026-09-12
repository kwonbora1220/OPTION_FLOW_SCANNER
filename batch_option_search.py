import os
import time

import pandas as pd
import requests

from selected_symbols import SELECTED_SYMBOLS

from option_search import (
    analyze_ticker,
    send_telegram
)

from option_visualizer import (
    create_final_top5_option_image
)

from oi_history import (
    calculate_oi_change,
    format_oi_change
)

from signal_history import (
    record_signal,
    update_signal_results,
    get_signal_stats,
    format_signal_stats
)


# ============================================================
# CONFIG
# ============================================================

TOP_ENTRY = 5

ENTRY_SCORE = 70
WATCH_SCORE = 45


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


RANKING_FILE = os.path.join(
    RESULT_DIR,
    "OPTION_FINAL_RANKING.csv"
)


# ============================================================
# VISUAL FILE
# ============================================================

FINAL_OPTION_IMAGE = os.path.join(
    RESULT_DIR,
    "FINAL_TOP5_OPTION.png"
)


# ============================================================
# FORMAT MONEY
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

        return (
            f"{sign}${x / 1_000_000_000:.2f}B"
        )


    if x >= 1_000_000:

        return (
            f"{sign}${x / 1_000_000:.2f}M"
        )


    if x >= 1_000:

        return (
            f"{sign}${x / 1_000:.1f}K"
        )


    return (
        f"{sign}${x:.0f}"
    )


# ============================================================
# SCORE
# ============================================================

def calculate_score(
    analysis
):

    df = analysis["df"]

    greeks = analysis["greeks"]

    flow = analysis["flow"]

    walls = analysis["walls"]

    quality = analysis["quality"]

    current_price = analysis[
        "current_price"
    ]

    score = 50.0

    reasons = []

    bullish_signals = 0
    bearish_signals = 0


    # ========================================================
    # 1. CALL / PUT VOLUME
    # ========================================================

    volume_ratio = (
        flow.get(
            "call_volume_ratio",
            0.5
        )
    )

    if volume_ratio >= 0.60:

        score += 6

        bullish_signals += 1

        reasons.append(
            "Call 거래량 우세"
        )

    elif volume_ratio >= 0.55:

        score += 3

        bullish_signals += 1

        reasons.append(
            "Call 거래량 소폭 우세"
        )

    elif volume_ratio <= 0.40:

        score -= 6

        bearish_signals += 1

        reasons.append(
            "Put 거래량 우세"
        )

    elif volume_ratio <= 0.45:

        score -= 3

        bearish_signals += 1

        reasons.append(
            "Put 거래량 소폭 우세"
        )


    # ========================================================
    # 2. TRADED PREMIUM PROXY
    # ========================================================

    premium_ratio = (
        flow.get(
            "call_premium_ratio",
            0.5
        )
    )

    if premium_ratio >= 0.60:

        score += 5

        bullish_signals += 1

        reasons.append(
            "Call 거래대금 Proxy 우세"
        )

    elif premium_ratio >= 0.55:

        score += 2

        reasons.append(
            "Call 거래대금 Proxy 소폭 우세"
        )

    elif premium_ratio <= 0.40:

        score -= 5

        bearish_signals += 1

        reasons.append(
            "Put 거래대금 Proxy 우세"
        )

    elif premium_ratio <= 0.45:

        score -= 2

        reasons.append(
            "Put 거래대금 Proxy 소폭 우세"
        )


    # ========================================================
    # 3. DELTA EXPOSURE PROXY
    # ========================================================

    delta = float(
        greeks.get(
            "Delta",
            0
        )
    )

    if delta > 0:

        score += 7

        bullish_signals += 1

        reasons.append(
            "OI Delta Exposure Proxy 상방"
        )

    elif delta < 0:

        score -= 7

        bearish_signals += 1

        reasons.append(
            "OI Delta Exposure Proxy 하방"
        )


    # ========================================================
    # 4. VANNA
    # ========================================================

    vanna = float(
        greeks.get(
            "Vanna",
            0
        )
    )

    if vanna > 0:

        score += 3

        bullish_signals += 1

        reasons.append(
            "Vanna 상방"
        )

    elif vanna < 0:

        score -= 3

        bearish_signals += 1

        reasons.append(
            "Vanna 하방"
        )


    # ========================================================
    # 5. HIRO-LIKE PROXY
    # ========================================================

    hiro = float(
        greeks.get(
            "HIRO",
            0
        )
    )

    if hiro > 0:

        score += 2

        bullish_signals += 1

        reasons.append(
            "체결방향 Flow Proxy 상방"
        )

    elif hiro < 0:

        score -= 2

        bearish_signals += 1

        reasons.append(
            "체결방향 Flow Proxy 하방"
        )


    # ========================================================
    # 6. GEX
    # ========================================================

    gex = float(
        greeks.get(
            "GEX",
            0
        )
    )

    if gex > 0:

        reasons.append(
            "Positive GEX Regime"
        )

    elif gex < 0:

        reasons.append(
            "Negative GEX Regime"
        )


    # ========================================================
    # 7. ATM IV
    # ========================================================

    atm_iv = float(
        flow.get(
            "atm_iv",
            0
        )
    )

    iv_pct = (
        atm_iv * 100
    )

    if iv_pct <= 40:

        score += 3

        reasons.append(
            "ATM IV 낮음"
        )

    elif iv_pct <= 70:

        score += 1

        reasons.append(
            "ATM IV 적정"
        )

    elif iv_pct <= 100:

        score -= 2

        reasons.append(
            "ATM IV 높음"
        )

    elif iv_pct <= 150:

        score -= 5

        reasons.append(
            "ATM IV 과열"
        )

    else:

        score -= 8

        reasons.append(
            "ATM IV 극단적"
        )


    # ========================================================
    # 8. DTE
    # ========================================================

    dte = flow.get(
        "dte_buckets",
        {}
    )

    if dte.get(
        "31_60",
        0
    ) > 0:

        score += 2

        reasons.append(
            "31~60DTE 구조 존재"
        )

    if dte.get(
        "61_180",
        0
    ) > 0:

        score += 1

        reasons.append(
            "61~180DTE 구조 존재"
        )


    # ========================================================
    # 9. WALL STRUCTURE
    #
    # 핵심:
    # - Put Wall은 강한 하방 지지 후보
    # - 현재가가 Put Wall에 가까울수록 매수 매력 증가
    # - Put Wall을 이미 하향 이탈하면 강한 감점
    # - Call Wall이 가까우면 상방 여유 부족으로 감점
    # ========================================================

    call_wall = walls.get(
        "call_wall"
    )

    put_wall = walls.get(
        "put_wall"
    )

    call_distance = None
    put_distance = None

    put_wall_support_score = 0.0
    upside_room = None


    # ========================================================
    # 9-1. PUT WALL
    # ========================================================

    if (
        put_wall is not None
        and current_price
        and current_price > 0
    ):

        put_distance = (
            (
                current_price
                - put_wall
            )
            / current_price
            * 100
        )

        # ----------------------------------------------------
        # Put Wall 하향 이탈
        # ----------------------------------------------------

        if put_distance < 0:

            score -= 12

            bearish_signals += 1

            reasons.append(
                "⚠️ Put Wall 하향 이탈"
            )


        # ----------------------------------------------------
        # Put Wall 바로 위
        # 가장 강한 매수 후보 구간
        # ----------------------------------------------------

        elif put_distance <= 2:

            score += 12

            put_wall_support_score = 12

            bullish_signals += 1

            reasons.append(
                "🟢 Put Wall 초근접 지지"
            )


        # ----------------------------------------------------
        # Put Wall 2~4%
        # ----------------------------------------------------

        elif put_distance <= 4:

            score += 9

            put_wall_support_score = 9

            bullish_signals += 1

            reasons.append(
                "🟢 Put Wall 근접"
            )


        # ----------------------------------------------------
        # Put Wall 4~6%
        # ----------------------------------------------------

        elif put_distance <= 6:

            score += 6

            put_wall_support_score = 6

            bullish_signals += 1

            reasons.append(
                "Put Wall 지지권"
            )


        # ----------------------------------------------------
        # Put Wall 6~10%
        # ----------------------------------------------------

        elif put_distance <= 10:

            score += 2

            put_wall_support_score = 2

            reasons.append(
                "Put Wall 하방 완충"
            )


        # ----------------------------------------------------
        # 너무 멀리 떨어짐
        # ----------------------------------------------------

        else:

            put_wall_support_score = 0

            reasons.append(
                "Put Wall과 거리 있음"
            )


    # ========================================================
    # 9-2. CALL WALL
    # ========================================================

    if (
        call_wall is not None
        and current_price
        and current_price > 0
    ):

        call_distance = (
            (
                call_wall
                - current_price
            )
            / current_price
            * 100
        )

        # ----------------------------------------------------
        # 현재가 위에 Call Wall이 있는 정상 구조
        # ----------------------------------------------------

        if call_distance >= 0:

            upside_room = call_distance

            # Call Wall 바로 아래
            # 상승 여유가 거의 없으므로 감점

            if call_distance <= 3:

                score -= 7

                reasons.append(
                    "⚠️ Call Wall 바로 아래"
                )

            elif call_distance <= 6:

                score -= 3

                reasons.append(
                    "Call Wall 근접"
                )

            elif call_distance >= 10:

                score += 3

                reasons.append(
                    "🟢 Call Wall 상방 여유"
                )

            else:

                reasons.append(
                    "Call Wall 상방 여유"
                )


        # ----------------------------------------------------
        # 현재가가 Call Wall을 이미 돌파한 경우
        # ----------------------------------------------------

        else:

            upside_room = 0

            score += 2

            reasons.append(
                "🟢 Call Wall 돌파"
            )


    # ========================================================
    # 9-3. PUT WALL → CALL WALL 구조
    #
    # Put Wall은 가까우면서
    # Call Wall은 충분히 멀리 있는 종목을 우선
    # ========================================================

    if (
        put_distance is not None
        and call_distance is not None
        and put_distance >= 0
        and call_distance > 0
    ):

        # 좋은 구조:
        # Put Wall <= 6%
        # Call Wall >= 8%

        if (
            put_distance <= 6
            and call_distance >= 8
        ):

            score += 5

            reasons.append(
                "🟢 Put Wall 지지 + 상방 여유"
            )


        # 매우 좋은 구조:
        # Put Wall <= 4%
        # Call Wall >= 10%

        if (
            put_distance <= 4
            and call_distance >= 10
        ):

            score += 4

            reasons.append(
                "🔥 강한 매수 구조"
            )


    # ========================================================
    # 9-4. PUT WALL 이탈 + CALL WALL 근접
    #
    # 가장 피해야 하는 구조
    # ========================================================

    if (
        put_distance is not None
        and call_distance is not None
    ):

        if (
            put_distance < 0
            and call_distance <= 5
        ):

            score -= 8

            reasons.append(
                "🔴 Put Wall 이탈 + 상방 여유 부족"
            )

    # ========================================================
    # 10. SIGNAL CONFLICT
    # ========================================================

    if (
        bullish_signals >= 3
        and bearish_signals >= 2
    ):

        score -= 7

        reasons.append(
            "⚠️ Signal Conflict"
        )

    elif (
        bearish_signals >= 3
        and bullish_signals >= 2
    ):

        reasons.append(
            "⚠️ Bearish Signal Conflict"
        )


    # ========================================================
    # 11. DATA QUALITY
    # ========================================================

    quality_score = float(
        quality.get(
            "score",
            0
        )
    )

    if quality_score < 40:

        score -= 5

        reasons.append(
            "⚠️ 낮은 데이터 품질"
        )

    elif quality_score >= 75:

        reasons.append(
            "데이터 품질 양호"
        )


    # ========================================================
    # LIMIT
    # ========================================================

    score = max(
        0,
        min(
            100,
            score
        )
    )


    # ========================================================
    # DIRECTION
    # ========================================================

    if (
        bullish_signals
        >= bearish_signals + 2
    ):

        direction = "BULLISH"

    elif (
        bearish_signals
        >= bullish_signals + 2
    ):

        direction = "BEARISH"

    else:

        direction = "NEUTRAL"


    # ========================================================
    # STRUCTURE
    # ========================================================

    if gex > 0:

        structure = "STABLE_GEX"

    elif gex < 0:

        structure = "HIGH_VOL_GEX"

    else:

        structure = "NEUTRAL_GEX"


    # ========================================================
    # FINAL ACTION
    # ========================================================

    if (
        score >= ENTRY_SCORE
        and direction == "BULLISH"
        and iv_pct < 150
        and quality_score >= 40
    ):

        category = "🟢 오늘 진입 후보"

    elif (
        score <= 35
        or direction == "BEARISH"
    ):

        category = "🔴 회피"

    else:

        category = "🟡 관망"


    return {

        "score":
            score,

        "direction":
            direction,

        "structure":
            structure,

        "category":
            category,

        "reasons":
            reasons,

        "call_distance":
            call_distance,

        "put_distance":
            put_distance,

        "put_wall_support_score":
            put_wall_support_score,

        "upside_room":
            upside_room,

        "iv_pct":
            iv_pct,

        "bullish_signals":
            bullish_signals,

        "bearish_signals":
            bearish_signals
    }


# ============================================================
# FINAL RESULT
# ============================================================

def make_final_result(
    analysis
):

    score_data = calculate_score(
        analysis
    )

    greeks = analysis[
        "greeks"
    ]

    flow = analysis[
        "flow"
    ]

    walls = analysis[
        "walls"
    ]

    quality = analysis[
        "quality"
    ]


    return {

        "ticker":
            analysis["ticker"],

        "current_price":
            analysis["current_price"],

        "score":
            score_data["score"],

        "direction":
            score_data["direction"],

        "structure":
            score_data["structure"],

        "category":
            score_data["category"],

        "reasons":
            score_data["reasons"],

        "delta":
            greeks.get(
                "Delta",
                0
            ),

        "gex":
            greeks.get(
                "GEX",
                0
            ),

        "vanna":
            greeks.get(
                "Vanna",
                0
            ),

        "charm":
            greeks.get(
                "Charm",
                0
            ),

        "vega":
            greeks.get(
                "Vega",
                0
            ),

        "hiro":
            greeks.get(
                "HIRO",
                0
            ),

        "atm_iv":
            flow.get(
                "atm_iv",
                0
            ),

        "call_volume_ratio":
            flow.get(
                "call_volume_ratio",
                0.5
            ),

        "call_oi_ratio":
            flow.get(
                "call_oi_ratio",
                0.5
            ),

        "call_premium_ratio":
            flow.get(
                "call_premium_ratio",
                0.5
            ),

        "call_wall":
            walls.get(
                "call_wall"
            ),

        "put_wall":
            walls.get(
                "put_wall"
            ),

        "call_wall_gex":
            walls.get(
                "call_wall_gex",
                0
            ),

        "put_wall_gex":
            walls.get(
                "put_wall_gex",
                0
            ),

        "call_wall_distance":
            score_data.get(
                "call_distance"
            ),

        "put_wall_distance":
            score_data.get(
                "put_distance"
            ),

        "put_wall_support_score":
            score_data.get(
                "put_wall_support_score"
            ),

        "upside_room":
            score_data.get(
                "upside_room"
            ),

        "quality":
            quality.get(
                "score",
                0
            ),

        "bullish_signals":
            score_data[
                "bullish_signals"
            ],

        "bearish_signals":
            score_data[
                "bearish_signals"
            ],

        "oi_change":
            analysis.get(
                "oi_change"
            ),

        "signal_stats":
            None
    }


# ============================================================
# SAVE FINAL CSV
# ============================================================

def save_ranking(
    results
):

    rows = []

    for r in results:

        oi = r.get(
            "oi_change"
        )

        row = {

            "ticker":
                r["ticker"],

            "current_price":
                r["current_price"],

            "score":
                r["score"],

            "direction":
                r["direction"],

            "structure":
                r["structure"],

            "category":
                r["category"],

            "reasons":
                " | ".join(
                    r["reasons"]
                ),

            "delta":
                r["delta"],

            "gex":
                r["gex"],

            "vanna":
                r["vanna"],

            "charm":
                r["charm"],

            "vega":
                r["vega"],

            "hiro":
                r["hiro"],

            "atm_iv":
                r["atm_iv"],

            "call_volume_ratio":
                r["call_volume_ratio"],

            "call_oi_ratio":
                r["call_oi_ratio"],

            "call_premium_ratio":
                r["call_premium_ratio"],

                        "call_wall": 
                r["call_wall"], 

            "put_wall": 
                r["put_wall"], 

            "call_wall_distance":
                r.get(
                    "call_wall_distance"
                ),

            "put_wall_distance":
                r.get(
                    "put_wall_distance"
                ),

            "put_wall_support_score":
                r.get(
                    "put_wall_support_score"
                ),

            "upside_room":
                r.get(
                    "upside_room"
                ),

            "call_wall_gex": 
                r["call_wall_gex"], 

            "put_wall_gex": 
                r["put_wall_gex"],

            "data_quality":
                r["quality"],

            "bullish_signals":
                r["bullish_signals"],

            "bearish_signals":
                r["bearish_signals"]
        }


        # ====================================================
        # OI CSV
        # ====================================================

        if oi:

            current = oi.get(
                "current",
                {}
            )

            row["current_call_oi"] = (
                current.get(
                    "call_oi",
                    0
                )
            )

            row["current_put_oi"] = (
                current.get(
                    "put_oi",
                    0
                )
            )

            row["current_total_oi"] = (
                current.get(
                    "total_oi",
                    0
                )
            )

            row["call_oi_change"] = (
                oi.get(
                    "call_change"
                )
            )

            row["put_oi_change"] = (
                oi.get(
                    "put_change"
                )
            )

            row["total_oi_change"] = (
                oi.get(
                    "total_change"
                )
            )

            row["call_oi_change_pct"] = (
                oi.get(
                    "call_change_pct"
                )
            )

            row["put_oi_change_pct"] = (
                oi.get(
                    "put_change_pct"
                )
            )

            row["total_oi_change_pct"] = (
                oi.get(
                    "total_change_pct"
                )
            )

            row["call_ratio_change"] = (
                oi.get(
                    "call_ratio_change"
                )
            )

        else:

            row["current_call_oi"] = None
            row["current_put_oi"] = None
            row["current_total_oi"] = None
            row["call_oi_change"] = None
            row["put_oi_change"] = None
            row["total_oi_change"] = None
            row["call_oi_change_pct"] = None
            row["put_oi_change_pct"] = None
            row["total_oi_change_pct"] = None
            row["call_ratio_change"] = None


        # ====================================================
        # BACKTEST CSV
        # ====================================================

        stats = r.get(
            "signal_stats"
        )

        if stats:

            row["backtest_samples"] = (
                stats.get(
                    "samples"
                )
            )

            row["backtest_confidence"] = (
                stats.get(
                    "confidence"
                )
            )

            for days in [1, 3, 5]:

                item = stats[
                    "stats"
                ][days]

                row[
                    f"backtest_{days}d_win_rate"
                ] = item.get(
                    "win_rate"
                )

                row[
                    f"backtest_{days}d_samples"
                ] = item.get(
                    "count"
                )

        else:

            row["backtest_samples"] = None

            row["backtest_confidence"] = None

            for days in [1, 3, 5]:

                row[
                    f"backtest_{days}d_win_rate"
                ] = None

                row[
                    f"backtest_{days}d_samples"
                ] = None


        rows.append(
            row
        )


    df = pd.DataFrame(
        rows
    )


    df.to_csv(
        RANKING_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    print("")

    print(
        "💾 FINAL CSV 저장:"
    )

    print(
        RANKING_FILE
    )

    print(
        f"✅ 최종 종목 수: "
        f"{len(df)}"
    )


# ============================================================
# OI SUMMARY
# ============================================================

def get_oi_message(
    ticker,
    oi_change
):

    if not oi_change:

        return ""


    try:

        return format_oi_change(
            ticker,
            oi_change
        )

    except Exception as e:

        print(
            f"⚠️ {ticker} "
            f"OI 메시지 생성 실패: {e}"
        )

        return ""


# ============================================================
# BACKTEST SUMMARY
# ============================================================

def get_backtest_message(
    ticker,
    stats
):

    if not stats:

        return (
            "🧠 <b>SIGNAL BACKTEST</b>\n"
            "아직 과거 유사 신호 데이터가 부족합니다."
        )


    try:

        return format_signal_stats(
            stats
        )

    except Exception as e:

        print(
            f"⚠️ {ticker} "
            f"Backtest 메시지 생성 실패: {e}"
        )

        return (
            "🧠 <b>SIGNAL BACKTEST</b>\n"
            "통계 계산 실패"
        )


# ============================================================
# FINAL TELEGRAM MESSAGE
# ============================================================

def build_final_message(
    results
):

    lines = []


    # ========================================================
    # HEADER
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🧠 <b>오늘의 PORTFOLIO "
        "OPTION RANKING</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")


    # ========================================================
    # ENTRY
    # ========================================================

    entry = [
        r
        for r in results
        if r["category"]
        == "🟢 오늘 진입 후보"
    ][:TOP_ENTRY]


    lines.append(
        "🟢 <b>진입 후보 TOP 5</b>"
    )

    lines.append("")


    if entry:

        for i, r in enumerate(
            entry,
            1
        ):

            lines.append(
                f"<b>{i}. {r['ticker']}</b> "
                f"| {r['score']:.1f}점 "
                f"| {r['direction']}"
            )

            lines.append(
                "   → "
                + ", ".join(
                    r["reasons"][:4]
                )
            )

            lines.append("")

    else:

        lines.append(
            "오늘 진입 후보 없음"
        )

        lines.append("")


    # ========================================================
    # WATCH
    # ========================================================

    watch = [
        r
        for r in results
        if r["category"]
        == "🟡 관망"
    ]


    lines.append(
        "🟡 <b>관망</b>"
    )

    lines.append("")


    if watch:

        for r in watch:

            lines.append(
                f"• {r['ticker']} "
                f"| {r['score']:.1f} "
                f"| {r['direction']} "
                f"| {r['structure']}"
            )

    else:

        lines.append(
            "관망 종목 없음"
        )


    lines.append("")


    # ========================================================
    # AVOID
    # ========================================================

    avoid = [
        r
        for r in results
        if r["category"]
        == "🔴 회피"
    ]


    lines.append(
        "🔴 <b>회피</b>"
    )

    lines.append("")


    if avoid:

        for r in avoid:

            lines.append(
                f"• {r['ticker']} "
                f"| {r['score']:.1f} "
                f"| {r['direction']}"
            )

    else:

        lines.append(
            "회피 종목 없음"
        )


    lines.append("")


    # ========================================================
    # ALL RANKING
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📊 <b>전체 종목 순위</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")


    for i, r in enumerate(
        results,
        1
    ):

        lines.append(
            f"{i}. <b>{r['ticker']}</b> "
            f"| {r['score']:.1f} "
            f"| {r['direction']} "
            f"| {r['category']}"
        )


    lines.append("")


    # ========================================================
    # TOP DETAIL
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🎯 <b>TOP 5 구조 상세</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")


    for r in results[:TOP_ENTRY]:

        lines.append(
            f"📌 <b>{r['ticker']}</b> "
            f"${r['current_price']:.2f}"
        )

        lines.append(
            f"🎯 Score "
            f"{r['score']:.1f} "
            f"| {r['direction']}"
        )

        lines.append(
            f"🏗 Structure "
            f"{r['structure']}"
        )


        try:

            iv_text = (
                f"{float(r['atm_iv']) * 100:.1f}%"
            )

        except Exception:

            iv_text = "N/A"


        lines.append(
            f"IV {iv_text}"
        )


        lines.append(
            f"GEX "
            f"{format_money(r['gex'])}"
        )

        lines.append(
            f"Delta Proxy "
            f"{format_money(r['delta'])}"
        )

        lines.append(
            f"Vanna "
            f"{format_money(r['vanna'])}"
        )


        if r["call_wall"] is not None:

            lines.append(
                f"📈 Call Wall "
                f"${r['call_wall']:g}"
            )

        else:

            lines.append(
                "📈 Call Wall N/A"
            )


        if r["put_wall"] is not None:

            lines.append(
                f"📉 Put Wall "
                f"${r['put_wall']:g}"
            )

        else:

            lines.append(
                "📉 Put Wall N/A"
            )


        oi_message = get_oi_message(
            r["ticker"],
            r.get("oi_change")
        )


        if oi_message:

            lines.append("")

            lines.extend(
                oi_message.splitlines()
            )


        if (
            r["category"]
            == "🟢 오늘 진입 후보"
        ):

            stats = r.get(
                "signal_stats"
            )

            lines.append("")

            lines.extend(
                get_backtest_message(
                    r["ticker"],
                    stats
                ).splitlines()
            )


        lines.append("")


    # ========================================================
    # DISCLAIMER
    # ========================================================

    lines.append(
        "⚠️ GEX / Delta / Vanna는 "
        "OI 기반 Proxy입니다."
    )

    lines.append(
        "⚠️ 거래대금은 실제 Buy/Sell Flow가 아닙니다."
    )

    lines.append(
        "⚠️ 무료 yfinance 데이터에는 "
        "실제 체결 방향 정보가 없습니다."
    )

    lines.append(
        "⚠️ 승률은 과거 유사 진입 신호의 "
        "실제 가격 결과 기반입니다."
    )

    lines.append(
        "⚠️ 과거 표본이 부족하면 "
        "승률을 표시하지 않습니다."
    )


    return "\n".join(
        lines
    )


# ============================================================
# TELEGRAM PHOTO
# ============================================================

def send_telegram_photo(
    image_path,
    caption=""
):

    # --------------------------------------------------------
    # 환경변수
    #
    # 기존 프로젝트에서 사용하는 이름과
    # 일반적인 이름 둘 다 지원
    # --------------------------------------------------------

    bot_token = (
        os.getenv(
            "TELEGRAM_BOT_TOKEN"
        )
        or
        os.getenv(
            "BOT_TOKEN"
        )
    )


    chat_id = (
        os.getenv(
            "TELEGRAM_CHAT_ID"
        )
        or
        os.getenv(
            "CHAT_ID"
        )
    )


    if not bot_token:

        print(
            "⚠️ TELEGRAM_BOT_TOKEN 없음"
        )

        return False


    if not chat_id:

        print(
            "⚠️ TELEGRAM_CHAT_ID 없음"
        )

        return False


    if not os.path.exists(
        image_path
    ):

        print(
            f"⚠️ 이미지 없음: "
            f"{image_path}"
        )

        return False


    url = (
        f"https://api.telegram.org/"
        f"bot{bot_token}/sendPhoto"
    )


    try:

        with open(
            image_path,
            "rb"
        ) as photo:

            response = requests.post(
                url,
                data={
                    "chat_id":
                        chat_id,

                    "caption":
                        caption,

                    "parse_mode":
                        "HTML"
                },
                files={
                    "photo": (
                        os.path.basename(
                            image_path
                        ),
                        photo,
                        "image/png"
                    )
                },
                timeout=60
            )


        if response.ok:

            print(
                "✅ Telegram 옵션 이미지 전송 완료"
            )

            return True


        print(
            "⚠️ Telegram 이미지 전송 실패:"
        )

        print(
            response.text[:1000]
        )

        return False


    except Exception as e:

        print(
            f"⚠️ Telegram 이미지 전송 오류: "
            f"{type(e).__name__}: {e}"
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("")

    print("=" * 70)

    print(
        "🔥 PORTFOLIO OPTION SCANNER V3"
    )

    print("=" * 70)

    print("")


    print(
        f"📊 분석 종목: "
        f"{len(SELECTED_SYMBOLS)}개"
    )

    print("")


    print(
        "📌 처리 방식:"
    )

    print(
        "1. 종목별 OPTION SEARCH"
    )

    print(
        "2. 종목별 CSV"
    )

    print(
        "3. 종목별 Telegram"
    )

    print(
        "4. OI 전일 대비 변화"
    )

    print(
        "5. Direction / Structure / IV"
    )

    print(
        "6. FINAL SCORE"
    )

    print(
        "7. SIGNAL BACKTEST"
    )

    print(
        "8. 전체 Ranking"
    )

    print(
        "9. TOP 5 실제 옵션 시각화"
    )

    print(
        "10. 최종 Telegram"
    )

    print(
        "11. TOP 5 옵션 이미지 Telegram"
    )

    print("")


    # ========================================================
    # UPDATE OLD SIGNAL RESULTS
    # ========================================================

    print(
        "📈 과거 진입 신호 결과 업데이트..."
    )


    try:

        update_signal_results()

        print(
            "✅ SIGNAL_HISTORY 업데이트 완료"
        )

    except Exception as e:

        print(
            f"⚠️ SIGNAL_HISTORY 업데이트 실패: {e}"
        )


    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    processed_tickers = set()

    total = len(
        SELECTED_SYMBOLS
    )


    # ========================================================
    # IMPORTANT
    #
    # FINAL RESULT에는 df가 없기 때문에
    # 실제 analysis 객체를 별도로 보관한다.
    # ========================================================

    analysis_results = {}


    # ========================================================
    # EACH TICKER
    # ========================================================

    for i, ticker in enumerate(
        SELECTED_SYMBOLS,
        1
    ):

        ticker = (
            ticker
            .upper()
            .strip()
        )


        # ====================================================
        # DUPLICATE PROTECTION
        # ====================================================

        if ticker in processed_tickers:

            print(
                f"⚠️ {ticker} 중복 → 스킵"
            )

            continue


        processed_tickers.add(
            ticker
        )


        print("")

        print("=" * 70)

        print(
            f"🔥 {i}/{total} {ticker}"
        )

        print("=" * 70)


        # ====================================================
        # ANALYZE
        # ====================================================

        try:

            analysis = analyze_ticker(
                ticker
            )


            if analysis is None:

                print(
                    f"❌ {ticker} 분석 실패"
                )

                continue


            # =================================================
            # ★ IMPORTANT
            #
            # 실제 옵션 dataframe을 보존한다.
            # 나중에 TOP 5 시각화에서 사용.
            # =================================================

            analysis_results[
                ticker
            ] = analysis


            # =================================================
            # OI CHANGE
            # =================================================

            try:

                oi_change = (
                    calculate_oi_change(
                        ticker,
                        analysis["df"]
                    )
                )


                analysis[
                    "oi_change"
                ] = oi_change


                print(
                    f"📊 {ticker} "
                    f"OI 비교 완료"
                )


            except Exception as e:

                print(
                    f"⚠️ {ticker} "
                    f"OI 분석 실패: {e}"
                )


                analysis[
                    "oi_change"
                ] = None


            # =================================================
            # FINAL SCORE
            # =================================================

            final_result = (
                make_final_result(
                    analysis
                )
            )


            # =================================================
            # RECORD SIGNAL
            # =================================================

            if (
                final_result["category"]
                == "🟢 오늘 진입 후보"
            ):

                try:

                    record_signal(
                        ticker=ticker,

                        score=final_result[
                            "score"
                        ],

                        direction=final_result[
                            "direction"
                        ],

                        category=final_result[
                            "category"
                        ],

                        current_price=final_result[
                            "current_price"
                        ]
                    )


                    print(
                        f"📌 {ticker} "
                        f"진입 신호 기록"
                    )


                except Exception as e:

                    print(
                        f"⚠️ {ticker} "
                        f"신호 기록 실패: {e}"
                    )


            # =================================================
            # BACKTEST
            # =================================================

            try:

                stats = get_signal_stats(
                    ticker=ticker,

                    score=final_result[
                        "score"
                    ],

                    direction=final_result[
                        "direction"
                    ]
                )


                final_result[
                    "signal_stats"
                ] = stats


            except Exception as e:

                print(
                    f"⚠️ {ticker} "
                    f"Backtest 계산 실패: {e}"
                )


                final_result[
                    "signal_stats"
                ] = None


            # =================================================
            # ADD RESULT
            # =================================================

            results.append(
                final_result
            )


            print("")

            print(
                f"🎯 {ticker} "
                f"SCORE: "
                f"{final_result['score']:.1f}"
            )

            print(
                f"   방향: "
                f"{final_result['direction']}"
            )

            print(
                f"   구조: "
                f"{final_result['structure']}"
            )

            print(
                f"   판정: "
                f"{final_result['category']}"
            )


        except Exception as e:

            print("")

            print(
                f"❌ {ticker} 분석 실패"
            )

            print(
                f"   {type(e).__name__}: {e}"
            )


        # ====================================================
        # DELAY
        # ====================================================

        if i < total:

            print("")

            print(
                "⏳ 다음 종목 준비..."
            )

            time.sleep(3)


    # ========================================================
    # CHECK
    # ========================================================

    print("")

    print("=" * 70)

    print(
        "📊 ALL OPTION SEARCH FINISHED"
    )

    print("=" * 70)

    print("")


    print(
        f"✅ 분석 완료: "
        f"{len(results)}개"
    )


    if not results:

        print(
            "❌ 분석 결과가 없습니다."
        )

        raise SystemExit(1)


    # ========================================================
    # SORT
    # ========================================================

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )


    # ========================================================
    # SAVE
    # ========================================================

    save_ranking(
        results
    )


    # ========================================================
    # ★ FINAL TOP 5 OPTION VISUALIZATION
    # ========================================================

    print("")

    print("=" * 70)

    print(
        "🖼️ FINAL TOP 5 OPTION VISUALIZATION"
    )

    print("=" * 70)

    print("")


    top_results = results[
        :TOP_ENTRY
    ]


    visual_ok = False


    try:

        create_final_top5_option_image(
            analysis_results=analysis_results,

            top_results=top_results,

            output_path=FINAL_OPTION_IMAGE,

            top_n=TOP_ENTRY
        )


        visual_ok = (
            os.path.exists(
                FINAL_OPTION_IMAGE
            )
        )


        if visual_ok:

            print(
                "✅ TOP 5 옵션 이미지 생성 완료"
            )

            print(
                f"📁 {FINAL_OPTION_IMAGE}"
            )


    except Exception as e:

        print(
            "❌ TOP 5 옵션 이미지 생성 실패"
        )

        print(
            f"   {type(e).__name__}: {e}"
        )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    final_message = (
        build_final_message(
            results
        )
    )


    print("")

    print("=" * 70)

    print(
        "🧠 FINAL PORTFOLIO RANKING"
    )

    print("=" * 70)

    print("")


    print(
        final_message
    )


    # ========================================================
    # FINAL TELEGRAM TEXT
    # ========================================================

    print("")

    print("=" * 70)

    print(
        "📱 FINAL TELEGRAM"
    )

    print("=" * 70)


    telegram_ok = send_telegram(
        final_message
    )


    if telegram_ok:

        print(
            "✅ 최종 Telegram 전송 완료"
        )

    else:

        print(
            "⚠️ 최종 Telegram 전송 실패"
        )


    # ========================================================
    # TELEGRAM IMAGE
    # ========================================================

    if visual_ok:

        print("")

        print("=" * 70)

        print(
            "🖼️ TELEGRAM OPTION MAP"
        )

        print("=" * 70)


        caption_lines = [
            "🔥 <b>FINAL TOP 5 OPTION MAP</b>",
            "",
        ]


        for i, r in enumerate(
            top_results,
            1
        ):

            ticker = r[
                "ticker"
            ]

            price = r[
                "current_price"
            ]

            score = r[
                "score"
            ]

            direction = r[
                "direction"
            ]

            caption_lines.append(
                f"{i}. <b>{ticker}</b> "
                f"${price:.2f} "
                f"| {score:.1f} "
                f"| {direction}"
            )


        caption = "\n".join(
            caption_lines
        )


        photo_ok = send_telegram_photo(
            image_path=FINAL_OPTION_IMAGE,
            caption=caption
        )


        if photo_ok:

            print(
                "📱 FINAL TOP 5 옵션 이미지 전송 완료"
            )

        else:

            print(
                "⚠️ FINAL TOP 5 옵션 이미지 전송 실패"
            )


    else:

        print(
            "⚠️ 옵션 이미지가 없어 "
            "Telegram 사진 전송을 건너뜁니다."
        )


    # ========================================================
    # DONE
    # ========================================================

    print("")

    print("=" * 70)

    print(
        "🔥 PORTFOLIO OPTION SCANNER V3 COMPLETE"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

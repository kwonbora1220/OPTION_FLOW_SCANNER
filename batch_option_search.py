
import time
import os
import pandas as pd

from selected_symbols import SELECTED_SYMBOLS
from option_search import (
    get_current_price,
    get_option_data,
    calculate_option_metrics,
    calculate_aggregate_greeks,
)


# ============================================================
# CONFIG
# ============================================================

TOP_ENTRY = 5

ENTRY_SCORE = 70
WATCH_SCORE = 40

# ============================================================
# FORMAT
# ============================================================

def fmt_money(x):

    try:

        x = float(x)

        sign = "-" if x < 0 else ""
        x = abs(x)

        if x >= 1_000_000:
            return f"{sign}${x / 1_000_000:.2f}M"

        if x >= 1_000:
            return f"{sign}${x / 1_000:.1f}K"

        return f"{sign}${x:.0f}"

    except Exception:

        return "$0"


# ============================================================
# WALL CALCULATION
# ============================================================

def find_walls(df, current_price):

    active = df[
        df["volume"] > 0
    ].copy()

    if active.empty:

        return None, None

    # --------------------------------------------------------
    # 현재가 기준 ±30% 이내
    # --------------------------------------------------------

    active["distance_pct"] = (
        abs(
            active["strike"] - current_price
        )
        / current_price
        * 100
    )

    active = active[
        active["distance_pct"] <= 30
    ].copy()

    if active.empty:

        return None, None

    # --------------------------------------------------------
    # Wall은 현재가 기준으로 위/아래를 분리한다.
    #
    # CALL WALL
    # → 현재가보다 높은 Call Strike
    #
    # PUT WALL
    # → 현재가보다 낮은 Put Strike
    # --------------------------------------------------------

    calls = active[
        (active["option_type"] == "CALL")
        &
        (active["strike"] > current_price)
    ].copy()

    puts = active[
        (active["option_type"] == "PUT")
        &
        (active["strike"] < current_price)
    ].copy()

    call_wall = None
    put_wall = None

    # --------------------------------------------------------
    # CALL WALL
    # --------------------------------------------------------

    if not calls.empty:

        calls["wall_strength"] = (
            calls["GEX"].abs()
        )

        call_group = (
            calls
            .groupby("strike")["wall_strength"]
            .sum()
        )

        if not call_group.empty:

            call_wall = float(
                call_group.idxmax()
            )

    # --------------------------------------------------------
    # PUT WALL
    # --------------------------------------------------------

    if not puts.empty:

        puts["wall_strength"] = (
            puts["GEX"].abs()
        )

        put_group = (
            puts
            .groupby("strike")["wall_strength"]
            .sum()
        )

        if not put_group.empty:

            put_wall = float(
                put_group.idxmax()
            )

    return (
        call_wall,
        put_wall
    )


# ============================================================
# SCORE
# ============================================================

def calculate_score(
    df,
    greeks,
    current_price
):

    score = 50.0

    reasons = []

    calls = df[
        df["option_type"] == "CALL"
    ].copy()

    puts = df[
        df["option_type"] == "PUT"
    ].copy()

    if calls.empty or puts.empty:

        return (
            score,
            "NEUTRAL",
            reasons,
            None,
            None
        )

    # ========================================================
    # PREMIUM
    # ========================================================

    call_premium = calls[
        "premium_flow"
    ].sum()

    put_premium = puts[
        "premium_flow"
    ].sum()

    total_premium = (
        call_premium
        + put_premium
    )

    if total_premium > 0:

        call_ratio = (
            call_premium
            / total_premium
        )

        if call_ratio >= 0.60:

            score += 10

            reasons.append(
                "Call Premium 강세"
            )

        elif call_ratio >= 0.55:

            score += 6

            reasons.append(
                "Call Premium 우세"
            )

        elif call_ratio <= 0.40:

            score -= 10

            reasons.append(
                "Put Premium 강세"
            )

        elif call_ratio <= 0.45:

            score -= 6

            reasons.append(
                "Put Premium 우세"
            )

    # ========================================================
    # VOLUME
    # ========================================================

    call_volume = calls[
        "volume"
    ].sum()

    put_volume = puts[
        "volume"
    ].sum()

    total_volume = (
        call_volume
        + put_volume
    )

    if total_volume > 0:

        call_ratio = (
            call_volume
            / total_volume
        )

        if call_ratio >= 0.60:

            score += 8

            reasons.append(
                "Call 거래량 우세"
            )

        elif call_ratio <= 0.40:

            score -= 8

            reasons.append(
                "Put 거래량 우세"
            )

    # ========================================================
    # OI
    # ========================================================

    call_oi = calls[
        "openInterest"
    ].sum()

    put_oi = puts[
        "openInterest"
    ].sum()

    total_oi = (
        call_oi
        + put_oi
    )

    if total_oi > 0:

        call_ratio = (
            call_oi
            / total_oi
        )

        if call_ratio >= 0.60:

            score += 6

            reasons.append(
                "Call OI 우세"
            )

        elif call_ratio <= 0.40:

            score -= 6

            reasons.append(
                "Put OI 우세"
            )

    # ========================================================
    # DTE QUALITY
    # ========================================================

    dte_0_7 = df[
        (df["DTE"] >= 0)
        &
        (df["DTE"] <= 7)
    ]

    dte_8_30 = df[
        (df["DTE"] >= 8)
        &
        (df["DTE"] <= 30)
    ]

    dte_31_60 = df[
        (df["DTE"] >= 31)
        &
        (df["DTE"] <= 60)
    ]

    dte_61_180 = df[
        (df["DTE"] >= 61)
        &
        (df["DTE"] <= 180)
    ]

    # 너무 짧은 옵션만 존재하면 감점
    if (
        not dte_0_7.empty
        and dte_8_30.empty
        and dte_31_60.empty
        and dte_61_180.empty
    ):

        score -= 6

        reasons.append(
            "초단기 DTE 집중"
        )

    # 8~30일
    if not dte_8_30.empty:

        score += 2

        reasons.append(
            "8~30DTE 구조"
        )

    # 31~60일
    if not dte_31_60.empty:

        score += 4

        reasons.append(
            "31~60DTE 구조"
        )

    # 61~180일
    if not dte_61_180.empty:

        score += 2

        reasons.append(
            "61~180DTE 구조"
        )

    # 30D+ Call 구조
    long_calls = calls[
        calls["DTE"] >= 30
    ]

    if not long_calls.empty:

        score += 3

        reasons.append(
            "30D+ Call 구조 존재"
        )

    # 30D+ Put은 무조건 큰 감점하지 않고
    # 위험요인으로만 약하게 반영
    long_puts = puts[
        puts["DTE"] >= 30
    ]

    if not long_puts.empty:

        reasons.append(
            "30D+ Put 구조 존재"
        )

    # ========================================================
    # DELTA EXPOSURE
    # ========================================================

    delta = greeks.get(
        "Delta",
        0
    )

    delta_abs = abs(delta)

    if delta_abs > 5_000_000:

        if delta > 0:

            score += 10

            reasons.append(
                "Delta Exposure 강한 상방"
            )

        else:

            score -= 10

            reasons.append(
                "Delta Exposure 강한 하방"
            )

    elif delta > 0:

        score += 6

        reasons.append(
            "Delta Exposure 상방"
        )

    elif delta < 0:

        score -= 6

        reasons.append(
            "Delta Exposure 하방"
        )

    # ========================================================
    # GEX
    # ========================================================

    gex = greeks.get(
        "GEX",
        0
    )

    if gex > 0:

        score += 4

        reasons.append(
            "GEX Positive"
        )

    elif gex < 0:

        score -= 4

        reasons.append(
            "GEX Negative"
        )

    # ========================================================
    # HIRO
    # ========================================================

    hiro = greeks.get(
        "HIRO",
        0
    )

    if hiro > 0:

        score += 4

        reasons.append(
            "HIRO Proxy Positive"
        )

    elif hiro < 0:

        score -= 4

        reasons.append(
            "HIRO Proxy Negative"
        )

    # ========================================================
    # VANNA
    # ========================================================

    vanna = greeks.get(
        "Vanna",
        0
    )

    if vanna > 0:

        score += 3

        reasons.append(
            "Vanna 상방"
        )

    elif vanna < 0:

        score -= 3

        reasons.append(
            "Vanna 하방"
        )

    # ========================================================
    # WALL
    # ========================================================

    call_wall, put_wall = find_walls(
        df,
        current_price
    )

    # --------------------------------------------------------
    # CALL WALL DISTANCE
    # --------------------------------------------------------

    call_distance = None

    if call_wall is not None:

        call_distance = (
            call_wall
            - current_price
        ) / current_price * 100

        if call_distance >= 8:

            score += 4

            reasons.append(
                f"Call Wall 여유 +{call_distance:.1f}%"
            )

        elif call_distance >= 4:

            score += 2

            reasons.append(
                f"Call Wall +{call_distance:.1f}%"
            )

        elif call_distance < 2:

            score -= 5

            reasons.append(
                "Call Wall 근접"
            )

    # --------------------------------------------------------
    # PUT WALL DISTANCE
    # --------------------------------------------------------

    put_distance = None

    if put_wall is not None:

        put_distance = (
            current_price
            - put_wall
        ) / current_price * 100

        if put_distance <= 3:

            score += 4

            reasons.append(
                f"Put Wall 지지 -{put_distance:.1f}%"
            )

        elif put_distance <= 6:

            score += 2

            reasons.append(
                f"Put Wall -{put_distance:.1f}%"
            )

    # ========================================================
    # WALL SPACE
    # ========================================================

    if (
        call_distance is not None
        and put_distance is not None
    ):

        total_space = (
            call_distance
            + put_distance
        )

        # 벽이 너무 붙어 있음
        if total_space < 5:

            score -= 8

            reasons.append(
                "Wall 사이 공간 협소"
            )

        # 공간이 넓음
        elif total_space >= 12:

            score += 3

            reasons.append(
                "Wall 공간 양호"
            )

    # ========================================================
    # IV QUALITY
    # ========================================================

    iv = greeks.get(
        "IV",
        0
    )

    iv_pct = (
        float(iv)
        * 100
    )

    if iv_pct <= 40:

        score += 4

        reasons.append(
            "IV 낮음"
        )

    elif iv_pct <= 70:

        score += 2

        reasons.append(
            "IV 적정"
        )

    elif iv_pct <= 100:

        score -= 2

        reasons.append(
            "IV 높음"
        )

    elif iv_pct <= 150:

        score -= 5

        reasons.append(
            "IV 과열"
        )

    else:

        score -= 8

        reasons.append(
            "IV 극단적"
        )

    # ========================================================
    # SIGNAL CONFLICT
    # ========================================================

    bullish_signals = 0
    bearish_signals = 0

    # Delta
    if delta > 0:
        bullish_signals += 1

    elif delta < 0:
        bearish_signals += 1

    # GEX
    if gex > 0:
        bullish_signals += 1

    elif gex < 0:
        bearish_signals += 1

    # HIRO
    if hiro > 0:
        bullish_signals += 1

    elif hiro < 0:
        bearish_signals += 1

    # Vanna
    if vanna > 0:
        bullish_signals += 1

    elif vanna < 0:
        bearish_signals += 1

    # --------------------------------------------------------
    # 3 : 1 충돌
    # --------------------------------------------------------

    if (
        bullish_signals >= 3
        and bearish_signals >= 1
    ):

        score -= 8

        reasons.append(
            "⚠️ Signal Conflict"
        )

    elif (
        bearish_signals >= 3
        and bullish_signals >= 1
    ):

        score += 0

        reasons.append(
            "⚠️ Signal Conflict"
        )

    # ========================================================
    # SCORE LIMIT
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

    if score >= 70:

        direction = "BULLISH"

    elif score >= 55:

        direction = "SLIGHT BULLISH"

    elif score >= 45:

        direction = "NEUTRAL"

    elif score >= 30:

        direction = "SLIGHT BEARISH"

    else:

        direction = "BEARISH"

    return (
        score,
        direction,
        reasons,
        call_wall,
        put_wall
    )


# ============================================================
# CATEGORY
# ============================================================

def classify(score):

    if score >= ENTRY_SCORE:

        return "🟢 오늘 진입 후보"

    if score >= WATCH_SCORE:

        return "🟡 관망"

    return "🔴 회피"


# ============================================================
# START
# ============================================================

print(
    "=" * 70
)

print(
    "🔥 FIXED OPTION SEARCH"
)

print(
    "=" * 70
)

print(
    f"📊 검색 종목: "
    f"{len(SELECTED_SYMBOLS)}개"
)

print("")

results = []


# ============================================================
# SEARCH ALL SYMBOLS
# ============================================================

for i, ticker in enumerate(
    SELECTED_SYMBOLS,
    1
):

    ticker = ticker.upper().strip()

    print(
        "=" * 70
    )

    print(
        f"🔥 {i}/{len(SELECTED_SYMBOLS)} : {ticker}"
    )

    print("")

    try:

        # ----------------------------------------------------
        # CURRENT PRICE
        # ----------------------------------------------------

        current_price = get_current_price(
            ticker
        )

        if current_price is None:

            print(
                f"⏭️ {ticker} 건너뜀"
            )

            continue

        # ----------------------------------------------------
        # OPTION DATA
        # ----------------------------------------------------

        df = get_option_data(
            ticker
        )

        if df.empty:

            print(
                f"⏭️ {ticker} 옵션 데이터 없음"
            )

            continue

        # ----------------------------------------------------
        # GREEKS
        # ----------------------------------------------------

        df = calculate_option_metrics(
            df,
            current_price
        )

        greeks = calculate_aggregate_greeks(
            df
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        (
            score,
            direction,
            reasons,
            call_wall,
            put_wall
        ) = calculate_score(
            df,
            greeks,
            current_price
        )

        category = classify(
            score
        )

        results.append(
            {
                "ticker": ticker,

                "current_price":
                    current_price,

                "score":
                    score,

                "direction":
                    direction,

                "category":
                    category,

                "reasons":
                    reasons,

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

                "hiro":
                    greeks.get(
                        "HIRO",
                        0
                    ),

                "vanna":
                    greeks.get(
                        "Vanna",
                        0
                    ),

                "iv":
                    greeks.get(
                        "IV",
                        0
                    ),

                "call_wall":
                    call_wall,

                "put_wall":
                    put_wall,
            }
        )

        print("")

        print(
            f"📊 {ticker}"
        )

        print(
            f"현재가: "
            f"${current_price:.2f}"
        )

        print(
            f"점수: "
            f"{score:.1f}"
        )

        print(
            f"방향: "
            f"{direction}"
        )

        print(
            f"판정: "
            f"{category}"
        )

        if call_wall is not None:

            print(
                f"📈 Call Wall: "
                f"${call_wall:g}"
            )

        else:

            print(
                "📈 Call Wall: N/A"
            )

        if put_wall is not None:

            print(
                f"📉 Put Wall: "
                f"${put_wall:g}"
            )

        else:

            print(
                "📉 Put Wall: N/A"
            )

        if reasons:

            print(
                "→ "
                + ", ".join(
                    reasons
                )
            )

    except Exception as e:

        print("")

        print(
            f"❌ {ticker} 검색 오류"
        )

        print(
            f"   {e}"
        )

    if i < len(
        SELECTED_SYMBOLS
    ):

        print("")

        print(
            "⏳ 다음 종목 준비..."
        )

        time.sleep(3)


# ============================================================
# FINAL RANKING
# ============================================================

if not results:

    print("")

    print(
        "❌ 분석 가능한 종목이 없습니다."
    )

    raise SystemExit


results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)


# ============================================================
# TELEGRAM FINAL MESSAGE
# ============================================================

lines = []

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append(
    "🧠 오늘의 OPTION FINAL RANKING"
)

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append("")


# ============================================================
# TOP 5
# ============================================================

entry = [
    x
    for x in results
    if x["score"] >= ENTRY_SCORE
][:TOP_ENTRY]

lines.append(
    "🟢 오늘 진입 후보 TOP 5"
)

lines.append("")

if entry:

    for i, r in enumerate(
        entry,
        1
    ):

        lines.append(
            f"{i}. "
            f"{r['ticker']} | "
            f"{r['score']:.1f}점 | "
            f"{r['direction']}"
        )

        if r["reasons"]:

            lines.append(
                "   → "
                + ", ".join(
                    r["reasons"]
                )
            )

else:

    lines.append(
        "오늘 진입 후보 없음"
    )


lines.append("")


# ============================================================
# WATCH
# ============================================================

watch = [
    x
    for x in results
    if WATCH_SCORE
    <= x["score"]
    < ENTRY_SCORE
]

lines.append(
    "🟡 관망"
)

lines.append("")

if watch:

    for r in watch:

        lines.append(
            f"• {r['ticker']} | "
            f"{r['score']:.1f}점 | "
            f"{r['direction']}"
        )

        if r["reasons"]:

            lines.append(
                "→ "
                + ", ".join(
                    r["reasons"]
                )
            )

else:

    lines.append(
        "관망 종목 없음"
    )


lines.append("")


# ============================================================
# AVOID
# ============================================================

avoid = [
    x
    for x in results
    if x["score"] < WATCH_SCORE
]

lines.append(
    "🔴 회피"
)

lines.append("")

if avoid:

    for r in avoid:

        lines.append(
            f"• {r['ticker']} | "
            f"{r['score']:.1f}점 | "
            f"{r['direction']}"
        )

        if r["reasons"]:

            lines.append(
                "→ "
                + ", ".join(
                    r["reasons"]
                )
            )

else:

    lines.append(
        "회피 종목 없음"
    )


lines.append("")


# ============================================================
# ALL SCORES
# ============================================================

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append(
    "📊 전체 종목 점수"
)

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append("")

for i, r in enumerate(
    results,
    1
):

    lines.append(
        f"{i}. "
        f"{r['ticker']:<6} "
        f"{r['score']:>5.1f}점 "
        f"{r['category']}"
    )


lines.append("")


# ============================================================
# TOP 5 DETAIL
# ============================================================

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append(
    "🎯 TOP 5 구조 상세"
)

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append("")

if entry:

    for r in entry:

        lines.append(
            f"📌 {r['ticker']} "
            f"${r['current_price']:.2f}"
        )

        if r["call_wall"] is not None:

            call_distance = (
                r["call_wall"]
                - r["current_price"]
            ) / r["current_price"] * 100

            lines.append(
                f"📈 Call Wall "
                f"${r['call_wall']:g} "
                f"(+{call_distance:.1f}%)"
            )

        else:

            lines.append(
                "📈 Call Wall N/A"
            )

        if r["put_wall"] is not None:

            put_distance = (
                r["current_price"]
                - r["put_wall"]
            ) / r["current_price"] * 100

            lines.append(
                f"📉 Put Wall "
                f"${r['put_wall']:g} "
                f"(-{put_distance:.1f}%)"
            )

        else:

            lines.append(
                "📉 Put Wall N/A"
            )

        lines.append(
            f"IV "
            f"{r['iv'] * 100:.1f}%"
        )

        lines.append("")


# ============================================================
# SCORE COMPONENTS
# ============================================================

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append(
    "📊 최종 스코어 구성"
)

lines.append(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

lines.append("")

lines.append(
    "Volume / OI / Premium"
)

lines.append(
    "- DTE Quality"
)

lines.append(
    "- Delta Exposure"
)

lines.append(
    "- GEX"
)

lines.append(
    "- HIRO Proxy"
)

lines.append(
    "- Vanna Exposure"
)

lines.append(
    "- Call Wall / Put Wall Position"
)

lines.append(
    "- Wall Distance"
)

lines.append(
    "- Wall Space"
)

lines.append(
    "- IV Quality"
)

lines.append(
    "- Signal Conflict Penalty"
)

lines.append("")

lines.append(
    "⚠️ Delta/Gamma/Vanna/Charm은 "
    "Black-Scholes 기반 계산값입니다."
)

lines.append(
    "⚠️ GEX는 OI × Gamma 기반 "
    "계산값입니다."
)

lines.append(
    "⚠️ Wall은 현재가 기준으로 "
    "상방 Call / 하방 Put을 분리하여 계산합니다."
)

lines.append(
    "⚠️ HIRO는 실제 체결 방향 데이터가 "
    "없는 yfinance 환경의 Proxy입니다."
)

lines.append(
    "⚠️ 옵션 거래량만으로 실제 BUY/SELL을 "
    "확정할 수 없습니다."
)


final_message = "\n".join(
    lines
)


# ============================================================
# PRINT
# ============================================================

print("")

print(
    "=" * 70
)

print(
    final_message
)

print(
    "=" * 70
)


# ============================================================
# TELEGRAM
# ============================================================

try:

    from option_search import send_telegram

    send_telegram(
        final_message
    )

except Exception as e:

    print("")

    print(
        f"❌ 최종 Telegram 전송 오류: {e}"
    )


# ============================================================
# CSV
# ============================================================

out_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(
            os.path.abspath(
                __file__
            )
        ),
        "..",
        "03_RESULTS",
        "daily"
    )
)

os.makedirs(
    out_dir,
    exist_ok=True
)


ranking_file = os.path.join(
    out_dir,
    "OPTION_FINAL_RANKING.csv"
)


ranking_rows = []

for r in results:

    ranking_rows.append(
        {
            "ticker":
                r["ticker"],

            "current_price":
                r["current_price"],

            "score":
                r["score"],

            "direction":
                r["direction"],

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

            "hiro":
                r["hiro"],

            "vanna":
                r["vanna"],

            "iv":
                r["iv"],

            "call_wall":
                r["call_wall"],

            "put_wall":
                r["put_wall"],
        }
    )


pd.DataFrame(
    ranking_rows
).to_csv(
    ranking_file,
    index=False,
    encoding="utf-8-sig"
)


print("")

print(
    f"💾 최종 순위 저장: "
    f"{ranking_file}"
)

print("")

print(
    "=" * 70
)

print(
    "🔥 FIXED OPTION SEARCH 완료"
)

print(
    "=" * 70
)


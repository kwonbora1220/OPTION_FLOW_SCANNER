import os
import time
import math
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from option_flow_filter import (
    calculate_premium_flow,
    calculate_dte,
    calculate_basic_flow,
    dte_quality,
    classify
)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "01_DATA"
)

MARKET_DIR = os.path.join(
    DATA_DIR,
    "market"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "03_RESULTS",
    "daily"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


MAX_WORKERS = 8
MAX_UNIVERSE = 120
FLOW_PRESELECT = 20
TOP_ENTRY = 5

MIN_DTE = 0
MAX_DTE = 180

MIN_OPTION_VOLUME = 100
MIN_OPEN_INTEREST = 100

ENTRY_SCORE = 70
WATCH_SCORE = 40

TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    ""
)

TELEGRAM_CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID",
    ""
)

TELEGRAM_API_URL = (
    "https://api.telegram.org/bot{}/sendMessage"
)

TELEGRAM_MAX_LENGTH = 4000


def fmt_money(x):
    try:
        x = float(x)

        sign = "-" if x < 0 else ""
        x = abs(x)

        if x >= 1_000_000_000:
            return f"{sign}${x / 1_000_000_000:.2f}B"

        if x >= 1_000_000:
            return f"{sign}${x / 1_000_000:.2f}M"

        if x >= 1_000:
            return f"{sign}${x / 1_000:.1f}K"

        return f"{sign}${x:.0f}"

    except Exception:
        return "$0"


def load_latest_universe():

    if not os.path.exists(MARKET_DIR):
        return pd.DataFrame()

    files = []

    for filename in os.listdir(MARKET_DIR):

        if (
            filename.startswith("AUTO_UNIVERSE_")
            and filename.endswith(".csv")
        ):
            files.append(
                os.path.join(
                    MARKET_DIR,
                    filename
                )
            )

    if not files:
        return pd.DataFrame()

    files.sort(
        key=os.path.getmtime,
        reverse=True
    )

    latest = files[0]

    print("")
    print("Universe:")
    print(latest)

    try:
        df = pd.read_csv(
            latest,
            encoding="utf-8-sig"
        )
    except Exception:
        df = pd.read_csv(latest)

    if "Ticker" not in df.columns:
        return pd.DataFrame()

    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df = df.drop_duplicates(
        subset=["Ticker"]
    )

    return df.head(MAX_UNIVERSE)


def get_current_price(ticker):

    try:

        stock = yf.Ticker(ticker)

        hist = stock.history(
            period="5d",
            interval="1d",
            auto_adjust=False
        )

        if hist.empty:
            return None

        close = hist["Close"].dropna()

        if close.empty:
            return None

        return float(close.iloc[-1])

    except Exception:
        return None


def get_option_data(ticker):

    try:

        stock = yf.Ticker(ticker)

        expirations = stock.options

        if not expirations:
            return pd.DataFrame()

        today = pd.Timestamp.now().normalize()

        selected = []

        for exp in expirations:

            try:

                exp_date = pd.Timestamp(exp)

                dte = (
                    exp_date - today
                ).days

                if MIN_DTE <= dte <= MAX_DTE:
                    selected.append(exp)

            except Exception:
                continue

        if not selected:
            return pd.DataFrame()

        rows = []

        for expiration in selected:

            try:

                chain = stock.option_chain(
                    expiration
                )

                calls = chain.calls.copy()
                puts = chain.puts.copy()

                calls["option_type"] = "CALL"
                puts["option_type"] = "PUT"

                calls["expiration"] = expiration
                puts["expiration"] = expiration

                rows.append(calls)
                rows.append(puts)

            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        return pd.concat(
            rows,
            ignore_index=True
        )

    except Exception:
        return pd.DataFrame()


def calculate_walls(
    df,
    current_price
):

    calls = df[
        df["option_type"] == "CALL"
    ].copy()

    puts = df[
        df["option_type"] == "PUT"
    ].copy()

    call_wall = None
    put_wall = None

    calls_above = calls[
        calls["strike"] > current_price
    ].copy()

    if not calls_above.empty:

        calls_above["wall_strength"] = (
            calls_above["openInterest"]
            * calls_above["volume"].clip(lower=1)
            * calls_above["mid_price"].clip(lower=0.01)
        )

        grouped = (
            calls_above
            .groupby("strike")["wall_strength"]
            .sum()
        )

        if not grouped.empty:
            call_wall = float(grouped.idxmax())

    puts_below = puts[
        puts["strike"] < current_price
    ].copy()

    if not puts_below.empty:

        puts_below["wall_strength"] = (
            puts_below["openInterest"]
            * puts_below["volume"].clip(lower=1)
            * puts_below["mid_price"].clip(lower=0.01)
        )

        grouped = (
            puts_below
            .groupby("strike")["wall_strength"]
            .sum()
        )

        if not grouped.empty:
            put_wall = float(grouped.idxmax())

    call_distance = None
    put_distance = None

    if call_wall is not None:
        call_distance = (
            (call_wall - current_price)
            / current_price
            * 100
        )

    if put_wall is not None:
        put_distance = (
            (put_wall - current_price)
            / current_price
            * 100
        )

    return {
        "call_wall": call_wall,
        "put_wall": put_wall,
        "call_distance": call_distance,
        "put_distance": put_distance
    }


def calculate_iv_quality(df):

    iv = pd.to_numeric(
        df.get(
            "impliedVolatility",
            pd.Series(dtype=float)
        ),
        errors="coerce"
    )

    iv = iv.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if iv.empty:
        return 0, "IV 데이터 없음", None

    median_iv = float(iv.median()) * 100

    if median_iv < 40:
        return 5, "IV 낮음", median_iv

    if median_iv < 80:
        return 3, "IV 적정", median_iv

    if median_iv < 120:
        return -3, "IV 과열", median_iv

    return -7, "IV 극단적", median_iv


def calculate_greeks(
    df,
    current_price
):

    calls = df[
        df["option_type"] == "CALL"
    ]

    puts = df[
        df["option_type"] == "PUT"
    ]

    delta = 0.0
    hiro = 0.0
    gex = 0.0
    vanna = 0.0

    for _, row in calls.iterrows():

        oi = float(row.get("openInterest", 0) or 0)
        volume = float(row.get("volume", 0) or 0)

        strike = float(
            row.get(
                "strike",
                current_price
            )
        )

        moneyness = strike / current_price

        if moneyness < 0.95:
            d = 0.80
        elif moneyness < 1.02:
            d = 0.50
        else:
            d = 0.20

        delta += (
            oi * d * 100 * current_price
        )

        hiro += (
            volume * d * 100 * current_price
        )

    for _, row in puts.iterrows():

        oi = float(row.get("openInterest", 0) or 0)
        volume = float(row.get("volume", 0) or 0)

        strike = float(
            row.get(
                "strike",
                current_price
            )
        )

        moneyness = strike / current_price

        if moneyness > 1.05:
            d = -0.80
        elif moneyness > 0.98:
            d = -0.50
        else:
            d = -0.20

        delta += (
            oi * d * 100 * current_price
        )

        hiro += (
            volume * d * 100 * current_price
        )

    for _, row in df.iterrows():

        oi = float(row.get("openInterest", 0) or 0)

        iv = float(
            row.get(
                "impliedVolatility",
                0
            ) or 0
        )

        if iv <= 0:
            iv = 0.50

        strike = float(
            row.get(
                "strike",
                current_price
            )
        )

        distance = (
            abs(strike - current_price)
            / current_price
        )

        gamma = (
            math.exp(-distance * 10)
            / (
                current_price
                * iv
                * 2
            )
        )

        sign = (
            1
            if row["option_type"] == "CALL"
            else -1
        )

        gex += (
            oi
            * gamma
            * 100
            * current_price
            * sign
        )

        relative_distance = (
            strike - current_price
        ) / current_price

        v = (
            oi
            * relative_distance
            * iv
        )

        if row["option_type"] == "CALL":
            vanna += v
        else:
            vanna -= v

    return {
        "Delta": delta,
        "GEX": gex,
        "HIRO": hiro,
        "Vanna": vanna
    }


def calculate_final_score(
    df,
    current_price
):

    score = 50.0
    reasons = []

    calls = df[
        df["option_type"] == "CALL"
    ]

    puts = df[
        df["option_type"] == "PUT"
    ]

    if calls.empty or puts.empty:
        return (
            score,
            "NEUTRAL",
            reasons,
            {},
            {}
        )

    call_premium = calls["premium_flow"].sum()
    put_premium = puts["premium_flow"].sum()

    total_premium = (
        call_premium + put_premium
    )

    if total_premium > 0:

        ratio = (
            call_premium
            / total_premium
        )

        if ratio >= 0.60:
            score += 12
            reasons.append(
                "Call Premium 강세"
            )

        elif ratio >= 0.55:
            score += 7
            reasons.append(
                "Call Premium 우세"
            )

        elif ratio <= 0.40:
            score -= 12
            reasons.append(
                "Put Premium 강세"
            )

        elif ratio <= 0.45:
            score -= 7
            reasons.append(
                "Put Premium 우세"
            )

    call_volume = calls["volume"].sum()
    put_volume = puts["volume"].sum()

    total_volume = (
        call_volume + put_volume
    )

    if total_volume > 0:

        ratio = (
            call_volume
            / total_volume
        )

        if ratio >= 0.60:
            score += 10
            reasons.append(
                "Call 거래량 우세"
            )

        elif ratio <= 0.40:
            score -= 10
            reasons.append(
                "Put 거래량 우세"
            )

    call_oi = calls["openInterest"].sum()
    put_oi = puts["openInterest"].sum()

    total_oi = call_oi + put_oi

    if total_oi > 0:

        ratio = (
            call_oi / total_oi
        )

        if ratio >= 0.60:
            score += 8
            reasons.append(
                "Call OI 우세"
            )

        elif ratio <= 0.40:
            score -= 8
            reasons.append(
                "Put OI 우세"
            )

    dte_score, dte_reasons = (
        dte_quality(df)
    )

    score += dte_score
    reasons.extend(dte_reasons)

    long_calls = calls[
        calls["DTE"] >= 30
    ]

    if not long_calls.empty:
        score += 5
        reasons.append(
            "30D+ Call 구조 존재"
        )

    long_puts = puts[
        puts["DTE"] >= 30
    ]

    if not long_puts.empty:
        score -= 2
        reasons.append(
            "30D+ Put 구조 존재"
        )

    greeks = calculate_greeks(
        df,
        current_price
    )

    delta = greeks["Delta"]
    gex = greeks["GEX"]
    hiro = greeks["HIRO"]
    vanna = greeks["Vanna"]

    if abs(delta) > 5_000_000:

        if delta > 0:
            score += 12
            reasons.append(
                "Delta Exposure 강한 양수"
            )
        else:
            score -= 12
            reasons.append(
                "Delta Exposure 강한 음수"
            )

    elif delta > 0:
        score += 7
        reasons.append(
            "Delta Exposure 양수"
        )

    elif delta < 0:
        score -= 7
        reasons.append(
            "Delta Exposure 음수"
        )

    if gex > 0:
        score += 3
        reasons.append("GEX Positive")

    elif gex < 0:
        score -= 3
        reasons.append("GEX Negative")

    if hiro > 0:
        score += 5
        reasons.append(
            "HIRO Proxy Positive"
        )

    elif hiro < 0:
        score -= 5
        reasons.append(
            "HIRO Proxy Negative"
        )

    if vanna > 0:
        score += 3
        reasons.append(
            "Vanna Positive"
        )

    elif vanna < 0:
        score -= 3
        reasons.append(
            "Vanna Negative"
        )

    walls = calculate_walls(
        df,
        current_price
    )

    call_distance = walls["call_distance"]
    put_distance = walls["put_distance"]

    if call_distance is not None:

        if call_distance >= 10:
            score += 5
            reasons.append(
                f"Call Wall 여유 +{call_distance:.1f}%"
            )

        elif call_distance >= 5:
            score += 3
            reasons.append(
                f"Call Wall +{call_distance:.1f}%"
            )

        elif call_distance < 2:
            score -= 5
            reasons.append(
                "Call Wall 근접"
            )

    if put_distance is not None:

        if abs(put_distance) >= 8:
            score += 3
            reasons.append(
                f"Put Wall 지지 {put_distance:+.1f}%"
            )

        elif abs(put_distance) < 4:
            score -= 2
            reasons.append(
                "Put Wall 근접"
            )

    iv_score, iv_reason, median_iv = (
        calculate_iv_quality(df)
    )

    score += iv_score

    if iv_reason:
        reasons.append(iv_reason)

    score = max(
        0,
        min(
            100,
            score
        )
    )

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

    details = {
        "delta": delta,
        "gex": gex,
        "hiro": hiro,
        "vanna": vanna,
        "call_wall": walls["call_wall"],
        "put_wall": walls["put_wall"],
        "call_distance": call_distance,
        "put_distance": put_distance,
        "iv": median_iv
    }

    return (
        score,
        direction,
        reasons,
        details,
        greeks
    )


def fast_scan(ticker):

    try:

        price = get_current_price(ticker)

        if price is None:
            return None

        df = get_option_data(ticker)

        if df.empty:
            return None

        df = calculate_premium_flow(df)
        df = calculate_dte(
            df,
            MIN_DTE,
            MAX_DTE
        )

        if df.empty:
            return None

        flow = calculate_basic_flow(df)

        return {
            "ticker": ticker,
            "price": price,
            "df": df,
            **flow
        }

    except Exception:
        return None


def analyze_one(item):

    try:

        score, direction, reasons, details, _ = (
            calculate_final_score(
                item["df"],
                item["price"]
            )
        )

        return {
            "ticker": item["ticker"],
            "price": item["price"],
            "score": score,
            "direction": direction,
            "category": classify(score),
            "reasons": reasons,
            "_df": item["df"],
            **details
        }

    except Exception as e:

        print(
            f"Analyze error {item['ticker']}: {e}"
        )

        return None


def send_telegram(message):

    if not TELEGRAM_BOT_TOKEN:
        print(
            "Telegram token not configured."
        )
        return

    if not TELEGRAM_CHAT_ID:
        print(
            "Telegram chat ID not configured."
        )
        return

    url = TELEGRAM_API_URL.format(
        TELEGRAM_BOT_TOKEN
    )

    chunks = []
    current = ""

    for line in message.split("\n"):

        if (
            len(current)
            + len(line)
            + 1
            > TELEGRAM_MAX_LENGTH
        ):

            if current:
                chunks.append(current)
                current = ""

            while len(line) > TELEGRAM_MAX_LENGTH:

                chunks.append(
                    line[:TELEGRAM_MAX_LENGTH]
                )

                line = line[
                    TELEGRAM_MAX_LENGTH:
                ]

        if current:
            current += "\n" + line
        else:
            current = line

    if current:
        chunks.append(current)

    for chunk in chunks:

        payload = urllib.parse.urlencode(
            {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
                "disable_web_page_preview": "true"
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=payload,
            method="POST"
        )

        with urllib.request.urlopen(
            request,
            timeout=30
        ) as response:

            result = (
                response
                .read()
                .decode("utf-8")
            )

        if '"ok":true' not in result.replace(
            " ",
            ""
        ):

            raise RuntimeError(
                f"Telegram API error: {result}"
            )

    print(
        "Telegram 전송 완료"
    )


def build_message(final_results):

    lines = []

    lines.append(
        "🔥 OPTION FLOW SCANNER V2"
    )

    lines.append(
        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    lines.append("")

    entry = [
        x for x in final_results
        if x["score"] >= ENTRY_SCORE
    ][:TOP_ENTRY]

    lines.append(
        "🟢 오늘 진입 후보 TOP 5"
    )

    lines.append("")

    if entry:

        for i, r in enumerate(entry, 1):

            lines.append(
                f"{i}. {r['ticker']} | "
                f"{r['score']:.1f} | "
                f"{r['direction']}"
            )

            if r["reasons"]:

                lines.append(
                    "   "
                    + " | ".join(
                        r["reasons"][:8]
                    )
                )

    else:

        lines.append(
            "오늘 진입 후보 없음"
        )

    lines.append("")
    lines.append(
        "🟡 WATCH"
    )

    watch = [
        x for x in final_results
        if WATCH_SCORE
        <= x["score"]
        < ENTRY_SCORE
    ]

    if watch:

        for r in watch:

            lines.append(
                f"{r['ticker']} | "
                f"{r['score']:.1f} | "
                f"{r['direction']}"
            )

    else:

        lines.append(
            "감시 종목 없음"
        )

    lines.append("")
    lines.append(
        "📊 전체 순위"
    )

    for i, r in enumerate(
        final_results,
        1
    ):

        lines.append(
            f"{i}. {r['ticker']:<6} "
            f"{r['score']:>5.1f} | "
            f"{r['direction']}"
        )

    lines.append("")
    lines.append(
        "🎯 TOP 5 WALL / IV"
    )

    for r in final_results[:5]:

        lines.append(
            f"{r['ticker']} "
            f"${r['price']:.2f}"
        )

        if r["call_wall"] is not None:

            distance = r[
                "call_distance"
            ]

            lines.append(
                f"  Call Wall "
                f"${r['call_wall']:.2f} "
                f"({distance:+.1f}%)"
            )

        if r["put_wall"] is not None:

            distance = r[
                "put_distance"
            ]

            lines.append(
                f"  Put Wall "
                f"${r['put_wall']:.2f} "
                f"({distance:+.1f}%)"
            )

        if r["iv"] is not None:

            lines.append(
                f"  IV {r['iv']:.1f}%"
            )

    lines.append("")
    lines.append(
        "⚠️ Greeks는 yfinance 옵션 데이터 기반 Proxy 계산입니다."
    )

    return "\n".join(lines)


def save_csv(final_results):

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    output = os.path.join(
        RESULT_DIR,
        f"OPTION_FINAL_RANKING_V2_{today}.csv"
    )

    rows = []

    for r in final_results:

        rows.append(
            {
                "ticker": r["ticker"],
                "price": r["price"],
                "score": r["score"],
                "direction": r["direction"],
                "category": r["category"],
                "reasons": " | ".join(
                    r["reasons"]
                ),
                "delta": r["delta"],
                "gex": r["gex"],
                "hiro": r["hiro"],
                "vanna": r["vanna"],
                "call_wall": r["call_wall"],
                "put_wall": r["put_wall"],
                "call_distance": r[
                    "call_distance"
                ],
                "put_distance": r[
                    "put_distance"
                ],
                "iv": r["iv"]
            }
        )

    pd.DataFrame(rows).to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )

    return output


def main():

    print("=" * 70)
    print("🔥 OPTION FLOW SCANNER V2")
    print("=" * 70)

    universe = load_latest_universe()

    if universe.empty:

        print(
            "Universe가 없습니다."
        )

        raise SystemExit(1)

    tickers = (
        universe["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
        .drop_duplicates()
        .head(MAX_UNIVERSE)
        .tolist()
    )

    print(
        f"Universe: {len(tickers)}"
    )

    print(
        f"Workers: {MAX_WORKERS}"
    )

    fast_results = []

    completed = 0

    start = time.time()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                fast_scan,
                ticker
            ): ticker
            for ticker in tickers
        }

        for future in as_completed(
            futures
        ):

            ticker = futures[future]

            completed += 1

            try:

                result = future.result()

                if result:

                    fast_results.append(
                        result
                    )

                    print(
                        f"[{completed}/{len(tickers)}] "
                        f"{ticker} "
                        f"Flow={result['flow_score']:.1f}"
                    )

                else:

                    print(
                        f"[{completed}/{len(tickers)}] "
                        f"{ticker} SKIP"
                    )

            except Exception as e:

                print(
                    f"[{completed}/{len(tickers)}] "
                    f"{ticker} ERROR {e}"
                )

    elapsed = time.time() - start

    print("")
    print(
        f"1차 스캔 완료: {elapsed:.1f}s"
    )

    fast_results.sort(
        key=lambda x: x["flow_score"],
        reverse=True
    )

    preselected = fast_results[
        :FLOW_PRESELECT
    ]

    print("")
    print(
        f"1차 TOP {len(preselected)}"
    )

    final_results = []

    for item in preselected:

        print(
            f"FINAL ANALYSIS: "
            f"{item['ticker']}"
        )

        result = analyze_one(item)

        if result:
            final_results.append(result)

    if not final_results:

        print(
            "최종 분석 결과 없음"
        )

        raise SystemExit(1)

    final_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    for i, r in enumerate(
        final_results,
        1
    ):

        print(
            f"{i}. "
            f"{r['ticker']} "
            f"{r['score']:.1f} "
            f"{r['direction']}"
        )

    message = build_message(
        final_results
    )

    print("")
    print("=" * 70)
    print(message)
    print("=" * 70)

    csv_file = save_csv(
        final_results
    )

    print(
        f"CSV: {csv_file}"
    )

    send_telegram(
        message
    )

    print("")
    print(
        "🔥 OPTION FLOW SCANNER V2 완료"
    )


if __name__ == "__main__":
    main()

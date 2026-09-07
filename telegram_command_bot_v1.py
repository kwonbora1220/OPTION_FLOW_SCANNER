import os
import requests
import pandas as pd


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


# ============================================================
# RESULT FILE
# ============================================================

def find_result_file():

    target = "OPTION_FINAL_RANKING.csv"

    for root, dirs, files in os.walk("."):

        if target in files:

            return os.path.join(
                root,
                target
            )

    return None


# ============================================================
# TELEGRAM
# ============================================================

def send_message(text):

    if not BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN이 없습니다."
        )

    if not CHAT_ID:

        raise RuntimeError(
            "TELEGRAM_CHAT_ID가 없습니다."
        )

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "📱 TELEGRAM V1 RESULT SENDER"
    )
    print("=" * 70)

    if not BOT_TOKEN:

        print(
            "❌ TELEGRAM_BOT_TOKEN 없음"
        )

        raise SystemExit(1)

    if not CHAT_ID:

        print(
            "❌ TELEGRAM_CHAT_ID 없음"
        )

        raise SystemExit(1)

    print(
        "✅ Telegram 환경변수 확인"
    )

    result_file = find_result_file()

    if not result_file:

        print(
            "❌ OPTION_FINAL_RANKING.csv 없음"
        )

        raise SystemExit(1)

    print(
        f"✅ 결과 파일: {result_file}"
    )

    df = pd.read_csv(
        result_file
    )

    df = df.sort_values(
        "score",
        ascending=False
    )

    message = (
        "🔥 OPTION FLOW SCANNER V1\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📊 전체 종목 점수\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    for i, (_, row) in enumerate(
        df.head(20).iterrows(),
        1
    ):

        ticker = str(
            row.get(
                "ticker",
                "UNKNOWN"
            )
        )

        try:

            score = float(
                row.get(
                    "score",
                    0
                )
            )

        except Exception:

            score = 0

        if score >= 70:

            status = "🟢 오늘 진입 후보"

        elif score >= 40:

            status = "🟡 관망"

        else:

            status = "🔴 회피"

        message += (
            f"{i}. {ticker:<6} "
            f"{score:.1f}점 "
            f"{status}\n"
        )

    send_message(
        message
    )

    print(
        "✅ Telegram 전송 완료"
    )


if __name__ == "__main__":

    main()

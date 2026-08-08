import os
import time
import urllib.parse
import urllib.request
import json
import subprocess
import sys


TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    ""
)

CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID",
    ""
)


def api(method, data=None):

    url = (
        f"https://api.telegram.org/"
        f"bot{TOKEN}/{method}"
    )

    if data is None:
        data = {}

    encoded = urllib.parse.urlencode(
        data
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=encoded,
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:

        return json.loads(
            response.read().decode(
                "utf-8"
            )
        )


def send_message(
    chat_id,
    text
):

    api(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text
        }
    )


def main():

    if not TOKEN:

        print(
            "TELEGRAM_BOT_TOKEN 없음"
        )

        return

    print(
        "Telegram Command Bot V2 시작"
    )

    offset = None

    while True:

        try:

            data = {
                "timeout": 30
            }

            if offset is not None:
                data["offset"] = offset

            result = api(
                "getUpdates",
                data
            )

            updates = result.get(
                "result",
                []
            )

            for update in updates:

                offset = (
                    update["update_id"]
                    + 1
                )

                message = update.get(
                    "message",
                    {}
                )

                chat = message.get(
                    "chat",
                    {}
                )

                chat_id = chat.get(
                    "id"
                )

                text = message.get(
                    "text",
                    ""
                ).strip()

                if not chat_id or not text:
                    continue

                if (
                    CHAT_ID
                    and str(chat_id)
                    != str(CHAT_ID)
                ):
                    continue

                if text == "/start":

                    send_message(
                        chat_id,
                        "🔥 OPTION FLOW SCANNER V2\n\n"
                        "사용법:\n"
                        "/scan AAPL\n"
                        "/scan NVDA\n"
                        "/scan SOXL"
                    )

                    continue

                if text.startswith(
                    "/scan "
                ):

                    ticker = (
                        text.split(
                            " ",
                            1
                        )[1]
                        .upper()
                        .strip()
                    )

                    send_message(
                        chat_id,
                        f"🔎 {ticker} 분석 시작"
                    )

                    script = os.path.join(
                        os.path.dirname(
                            os.path.abspath(__file__)
                        ),
                        "option_search.py"
                    )

                    try:

                        process = subprocess.run(
                            [
                                sys.executable,
                                script,
                                ticker
                            ],
                            capture_output=True,
                            text=True,
                            timeout=120
                        )

                        output = (
                            process.stdout
                            or process.stderr
                        )

                        if len(output) > 3500:
                            output = output[
                                -3500:
                            ]

                        send_message(
                            chat_id,
                            output
                        )

                    except Exception as e:

                        send_message(
                            chat_id,
                            f"❌ 오류: {e}"
                        )

                else:

                    send_message(
                        chat_id,
                        "사용법: /scan AAPL"
                    )

        except Exception as e:

            print(
                f"Bot error: {e}"
            )

            time.sleep(5)


if __name__ == "__main__":
    main()

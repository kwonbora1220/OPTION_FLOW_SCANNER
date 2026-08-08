
import os
import subprocess
import threading
import time
import requests

# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = ""
CHAT_ID = ""

BASE_DIR = r"C:\Users\bora\Desktop\2 OPTION FLOW SCANNER"
PROGRAM_DIR = os.path.join(
    BASE_DIR,
    "02_PROGRAM"
)

V1_FILE = os.path.join(
    PROGRAM_DIR,
    "batch_option_search.py"
)

V2_FILE = os.path.join(
    PROGRAM_DIR,
    "option_flow_scanner_v2.py"
)

TELEGRAM_URL = (
    f"https://api.telegram.org/bot{BOT_TOKEN}"
)

# ============================================================
# STATE
# ============================================================

running = False
running_version = None
process = None


# ============================================================
# TELEGRAM SEND
# ============================================================

def send_message(text):

    try:

        requests.post(
            f"{TELEGRAM_URL}/sendMessage",
            data={
                "chat_id": CHAT_ID,
                "text": text
            },
            timeout=20
        )

    except Exception as e:

        print(
            f"Telegram 전송 오류: {e}"
        )


# ============================================================
# RUN SCANNER
# ============================================================

def run_scanner(version, scanner_file):

    global running
    global running_version
    global process

    if running:

        send_message(
            "⚠️ 이미 스캐너가 실행 중입니다.\n\n"
            f"현재 실행 중: V{running_version}"
        )

        return

    if not os.path.exists(scanner_file):

        send_message(
            "❌ 스캐너 파일을 찾을 수 없습니다.\n\n"
            f"{scanner_file}"
        )

        return

    running = True
    running_version = version

    try:

        send_message(
            f"🔥 OPTION FLOW SCANNER V{version} 시작\n\n"
            f"💻 PC에서 V{version} 분석을 시작합니다.\n"
            f"📂 {os.path.basename(scanner_file)}\n\n"
            "완료되면 Telegram으로 알려드립니다."
        )

        print("")
        print("=" * 70)
        print(
            f"🔥 OPTION FLOW SCANNER V{version}"
        )
        print("=" * 70)
        print(
            f"FILE : {scanner_file}"
        )
        print("")

        process = subprocess.Popen(
            [
                "python",
                scanner_file
            ],
            cwd=PROGRAM_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        for line in process.stdout:

            line = line.rstrip()

            if line:

                print(line)

        process.wait()

        return_code = process.returncode

        if return_code == 0:

            send_message(
                f"✅ OPTION FLOW SCANNER V{version} 완료\n\n"
                "PC에서 분석이 정상적으로 종료되었습니다.\n"
                "Telegram 최종 결과와 CSV를 확인하세요."
            )

            print("")
            print(
                f"✅ V{version} 완료"
            )

        else:

            send_message(
                f"❌ OPTION FLOW SCANNER V{version} 오류\n\n"
                f"Return Code: {return_code}"
            )

            print("")
            print(
                f"❌ V{version} 종료 코드: {return_code}"
            )

    except Exception as e:

        print("")
        print(
            f"❌ V{version} 실행 오류"
        )

        print(e)

        send_message(
            f"❌ OPTION FLOW SCANNER V{version} 실행 오류\n\n"
            f"{e}"
        )

    finally:

        process = None
        running = False
        running_version = None


# ============================================================
# START SCANNER
# ============================================================

def start_scanner(version):

    if version == 1:

        scanner_file = V1_FILE

    elif version == 2:

        scanner_file = V2_FILE

    else:

        return

    thread = threading.Thread(
        target=run_scanner,
        args=(
            version,
            scanner_file
        ),
        daemon=True
    )

    thread.start()


# ============================================================
# TELEGRAM GET UPDATES
# ============================================================

def get_updates(offset=None):

    params = {
        "timeout": 30
    }

    if offset is not None:

        params["offset"] = offset

    response = requests.get(
        f"{TELEGRAM_URL}/getUpdates",
        params=params,
        timeout=40
    )

    return response.json()


# ============================================================
# TELEGRAM BOT LOOP
# ============================================================

def bot_loop():

    print("=" * 70)
    print("🤖 TELEGRAM OPTION FLOW COMMAND BOT")
    print("=" * 70)

    print("")
    print(
        f"CHAT ID : {CHAT_ID}"
    )

    print("")
    print(
        "사용 가능한 명령어:"
    )

    print(
        "/scan1"
    )

    print(
        "/scan2"
    )

    print(
        "/status"
    )

    print(
        "/help"
    )

    print("")
    print(
        "📱 Telegram 명령 대기 중..."
    )

    print("")

    offset = None

    while True:

        try:

            data = get_updates(
                offset
            )

            if not data.get("ok"):

                print(
                    "Telegram API 오류:"
                )

                print(data)

                time.sleep(5)

                continue

            updates = data.get(
                "result",
                []
            )

            for update in updates:

                offset = (
                    update["update_id"] + 1
                )

                message = update.get(
                    "message"
                )

                if not message:

                    continue

                chat = message.get(
                    "chat",
                    {}
                )

                chat_id = str(
                    chat.get(
                        "id",
                        ""
                    )
                )

                # ------------------------------------------------
                # 지정된 Chat ID만 허용
                # ------------------------------------------------

                if chat_id != CHAT_ID:

                    continue

                text = message.get(
                    "text",
                    ""
                ).strip().lower()

                print(
                    f"📱 Telegram 명령: {text}"
                )

                # =================================================
                # /start
                # =================================================

                if text == "/start":

                    send_message(
                        "🤖 OPTION FLOW SCANNER BOT\n\n"
                        "사용 가능한 명령어\n\n"
                        "/scan1\n"
                        "→ OPTION FLOW SCANNER V1 실행\n\n"
                        "/scan2\n"
                        "→ OPTION FLOW SCANNER V2 실행\n\n"
                        "/status\n"
                        "→ 현재 실행 상태\n\n"
                        "/help\n"
                        "→ 명령어 안내"
                    )

                # =================================================
                # /help
                # =================================================

                elif text == "/help":

                    send_message(
                        "📋 OPTION FLOW SCANNER 명령어\n\n"
                        "🟢 /scan1\n"
                        "V1 실행\n\n"
                        "🔵 /scan2\n"
                        "V2 실행\n\n"
                        "📊 /status\n"
                        "현재 실행 상태 확인\n\n"
                        "ℹ️ /help\n"
                        "도움말"
                    )

                # =================================================
                # /status
                # =================================================

                elif text == "/status":

                    if running:

                        send_message(
                            f"🟢 현재 실행 중\n\n"
                            f"OPTION FLOW SCANNER V{running_version}\n\n"
                            "분석이 끝날 때까지 "
                            "다른 스캐너는 실행할 수 없습니다."
                        )

                    else:

                        send_message(
                            "⚪ 현재 대기 상태입니다.\n\n"
                            "V1 / V2 모두 실행 가능합니다."
                        )

                # =================================================
                # /scan1
                # =================================================

                elif text == "/scan1":

                    if running:

                        send_message(
                            f"⚠️ 실행할 수 없습니다.\n\n"
                            f"현재 V{running_version}이 "
                            "실행 중입니다."
                        )

                    else:

                        send_message(
                            "🚀 V1 실행 명령을 받았습니다."
                        )

                        start_scanner(1)

                # =================================================
                # /scan2
                # =================================================

                elif text == "/scan2":

                    if running:

                        send_message(
                            f"⚠️ 실행할 수 없습니다.\n\n"
                            f"현재 V{running_version}이 "
                            "실행 중입니다."
                        )

                    else:

                        send_message(
                            "🚀 V2 실행 명령을 받았습니다."
                        )

                        start_scanner(2)

                # =================================================
                # UNKNOWN
                # =================================================

                else:

                    send_message(
                        "❓ 알 수 없는 명령입니다.\n\n"
                        "/help 를 입력하세요."
                    )

        except Exception as e:

            print("")
            print(
                "❌ BOT LOOP 오류"
            )

            print(e)

            time.sleep(5)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if BOT_TOKEN.startswith(
        "여기에_"
    ):

        print("")
        print(
            "❌ Telegram Bot Token이 없습니다."
        )

        print("")
        print(
            'BOT_TOKEN = "여기에_새봇_TOKEN_입력"'
        )

        print(
            "부분을 실제 Bot Token으로 바꿔주세요."
        )

        input(
            "\nEnter를 누르면 종료합니다..."
        )

        raise SystemExit

    print("")
    print(
        "🔥 Telegram Option Flow Command Bot 시작"
    )

    print("")
    print(
        f"V1 : {V1_FILE}"
    )

    print(
        f"V2 : {V2_FILE}"
    )

    print("")

    bot_loop()

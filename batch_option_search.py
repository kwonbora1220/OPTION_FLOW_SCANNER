import os
import sys
import subprocess


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


def main():

    if len(sys.argv) < 2:

        print(
            "사용법:"
        )

        print(
            "python batch_option_search.py AAPL MSFT NVDA"
        )

        return

    tickers = [
        x.upper().strip()
        for x in sys.argv[1:]
        if x.strip()
    ]

    script = os.path.join(
        BASE_DIR,
        "option_search.py"
    )

    for ticker in tickers:

        print("")
        print("=" * 70)
        print(
            f"OPTION SEARCH: {ticker}"
        )
        print("=" * 70)

        result = subprocess.run(
            [
                sys.executable,
                script,
                ticker
            ]
        )

        if result.returncode != 0:

            print(
                f"{ticker}: ERROR"
            )


if __name__ == "__main__":
    main()

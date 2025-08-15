from __future__ import annotations

import subprocess


def main() -> None:
    subprocess.run(
        [
            "streamlit",
            "run",
            "app/streamlit_app.py",
        ]
    )


if __name__ == "__main__":
    main()

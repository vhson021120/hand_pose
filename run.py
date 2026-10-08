import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    launcher = ROOT / 'run_app' / 'main.py'
    return subprocess.call([sys.executable, str(launcher), *sys.argv[1:]], cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())

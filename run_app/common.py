import subprocess
import sys
import ctypes
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def choose_file(title: str, filetypes):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    finally:
        root.destroy()
    return file_path


def run_python_script(script_name: str, args, cwd=None):
    target = PROJECT_ROOT / script_name
    cmd = [sys.executable, str(target), *args]
    workdir = str(cwd) if cwd is not None else str(PROJECT_ROOT)
    if sys.platform == "win32" and Path(workdir).drive:
        short_path = ctypes.create_unicode_buffer(260)
        if ctypes.windll.kernel32.GetShortPathNameW(workdir, short_path, len(short_path)):
            workdir = short_path.value
    command_line = f"\n[RUN] {' '.join(cmd)}"
    try:
        print(command_line)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe_command_line = command_line.encode(encoding, errors="backslashreplace").decode(encoding)
        print(safe_command_line)
    print(f"[CWD] {workdir}")
    return subprocess.run(cmd, cwd=workdir)

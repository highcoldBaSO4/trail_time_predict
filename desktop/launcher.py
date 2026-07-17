from __future__ import annotations

import argparse
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from desktop.runtime_paths import resource_root, user_data_directory


APP_TITLE = "越野跑比赛时间预测"
HEALTH_PATH = "/_stcore/health"


def _available_port(preferred: int = 8501) -> int:
    for port in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", port))
                return int(probe.getsockname()[1])
        except OSError:
            continue
    raise RuntimeError("无法获取本地服务端口")


def _server_command(port: int) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--serve", str(port)]
    return [sys.executable, str(Path(__file__).resolve()), "--serve", str(port)]


def _run_server(port: int) -> None:
    from streamlit.web import bootstrap

    root = resource_root()
    app_path = root / "app.py"
    if not app_path.is_file():
        raise FileNotFoundError(f"找不到网页入口：{app_path}")
    os.chdir(root)
    bootstrap.run(
        str(app_path),
        False,
        [],
        {
            "global_developmentMode": False,
            "server_address": "127.0.0.1",
            "server_port": port,
            "server_headless": True,
            "browser_gatherUsageStats": False,
        },
    )


class DesktopController:
    def __init__(self) -> None:
        self.port = _available_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle = None
        self.ready = False
        self.events: queue.Queue[tuple[str, str | None]] = queue.Queue()

        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("460x230")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        tk.Label(self.root, text=APP_TITLE, font=("Microsoft YaHei UI", 16, "bold")).pack(pady=(24, 10))
        self.status = tk.StringVar(value="正在启动本地服务……")
        tk.Label(self.root, textvariable=self.status, font=("Microsoft YaHei UI", 10)).pack(pady=4)
        tk.Label(self.root, text=self.url, fg="#0f6a31", font=("Segoe UI", 10)).pack(pady=5)

        buttons = tk.Frame(self.root)
        buttons.pack(pady=18)
        self.open_button = tk.Button(buttons, text="重新打开页面", width=15, state="disabled", command=self.open_browser)
        self.open_button.pack(side="left", padx=8)
        tk.Button(buttons, text="退出应用", width=12, command=self.close).pack(side="left", padx=8)
        tk.Label(
            self.root,
            text="关闭此窗口会同时停止后台服务",
            fg="#667085",
            font=("Microsoft YaHei UI", 9),
        ).pack()
        self.root.after(100, self._poll_events)

    def start(self) -> None:
        log_dir = user_data_directory() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = (log_dir / "launcher.log").open("ab")
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            _server_command(self.port),
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
        threading.Thread(target=self._wait_until_ready, daemon=True).start()
        self.root.mainloop()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                self.events.put(("failed", "本地服务启动失败，请查看运行日志。"))
                return
            try:
                with urllib.request.urlopen(self.url + HEALTH_PATH, timeout=1.0) as response:
                    if response.status == 200:
                        self.events.put(("ready", None))
                        return
            except Exception:
                time.sleep(0.25)
        self.events.put(("failed", "启动超时，请退出后重试。"))

    def _poll_events(self) -> None:
        try:
            while True:
                event, message = self.events.get_nowait()
                if event == "ready":
                    self._startup_ready()
                elif event == "failed":
                    self._startup_failed(message or "启动失败")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _startup_ready(self) -> None:
        self.ready = True
        self.status.set("应用正在运行")
        self.open_button.configure(state="normal")
        self.open_browser()

    def _startup_failed(self, message: str) -> None:
        self.status.set("启动失败")
        messagebox.showerror(APP_TITLE, message)

    def open_browser(self) -> None:
        if self.ready:
            webbrowser.open(self.url, new=2)

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.log_handle is not None:
            self.log_handle.close()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--serve", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.serve is not None:
        _run_server(args.serve)
        return
    DesktopController().start()


if __name__ == "__main__":
    main()

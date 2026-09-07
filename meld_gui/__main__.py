"""Command line entry point: ``python -m meld_gui``."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import webbrowser

from .config import settings


def _open_browser(url: str, delay: float = 1.5) -> None:
    threading.Timer(delay, lambda: webbrowser.open(url)).start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meld-gui",
        description="Serve the MELD AI-text detector web interface.",
    )
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument(
        "--device",
        default=settings.device,
        help="Torch device: auto, cpu or cuda.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Download and load the checkpoint at startup instead of on demand.",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser window."
    )
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    parser.add_argument(
        "--model-dir", default=settings.model_dir,
        help="Use a local checkpoint folder instead of the HuggingFace cache.",
    )
    args = parser.parse_args(argv)

    settings.host = args.host
    settings.port = args.port
    settings.device = args.device
    settings.model_dir = args.model_dir
    settings.eager_load = args.preload

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    url = f"http://{args.host}:{args.port}"
    print(f"\n  MELD Detector  ->  {url}\n  API docs       ->  {url}/docs\n")
    if not args.no_browser:
        _open_browser(url)

    import uvicorn

    uvicorn.run(
        "meld_gui.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

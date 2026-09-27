"""Run the web app with `python -m app`."""

import os
import socket

import uvicorn

from . import APP_NAME


def _lan_addresses() -> list[str]:
    """Best-effort list of this machine's local network addresses."""
    found = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; this picks the outgoing interface
            found.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    return sorted(a for a in found if not a.startswith("127."))


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    if host in ("0.0.0.0", ""):
        print(f"{APP_NAME} is running. Open one of these addresses:")
        print(f"  On this computer:      http://127.0.0.1:{port}")
        for address in _lan_addresses():
            print(f"  From another device:   http://{address}:{port}")
        print("  Warning: anyone on this network can use the app while it runs (there is no login).")
    else:
        print(f"{APP_NAME} running at http://{host}:{port}")
    print("  Press Ctrl+C to stop.")
    uvicorn.run("app.main:app", host=host, port=port)


if __name__ == "__main__":
    main()

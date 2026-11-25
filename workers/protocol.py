"""
The tiny message loop shared by all four tool workers.

A worker is a long-lived Python process started by the agent inside the tool's
OWN conda env. It loads its model once, then answers one JSON request per line:

    agent  -> worker   {"cmd": "init", "settings": {...}}      (once)
    worker -> agent    {"ok": true, "result": {...}}
    agent  -> worker   {"cmd": "run", "args": {...}}           (per question)
    worker -> agent    {"ok": true, "result": {...}}  or  {"ok": false, "error": "..."}

stdout is reserved for these replies. Model libraries love to print, so the real
stdout file descriptor is duplicated for the protocol and fd 1 is pointed at
stderr -- any print (Python or C level) lands in the worker's log file instead of
corrupting a reply.

Standard library only, Python 3.8 compatible (the LAE env is 3.8).
"""

import json
import os
import sys
import traceback


def serve(init_fn, run_fn):
    """init_fn(settings) -> dict ; run_fn(args) -> dict. Blocks until stdin closes."""
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    proto = os.fdopen(proto_fd, "w", encoding="utf-8")

    def reply(message):
        proto.write(json.dumps(message) + "\n")
        proto.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            cmd = request.get("cmd")
            if cmd == "init":
                reply({"ok": True, "result": init_fn(request.get("settings", {}))})
            elif cmd == "run":
                reply({"ok": True, "result": run_fn(request.get("args", {}))})
            elif cmd == "ping":
                reply({"ok": True, "result": "pong"})
            elif cmd == "exit":
                reply({"ok": True, "result": "bye"})
                break
            else:
                reply({"ok": False, "error": "unknown cmd %r" % cmd})
        except Exception:  # noqa: BLE001 -- the traceback goes back to the agent
            traceback.print_exc()
            reply({"ok": False, "error": traceback.format_exc()})

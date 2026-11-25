"""
Agent-side handle on one tool worker (see workers/protocol.py).

    worker = ToolWorker("LAE-DINO", python_exe, "workers/lae_worker.py", settings)
    result = worker.call({...})      # starts + loads the model on first use

The process stays alive between questions, so each model is loaded once per
Streamlit session. The worker's stderr (model-loading chatter, tracebacks) goes
to logs/<name>.log; the tail of that log is attached to any error.
"""

import atexit
import json
import os
import subprocess
import threading
import queue


class ToolWorkerError(RuntimeError):
    pass


class ToolWorker:
    def __init__(self, name, python_exe, script, settings, log_dir, timeout=None):
        self.name = name
        self.python_exe = python_exe
        self.script = script
        self.settings = settings
        self.timeout = timeout
        self.log_path = os.path.join(log_dir, "%s.log" % name.lower().replace(" ", "_"))
        self.load_info = None
        self._proc = None
        self._log = None
        self._lock = threading.Lock()
        os.makedirs(log_dir, exist_ok=True)
        atexit.register(self.close)

    # -- lifecycle ------------------------------------------------------------
    @property
    def alive(self):
        return self._proc is not None and self._proc.poll() is None

    def _start(self):
        if not self.python_exe or not os.path.isfile(self.python_exe):
            raise ToolWorkerError(
                "%s: python interpreter not found (%r). Set the env name or the "
                "full interpreter path in .env." % (self.name, self.python_exe)
            )
        self._log = open(self.log_path, "a", encoding="utf-8", errors="replace")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   PYTHONDONTWRITEBYTECODE="1")
        self._proc = subprocess.Popen(
            [self.python_exe, "-u", self.script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            cwd=os.path.dirname(self.script),
            env=env,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        try:
            self.load_info = self._send({"cmd": "init", "settings": self.settings})
        except Exception:
            self.close()  # never leave a half-initialised worker running
            raise

    def close(self):
        """Polite shutdown: ask the worker to exit, then clean up."""
        if self.alive:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 -- pipe already broken: _discard kills it
                pass
        self._discard()

    def _discard(self):
        """Forget the current process, whatever state it is in.

        On Windows a dead worker is not always reported by poll() the instant its
        stdout closes, and writing to its stdin then fails with
        OSError [Errno 22]. So a worker is marked dead the moment a pipe fails --
        not when poll() eventually notices -- and the next call starts a new one.
        """
        proc, self._proc = self._proc, None
        if proc is not None:
            if proc.poll() is None:
                if self.timeout is not None:
                    # Native Delineate creates subprocesses. A hung run must not
                    # strand them after its parent worker is stopped.
                    try:
                        import psutil
                        for child in psutil.Process(proc.pid).children(recursive=True):
                            child.kill()
                    except Exception:  # best-effort descendant cleanup; still kill the worker
                        pass
                try:
                    proc.kill()
                except OSError:
                    pass
            for pipe in (proc.stdin, proc.stdout):
                try:
                    pipe.close()  # may try to flush into a dead pipe -- ignore
                except (OSError, ValueError):
                    pass
            try:
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        if self._log:
            self._log.close()
            self._log = None

    # -- calls ----------------------------------------------------------------
    def _send(self, message):
        try:
            self._proc.stdin.write(json.dumps(message) + "\n")
            self._proc.stdin.flush()
            if self.timeout is None:
                line = self._proc.stdout.readline()
            else:
                replies = queue.Queue(maxsize=1)
                pipe = self._proc.stdout
                def read_reply():
                    try:
                        replies.put(pipe.readline())
                    except (OSError, ValueError):
                        replies.put("")
                threading.Thread(target=read_reply, daemon=True).start()
                try:
                    line = replies.get(timeout=self.timeout)
                except queue.Empty:
                    self._discard()
                    raise ToolWorkerError("%s exceeded %.0f s.\n%s" % (
                        self.name, self.timeout, self.log_tail()))
        except (OSError, ValueError):  # broken pipe (Windows: Errno 22)
            line = ""
        if not line:  # process died (import error, out of memory, crash, ...)
            self._discard()
            raise ToolWorkerError("%s worker exited.\n%s" % (self.name, self.log_tail()))
        reply = json.loads(line)
        if not reply.get("ok"):
            raise ToolWorkerError("%s failed:\n%s" % (self.name, reply.get("error")))
        return reply["result"]

    def call(self, args):
        with self._lock:
            if not self.alive:
                self._start()
            return self._send({"cmd": "run", "args": args})

    def log_tail(self, lines=25):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-lines:])
        except OSError:
            return "(no log)"

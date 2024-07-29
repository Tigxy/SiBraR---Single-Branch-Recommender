from datetime import datetime
import traceback
import sys
from queue import Queue
import psutil
import os


# Context manager that copies stdout and any exceptions to a log file
class Tee:
    def __init__(self, filename=None, with_timestamps=True):
        self.stdout = sys.stdout
        self.with_timestamps = with_timestamps
        self.timestamp_in_current_line = False

        # in case file is not set, we buffer messages until it is
        self.queue = Queue()

        self.file = None
        if filename is not None:
            self.set_file(filename)

    def set_file(self, filename):
        self.file = open(filename, 'w')
        while not self.queue.empty():
            self.file.write(self.queue.get())

    def __getattr__(self, key):
        return getattr(self.stdout, key)

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self._write(traceback.format_exc())
        if self.file is not None:
            self.file.close()

    def _write(self, s):
        if self.with_timestamps and not self.timestamp_in_current_line:
            s = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + " | " + s
            self.timestamp_in_current_line = True

        if any(s.endswith(c) for c in ("\n", "\r")):
            self.timestamp_in_current_line = False

        if self.file is not None:
            self.file.write(s)
        else:
            self.queue.put(s)

    def write(self, data):
        self._write(data)
        self.stdout.write(data)

    def write_cmd_line_call(self):
        proc = psutil.Process(os.getpid())
        self._write(f"Process called via '{' '.join(proc.cmdline())}'\n")

    def flush(self):
        if self.file is not None:
            self.file.flush()
        self.stdout.flush()

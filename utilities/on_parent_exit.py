"""
####################################################################################
from https://gist.github.com/evansd/2346614
####################################################################################

Utility (Linux only) to ensure subprocesses exit when their parents do by sending
a specified signal when the parent dies.
Usage:
    subprocess.Popen(['some-executable'], preexec_fn=on_parent_exit('SIGHUP'))
"""

import signal
from ctypes import cdll
from sys import platform

# Constant taken from http://linux.die.net/include/linux/prctl.h
PR_SET_PDEATHSIG = 1


class PrCtlError(Exception):
    pass


def is_run_on_linux():
    return platform == "linux" or platform == "linux2"


def on_parent_exit(signame):
    """
    Return a function to be run in a child process which will trigger SIGNAME
    to be sent when the parent process dies
    """
    signum = getattr(signal, signame)

    def set_parent_exit_signal():
        # http://linux.die.net/man/2/prctl
        result = cdll['libc.so.6'].prctl(PR_SET_PDEATHSIG, signum)
        if result != 0:
            raise PrCtlError('prctl failed with error code %s' % result)

    if is_run_on_linux():
        return set_parent_exit_signal
    else:
        def dummy():
            pass

        return dummy()

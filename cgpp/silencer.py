"""http://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
https://stackoverflow.com/questions/36956083/how-can-the-terminal-output-of-executables-run-by-python-functions-be-silenced-i
"""
import contextlib
import os
import sys


@contextlib.contextmanager
def suppress_stdout():
    """
    with suppress_stdout():
        lines_you_want_to_suppress_python_stdouts()
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@contextlib.contextmanager
def suppress_cffi_out():
    """
    with suppress_cffi_out():
        lines_you_want_to_suppress_FFI_stdouts()
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    sys.stdout.flush()
    os.dup2(devnull, 1)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.close(old_stdout)

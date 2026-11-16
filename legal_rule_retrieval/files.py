"""Private text output for benchmark traces, which can contain case facts."""

import os


def private_text_output(path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        if os.name == 'posix':
            os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, 'w', encoding='utf-8')
    except BaseException:
        os.close(descriptor)
        raise

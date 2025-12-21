import socket


def get_free_port() -> int:
    """
    Ref: https://stackoverflow.com/a/1365284/30487149
    """
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]

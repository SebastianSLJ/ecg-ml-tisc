import errno
import os
import select
import termios
import time


class SerialException(Exception):
    pass


_BAUD_RATES = {
    50: termios.B50,
    75: termios.B75,
    110: termios.B110,
    134: termios.B134,
    150: termios.B150,
    200: termios.B200,
    300: termios.B300,
    600: termios.B600,
    1200: termios.B1200,
    1800: termios.B1800,
    2400: termios.B2400,
    4800: termios.B4800,
    9600: termios.B9600,
    19200: termios.B19200,
    38400: termios.B38400,
    57600: termios.B57600,
    115200: termios.B115200,
    230400: termios.B230400,
    460800: termios.B460800,
    921600: termios.B921600,
}


class Serial:
    def __init__(self, port, baudrate, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._fd = None
        self._open()

    def _open(self):
        flags = os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK
        try:
            self._fd = os.open(self.port, flags)
        except OSError as exc:
            raise SerialException(exc.strerror or str(exc)) from exc

        try:
            attrs = termios.tcgetattr(self._fd)
            speed = _BAUD_RATES.get(self.baudrate)
            if speed is None:
                raise SerialException(f"Unsupported baud rate: {self.baudrate}")

            attrs[0] = 0
            attrs[1] = 0
            attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
            attrs[3] = 0
            attrs[4] = speed
            attrs[5] = speed
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
        except Exception as exc:
            self.close()
            if isinstance(exc, SerialException):
                raise
            raise SerialException(str(exc)) from exc

        try:
            os.set_blocking(self._fd, False)
        except AttributeError:
            pass

    def reset_input_buffer(self):
        if self._fd is None:
            raise SerialException("Serial port is not open")
        try:
            termios.tcflush(self._fd, termios.TCIFLUSH)
        except OSError as exc:
            raise SerialException(exc.strerror or str(exc)) from exc

    def readline(self):
        if self._fd is None:
            raise SerialException("Serial port is not open")

        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        buffer = bytearray()

        while True:
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            ready, _, _ = select.select([self._fd], [], [], remaining)
            if not ready:
                break

            try:
                chunk = os.read(self._fd, 1)
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    continue
                raise SerialException(exc.strerror or str(exc)) from exc

            if not chunk:
                break

            buffer.extend(chunk)
            if chunk == b"\n":
                break

        return bytes(buffer)

    def read(self, size=1):
        if self._fd is None:
            raise SerialException("Serial port is not open")

        deadline = None if self.timeout is None else time.monotonic() + self.timeout

        while True:
            try:
                chunk = os.read(self._fd, size)
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    chunk = b""
                else:
                    raise SerialException(exc.strerror or str(exc)) from exc

            if chunk:
                return chunk

            if deadline is not None and time.monotonic() >= deadline:
                return b""

            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            ready, _, _ = select.select([self._fd], [], [], remaining)
            if not ready:
                return b""

    def close(self):
        if self._fd is None:
            return
        try:
            os.close(self._fd)
        finally:
            self._fd = None

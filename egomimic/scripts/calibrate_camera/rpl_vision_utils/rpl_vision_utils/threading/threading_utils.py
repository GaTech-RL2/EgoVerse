import threading


class Worker(threading.Thread):
    def __init__(self):
        self._halt = False
        self._count = 0
        super().__init__()

    def halt(self):
        self._halt = True

    @property
    def count(self):
        return self._count

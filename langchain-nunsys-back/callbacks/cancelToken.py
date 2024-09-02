class CancelToken:
    def __init__(self):
        self._isCancelled = False

    def cancel(self):
        self._isCancelled = True

    @property
    def isCancelled(self):
        return self._isCancelled
import logging
import multiprocessing as mp
import traceback

LOGGER = logging.getLogger(__name__)


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        """ invokes the callable target argument in a seperate thread

        Overwridden to enable raising exceptions to the parent.
        """
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as exception:
            traceback_ = traceback.format_exc()
            traceback_package = (exception, traceback_)
            LOGGER.error(f"{traceback_}\n{exception}")
            self._cconn.send(traceback_package)
            raise exception

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

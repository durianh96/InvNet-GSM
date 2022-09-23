import datetime
import pandas as pd
import logging

EPS = 1e-5

pd.options.mode.chained_assignment = None

logging.getLogger('pyomo.core').setLevel(logging.INFO)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    filemode='a'
)

logger = logging.getLogger('gsm')
file_handler = logging.FileHandler('output.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)


def timer(method):
    def call(*args, **kw):
        time0 = datetime.datetime.now()
        r = method(*args, **kw)
        tm = datetime.datetime.now() - time0
        logger.debug("func @%s use %s (s)" % (method.__name__, tm.total_seconds()))
        return r

    return call


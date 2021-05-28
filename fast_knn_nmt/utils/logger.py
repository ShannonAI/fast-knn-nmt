
import logging
import datetime
import warnings


# def beijing(sec, what):
#     """Return beijing time"""
#     beijing_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
#     return beijing_time.timetuple()
#
#
# logging.Formatter.converter = beijing


PACKAGE_NAME = "fast_knn_nmt"


def init_root_logger(root_name=PACKAGE_NAME):
    # use 'bert_ner' as root logger name to prevent changing other modules' logger
    logger = logging.getLogger(root_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='[%(asctime)s.%(msecs)03d][%(levelname)s]<%(name)s> %(message)s',
        datefmt='%I:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


init_root_logger(PACKAGE_NAME)


def get_logger(name: str):
    if not name.startswith(PACKAGE_NAME):
        warnings.warn(f"logger name {name} should starts with {PACKAGE_NAME}, add automatically")
        name = f"{PACKAGE_NAME}.{name}"
    logger = logging.getLogger(name)
    return logger


import logging


module_logger = logging.getLogger("stNMR")
module_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(f"%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
module_logger.addHandler(stream_handler)

import logging


class EasySortLogger(logging.Logger):
    def __init__(self, name="EasySortLogger", level=logging.NOTSET):
        super().__init__(name, level)
        self.formatter = logging.Formatter("%(levelname)s (%(asctime)s - %(filename)s:%(lineno)d): %(message)s")
        self.formatter.datefmt = "%Y-%m-%d %H:%M:%S"

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        self.addHandler(console_handler)

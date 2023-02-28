import logging
from torch.utils.tensorboard import SummaryWriter

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

import time
class timer(object):
    def __init__(self) -> None:
        pass
    
    def logtime(self):
        return time.strftime('%a %b %d %H:%M:%S %Y', time.localtime())
    
    def filetime(self):
        return time.strftime("%yY%mM%dD%Hh%Mm%Ss", time.localtime())

    def get_hms(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        return h, m, s

    start_time = 0
    epoch_time = 0
    running_time = 0
    def setStart(self):
        self.start_time = time.time()
    
    def setEpoch(self):
        self.epoch_time = time.time() - self.start_time
        self.running_time += self.epoch_time

    def runtime(self):
        return '| Running time : %d:%02d:%02d'  %(self.get_hms(self.running_time))

def setup_writer(name):
    writer = SummaryWriter(name)
    return writer
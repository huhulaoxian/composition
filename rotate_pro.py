import numpy as np
import logging
import os

path = "weights/ft_baseline_CNN1_aug/"
logger = logging.getLogger('probs')
logger.setLevel(level=logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelno)s: %(message)s')

file_handler = logging.FileHandler(os.path.join(path, 'rotate_probs.log'),'w')
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger2 = logging.getLogger('probs2')
logger2.setLevel(level=logging.DEBUG)
formatter2 = logging.Formatter('%(asctime)s - %(levelno)s: %(message)s')
file_handler = logging.FileHandler(os.path.join(path, 'rotate_probs2.log'),'w')
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter2)
logger2.addHandler(file_handler)
logger2.addHandler(stream_handler)

logger3 = logging.getLogger('corrs')
logger3.setLevel(level=logging.DEBUG)
formatter3 = logging.Formatter('%(asctime)s - %(levelno)s: %(message)s')
file_handler = logging.FileHandler(os.path.join(path, 'rotate_corrs.log'),'w')
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter3)
logger3.addHandler(file_handler)
logger3.addHandler(stream_handler)
probs = np.load(path+"rotate_probs.npy")
for prob in probs:
    logger.info(prob)

probs2 = np.load(path+"rotate_probs2.npy")
for prob in probs2:
    logger2.info(prob)

corrs = np.load(path+"rotate_corrs.npy")
for corr in corrs:
    logger3.info(corr)

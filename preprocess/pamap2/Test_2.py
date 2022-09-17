import os

import h5py
import numpy as np
import tensorflow as tf
import yaml
def okk(output_file_name):
    f = h5py.File(output_file_name, 'w')
    dset = f.create_dataset("mydataset", (100,), dtype='i')

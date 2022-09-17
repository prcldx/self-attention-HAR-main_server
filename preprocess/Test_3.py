import os

import h5py
import numpy as np
import tensorflow as tf
import yaml
from preprocess.pamap2.Test_2 import okk
output_file_name = 'pamap2/p4.h5'
okk(output_file_name=output_file_name)
# f = h5py.File(output_file_name,'w')
# dset = f.create_dataset("mydataset", (100,), dtype='i')
print(1)
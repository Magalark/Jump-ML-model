#
# ELEC 292 Final Project - Group 53
# Created by Boyan Fan, Naman Nagia, Walker Yee on 03/28/2024
#

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with h5py.File('./dataset.h5', 'r+') as hdf:
    window_size=15  #set window size
    for i in range(141):    #iterate through all segments
        set=hdf.get(f'Dataset/Train/segment{i}') #get segment
        array=np.array(set) #convert segment to array
        dataset=pd.DataFrame(array) #convert array to dataframe to allow for moving average filter
        ma=dataset.rolling(window_size).mean()  #apply moving average filter
        hdf.pop(f'Dataset/Train/segment{i}')        #delete original segment
        updated_set=hdf.create_dataset(f'Dataset/Train/segment{i}',data=ma) #replace segment with processed segment


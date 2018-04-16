import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def get_df_last(n,PATH=Path('data')):
	dtypes = {
		'ip'			: 'uint32',
		'app'			: 'uint16',
		'device'		: 'uint16',
		'os'			: 'uint16',
		'channel'		: 'uint16',
		'is_attributed'	: 'uint8',
		'click_id'		: 'uint32'
	}

	total_rows= 184903980-90
	cols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
	return pd.read_csv(PATH / "train.csv", 
                    skiprows=range(1,total_rows-n + 1), 
                    nrows=n, 
                    usecols=cols,
                    dtype=dtypes)

def get_feather(fname,PATH=Path('data')):
	return pd.read_feather(PATH / fname)
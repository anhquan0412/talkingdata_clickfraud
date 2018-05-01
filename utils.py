import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gc
import time
import platform

from sklearn.metrics import roc_auc_score
import pickle


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




def mean_enc_df_export(df,cols,targ,new_fea,glob_mean,save_day,alpha=0):
	group_means = df.groupby(cols)[targ].mean()
	n_group=df.groupby(cols).size()

	new_fea = f'{new_fea}_prevday_alpha{alpha}'
	print(f'Generating {new_fea} for day {save_day}...')
	mean_df = (group_means*n_group + glob_mean*alpha)/ (n_group+alpha)
	mean_df= mean_df.reset_index().rename(columns={0:new_fea})
	filename= f'{new_fea}.feather'
	mean_df.to_feather( Path(f'data/mean_enc_df/day{save_day}') / filename)

def merge_mean_to_val(val_df,glob_mean,prev_day):
	path = Path(f'data/mean_enc_df/day{prev_day}')
	mean_filenames = [str(i) for i in list(path.iterdir())]
	for filename in mean_filenames:
		mean_df = pd.read_feather(filename)
		path_div = '\\' if platform.system()=='Windows' else '/'
		new_fea = filename.split(path_div)[-1].split('.')[0]
		cols = new_fea[: new_fea.find('mean')-1].split('_')
		print(f'Generating {new_fea} for validation set...')
		val_df = pd.merge(val_df,mean_df,'left',on=cols)
		val_df[new_fea].fillna(glob_mean,inplace=True)

	del mean_df
	gc.collect()

	return val_df

def groupby_agg(spec,X_train,X_val=None):
	# Name of the aggregation we're applying
	agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']

	# Info
	print(f"Grouping by {spec['groupby']}, and aggregating {spec['select']} with {agg_name}")

	# Unique list of features to select
	all_features = list(set(spec['groupby'] + [spec['select']]))

	# Name of new feature
	new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])

	# Perform the groupby
	if spec['select']==None and spec['agg']=='size':
		gp = X_train.groupby(spec['groupby']).size().reset_index().rename(columns={0:new_feature})
	else:  
		gp = X_train[all_features]. \
			groupby(spec['groupby'])[spec['select']]. \
			agg(spec['agg']). \
			reset_index(). \
			rename(index=str, columns={spec['select']: new_feature})

	# Merge back to X_train
	X_train = X_train.merge(gp, on=spec['groupby'], how='left')
	if not X_val is None:
		print(f'Generating {new_feature} for validation set...')
		X_val = X_val.merge(gp, on=spec['groupby'], how='left')
		X_val[new_feature].fillna(0,inplace=True)

	del gp
	gc.collect()

	return X_train,X_val

def cum_count(cols,df):
	new_fea = '_'.join(cols)+'_cumcount'
	df[new_fea]=df.groupby(cols).cumcount()+1
	df[new_fea] = df[new_fea].astype(np.int32)
	return df

def time_till_next_click(df,cols,new_fea):
	df[new_fea]= df.groupby(cols).click_time.transform(lambda x: x.diff()).dt.seconds
	df[new_fea].fillna(-1,inplace=True)
	df[new_fea] = df[new_fea].astype(np.int32)
	return df
def next_prev_click(df,cols,next_fea,prev_fea):
	df[clicktime_sec]=(df['click_time'].astype(np.int64) // (10**9)).astype(np.int32)
	df[next_fea] = (df.groupby(cols).clicktime_sec.shift(-1) - df.clicktime_sec).astype(np.float32)
	#df[next_fea].fillna(-1,inplace=True)
	df[prev_fea]=(df.click_sec - df.groupby(cols).clicktime_sec.shift(1)).astype(np.float32)
	#df[prev_fea].fillna(-1,inplace=True)

	df.drop('clicktime_sec',axis=1,inplace=True)
	gc.collect()
	return df

def time_feature(df):
	df['day'] = df['click_time'].dt.day.astype('uint8')
	df['hour'] = df['click_time'].dt.hour.astype('uint8')
	df['minute'] = df['click_time'].dt.minute.astype('uint8')
	df['second'] = df['click_time'].dt.second.astype('uint8')
	df.drop('click_time',axis=1,inplace=True)
	gc.collect()
	return df

def downcast_dtypes(df):
	'''
    Changes column types in the dataframe: 

        `float64` type to `float32`
        `int64`   type to `int32`
    '''

	# Select columns to downcast
	float_cols = [c for c in df if df[c].dtype == "float64"]
	int_cols =   [c for c in df if df[c].dtype == "int64"]
	# Downcast
	df[float_cols] = df[float_cols].astype(np.float32)
	df[int_cols]   = df[int_cols].astype(np.int32)

	return df

def get_cv_idxs(n,portion):
	val_idxs=[]
	idxs = np.random.permutation(n)
	s,e= 0,n//portion
	for i in range(portion-1):
		val_idx = sorted(idxs[s:e])
		val_idxs.append(val_idx)
		s=e
		e+=n//portion

	val_idxs.append(sorted(idxs[s:]))
	return val_idxs

def get_sample_timeseries(filename,sz):
	df = get_feather(filename)
	sample_idx =np.random.permutation(df.shape[0])
	sample_idx=sorted(sample_idx[:sz])
	df = df.loc[sample_idx,:].reset_index().drop('index',axis=1)
	gc.collect()
	return df

def prediction_score(rf,train_df,y_train,val_df,y_val):
	y_train_pred = rf.predict_proba(train_df)[:,1]
	print(f'Train AUC: {roc_auc_score(y_train, y_train_pred)}')
	y_val_pred = rf.predict_proba(val_df)[:,1]
	val_auc = roc_auc_score(y_val, y_val_pred)
	print(f'Val AUC: {roc_auc_score(y_val, y_val_pred)}')
	return val_auc
def get_val_by_name(cols_to_drop,name):
	val_df = pd.read_feather(name)
	y_val = val_df.is_attributed
	val_df.drop(cols_to_drop,axis=1,inplace=True)
	gc.collect()
	return val_df,y_val
def get_train(cols_to_drop,sz=3000000):
	train_df=get_sample_timeseries('train_day8_3to16_FE.feather',sz)
	y_train = train_df.is_attributed
	train_df.drop(cols_to_drop,axis=1,inplace=True)
	gc.collect()
	return train_df,y_train
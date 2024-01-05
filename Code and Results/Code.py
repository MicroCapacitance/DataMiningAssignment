import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pytz
import math


def plot_evaluation(env):
	iter_count = env.iteration + 1
	print("no_____________________________")
	if iter_count % 100 == 0:
		ax = lgb.plot_metric(model, metric='loss', figsize=(12, 6))
		plt.show()
		lgb.plot_importance(model, importance_type='gain', max_num_features=20)
		plt.show()

def reduce_mem_usage(df, verbose=True):
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	start_mem = df.memory_usage().sum() / 1024**2
	for col in df.columns:
		col_type = df[col].dtypes
		if col_type in numerics:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)
	end_mem = df.memory_usage().sum() / 1024**2
	if verbose:
		print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
	return df


output_path = ''
data_path = ''


def no_use():
	print('sku_info.csv', '-'*50)
	info_df = pd.read_csv(data_path + 'sku_info.csv')
	print(info_df.head())
	print(info_df.shape)
	print(info_df.info())

	print('sku_price_and_status.csv', '-'*50)
	data_sku_price_and_status = pd.read_csv(data_path + 'sku_price_and_status.csv')
	print(data_sku_price_and_status.head())
	print(data_sku_price_and_status.shape)
	print(data_sku_price_and_status.info())

	print('sku_prom.csv', '-'*50)
	prom_df = pd.read_csv(data_path + 'sku_prom.csv')
	print(prom_df.head())
	print(prom_df.shape)
	print(prom_df.info())

	print('sku_sales.csv', '-'*50)
	sales_df = pd.read_csv(data_path + 'sku_sales.csv')
	print(sales_df.head())
	print(sales_df.shape)
	print(sales_df.info())

	print('store_weather.csv', '-'*50)
	weather_df = pd.read_csv(data_path + 'store_weather.csv')
	print(weather_df.head())
	print(weather_df.shape)
	print(weather_df.info())

	print('test_item.csv', '-'*50)
	test_df = pd.read_csv(data_path + 'test_item.csv')
	print(test_df.head())
	print(test_df.shape)
	print(test_df.info())

	sales_df['order_time'] = sales_df['order_time'].apply(lambda x: x[:10])
	sales_df = sales_df.groupby(['store_id', 'order_time', 'sku_id', 'channel'], as_index=False)['quantity'].sum()
	print(sales_df.head())
	print(sales_df.info())
	print(sales_df.shape)
	sales_df.to_parquet(output_path + 'sales_sum.parquet', index=False)

	sales_df = pd.read_parquet(output_path + 'sales_sum.parquet')
	print(sales_df['quantity'].sum())
	sales_df = sales_df.pivot_table(
		index=['store_id', 'sku_id', 'order_time'],
		columns=['channel'],
		values=['quantity'],
		fill_value=0).reset_index()
	sales_df.columns= ['store_id', 'sku_id', 'order_time', 'x_k', 'x_m']
	print(sales_df.head())
	print(sales_df.columns)
	print(sales_df['x_k'].sum()+sales_df['x_m'].sum())
	test_df = pd.read_csv(data_path + 'test_item.csv')
	test_df['x_k'] = 0
	test_df['x_m'] = 0
	test_df = test_df.rename(columns={'date': 'order_time'})
	train_df = pd.concat([sales_df, test_df], axis=0)
	train_df['x'] = train_df['x_k'] + train_df['x_m']
	train_df['order_time'] = pd.to_datetime(train_df['order_time'])
	train_df = train_df.sort_values(['store_id', 'sku_id', 'order_time'])
	train_df = train_df.reset_index(drop=True)
	print(train_df.head())
	print(train_df.info())
	print(train_df.shape)
	train_df.to_parquet(output_path + 'train_df.parquet', index=False)

	# x lag features
	train_df = pd.read_parquet(output_path + 'train_df.parquet')
	train_df = train_df.rename(columns={'order_time': 'date'})
	predict_days = 14
	for col in ['x', 'x_k', 'x_m']:
		for s_days in range(predict_days, predict_days+15):
			train_df[f'{col}_lag_{s_days}'] = train_df.groupby(['store_id', 'sku_id'])[col].transform(
				lambda x: x.shift(s_days))
	for col in ['x', 'x_k', 'x_m']:
		for r_days in [3, 5, 7, 14, 21, 28, 30, 60, 90]:
			train_df[f'{col}_rolling_mean_{r_days}'] = train_df.groupby(['store_id', 'sku_id'])[col].transform(
				lambda x: x.shift(predict_days).rolling(r_days, min_periods=1).mean())
			train_df[f'{col}_rolling_std_{r_days}'] = train_df.groupby(['store_id', 'sku_id'])[col].transform(
				lambda x: x.shift(predict_days).rolling(r_days, min_periods=1).std())
			train_df[f'{col}_rolling_mean_change_{r_days}'] = train_df.groupby(['store_id', 'sku_id'])[col].transform(
				lambda x: x.shift(predict_days).rolling(r_days, min_periods=1).mean().fillna(method='ffill').pct_change())
	print(train_df.head())
	print(train_df.info())

	# train_df = reduce_mem_usage(train_df)
	# train_df.to_pickle(output_path + 'info_x_lag_features.pkl')
	#
	# print("yesyesyes")

	train_df = train_df.drop(['x', 'x_k', 'x_m'], axis=1)
	train_df = reduce_mem_usage(train_df)
	train_df.to_pickle(output_path + 'x_lag_features.pkl')

	# info x lag features
	pd.options.display.max_columns = None
	train_df = pd.read_parquet(output_path + 'train_df.parquet')
	train_df = train_df.rename(columns={'order_time': 'date'})
	info_df = pd.read_csv(data_path + 'sku_info.csv')
	train_df = train_df.merge(info_df, on='sku_id', how='left')
	for col in ['item_first_cate_cd', 'item_second_cate_cd', 'item_third_cate_cd', 'brand_code']:
		g_list = ['store_id']+[col]+['date']
		tmp = train_df.groupby(g_list, as_index=False)[['x', 'x_k', 'x_m']].sum()
		tmp = tmp.sort_values(g_list)
		predict_days = 14
		for target in ['x', 'x_k', 'x_m']:
			for s_days in range(predict_days, predict_days+15):
				tmp[f'{col}_{target}_lag_{s_days}'] = tmp.groupby(['store_id']+[col])[target].transform(
					lambda x: x.shift(s_days))
		for target in ['x', 'x_k', 'x_m']:
			for s_days in [predict_days]:
				for r_days in [3, 5, 7, 14, 21, 28]:
					tmp[f'{col}_{target}_rolling_mean_{r_days}'] = tmp.groupby(['store_id']+[col])[target].transform(
						lambda x: x.shift(s_days).rolling(r_days,min_periods=1).mean())
					tmp[f'{col}_{target}_rolling_std_{r_days}'] = tmp.groupby(['store_id']+[col])[target].transform(
						lambda x: x.shift(s_days).rolling(r_days,min_periods=1).std())
		tmp = tmp.drop(['x', 'x_k', 'x_m'], axis=1)
		train_df = train_df.merge(tmp, on=g_list, how='left')

	train_df = train_df.drop(['x', 'x_k', 'x_m'], axis=1)
	print(train_df.head())
	print(train_df.info())

	train_df = reduce_mem_usage(train_df)
	train_df.to_pickle(output_path + 'info_x_lag_features.pkl')

	# features merge
	train_df = pd.read_parquet(output_path + 'train_df.parquet')
	train_df = train_df.rename(columns={'order_time': 'date'})
	data_sku_price_and_status = pd.read_csv(data_path + 'sku_price_and_status.csv')

	data_sku_price_and_status['date'] = pd.to_datetime(data_sku_price_and_status['date'])
	# 按照store_id, sku_id进行分组，然后补齐从最小日期到最大日期的数据
	data_sku_price_and_status = data_sku_price_and_status.groupby(["store_id", "sku_id"]).apply(lambda x: x.set_index("date").resample("D").ffill()).drop(["store_id", "sku_id"], axis=1).reset_index()
	data_sku_price_and_status['date'] = pd.to_datetime(data_sku_price_and_status['date']).dt.date
	data_sku_price_and_status.drop(["salable_status", "stock_status"], axis=1, inplace=True)
	# 原价格的统计量
	for operation in ['max', 'min', 'mean', 'std', 'median', 'skew']:
		data_sku_price_and_status['price_' + operation] = data_sku_price_and_status.groupby(['store_id', 'sku_id'])['original_price'].transform(operation)
	# 原价格的分位数
	for quantile in [0.25, 0.75, 0.50, 0.05, 0.95, 0.10, 0.90]:
		data_sku_price_and_status['price_quantile_' + str(int(quantile * 100))] = data_sku_price_and_status.groupby(['store_id', 'sku_id'])['original_price'].transform(lambda x: x.quantile(quantile))
	# 归一化
	data_sku_price_and_status['price_norm'] = data_sku_price_and_status['original_price'] / data_sku_price_and_status['price_max']
	# 商品的价格数量（商品有多少种价格）
	data_sku_price_and_status['price_nunique'] = data_sku_price_and_status.groupby(['store_id', 'sku_id'])['original_price'].transform('nunique')
	# 相同价格的商品数量
	data_sku_price_and_status['sku_nunique'] = data_sku_price_and_status.groupby(['store_id', 'original_price'])['sku_id'].transform('nunique')
	# 价格变化
	data_sku_price_and_status['price_momentum'] = data_sku_price_and_status['original_price'] / data_sku_price_and_status.groupby(['store_id', 'sku_id'])['original_price'].transform(lambda x: x.shift(1)).fillna(0)
	# 价格在全年、全月的变化
	data_sku_price_and_status['year'] = pd.to_datetime(data_sku_price_and_status['date']).dt.year
	data_sku_price_and_status['month'] = pd.to_datetime(data_sku_price_and_status['date']).dt.month
	data_sku_price_and_status['price_momentum_m'] = data_sku_price_and_status['original_price'] / data_sku_price_and_status.groupby(['store_id', 'sku_id', 'month'])['original_price'].transform('mean')
	data_sku_price_and_status['price_momentum_y'] = data_sku_price_and_status['original_price'] / data_sku_price_and_status.groupby(['store_id', 'sku_id', 'year'])['original_price'].transform('mean')

	prom_df = pd.read_csv(data_path + 'sku_prom.csv')
	prom_df['date'] = pd.to_datetime(prom_df['date'])

	x_k = train_df[['store_id', 'sku_id', 'date', 'x_k']]
	x_k = x_k.rename(columns={'x_k': 'quantity'})
	x_k['channel'] = 1
	x_m = train_df[['store_id', 'sku_id', 'date', 'x_m']]
	x_m = x_m.rename(columns={'x_m': 'quantity'})
	x_m['channel'] = 2
	x = pd.concat([x_k, x_m], axis=0, ignore_index=True)

	# price features
	x = x.merge(data_sku_price_and_status, on=['store_id', 'sku_id', 'date'], how='left')
	tmp = x[x['original_price'].isna()]
	print(set(tmp['sku_id']))
	# prom features
	x = x.merge(prom_df, on=['store_id', 'sku_id', 'date', 'channel'], how='left')
	x['discount_off'] = x['discount_off'].fillna(0.0)
	# lag_features
	x_lag_features = pd.read_pickle(output_path + 'x_lag_features.pkl')
	#x_lag_features = x_lag_features.drop(['x', 'x_k', 'x_m'], axis=1)
	x = x.merge(x_lag_features, on=['store_id', 'sku_id', 'date'], how='left')
	# info lag features
	info_x_lag_features = pd.read_pickle(output_path + 'info_x_lag_features.pkl')
	x = x.merge(info_x_lag_features, on=['store_id', 'sku_id', 'date'], how='left')
	# weather features
	weather_df = pd.read_csv(data_path + 'store_weather.csv')
	weather_df['date'] = pd.to_datetime(weather_df['date'])
	x = x.merge(weather_df, on=['store_id', 'date'], how='left')
	# time features
	x['day'] = x['date'].dt.day.astype('int')
	x['week'] = x['date'].dt.dayofweek.astype('int')
	x['weekend'] = (x['week'] >= 5).astype('int')
	x['weeknum'] = x['date'].dt.isocalendar().week.astype('int')
	x['month'] = x['date'].dt.month.astype('int')
	x['quarter'] = x['date'].dt.quarter.astype('int')
	x['year'] = x['date'].dt.year.astype('int')
	x['year'] = x['year'] - x['year'].min()
	x['dayofyear'] = x['date'].dt.dayofyear.astype('int')
	x['is_month_start'] = x['date'].dt.is_month_start.astype('int')
	x['is_month_end'] = x['date'].dt.is_month_end.astype('int')
	x['is_quarter_start'] = x['date'].dt.is_quarter_start.astype('int')
	x['is_quarter_end'] = x['date'].dt.is_quarter_end.astype('int')
	x['is_year_start'] = x['date'].dt.is_year_start.astype('int')
	x['is_year_end'] = x['date'].dt.is_year_end.astype('int')
	x['days_in_month'] = x['date'].dt.days_in_month.astype('int')
	x['is_leap_year'] = x['date'].dt.is_leap_year.astype('int')

	for col in [
		'channel',
		'promotion_id', 'promotion_type', 'item_first_cate_cd',
		'item_second_cate_cd', 'item_third_cate_cd', 'brand_code', 'weather_type'
	]:
		x[col] = x[col].astype('category')

	x = reduce_mem_usage(x)
	x.to_pickle(output_path + 'train_dataset.pkl')
	print(x.info())
	print(x.shape)
	for col in x.columns:
		print(col, x[col].dtype, round(x[col].isna().sum() / len(x), 2))
		print('-' * 50)
	train_df.to_csv('baseline_result.csv',index = False)
import lightgbm as lgb
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

SEED = 2023
output_path = ''
remove_cols = [
	'salable_status', 'stock_status',
	'store_id', 'sku_id', 'date', 'quantity', 'promotion_id'
]
cate_features = []
for col in [
	'channel',
	'promotion_id', 'promotion_type', 'item_first_cate_cd',
	'item_second_cate_cd', 'item_third_cate_cd', 'brand_code', 'weather_type'
]:
	if col not in remove_cols:
		cate_features.append(col)
train_all = pd.read_pickle(output_path + 'train_dataset.pkl')
print(train_all.shape)
feature_cols = [col for col in train_all.columns if col not in remove_cols]

sub_df = []
valid_df = []
for store_id in range(1, 13):
	print('store_id :', store_id, '-' * 50)
	train_store = train_all[train_all['store_id'] == store_id].copy()
	test = train_store[train_store['date'] >= pd.to_datetime('2023-09-01')]
	train_store = train_store[train_store['date'] < pd.to_datetime('2023-09-01')]
	valid = train_store[train_store['date'] >= pd.to_datetime('2023-08-01')]
	train = train_store[train_store['date'] < pd.to_datetime('2023-08-01')]
	train_data = lgb.Dataset(
		train[feature_cols],
		label=train['quantity']
	)
	valid_data = lgb.Dataset(
		valid[feature_cols],
		label=valid['quantity']
	)
	train_data_all = lgb.Dataset(
		train_store[feature_cols],
		label=train_store['quantity']
	)

	random.seed(SEED)
	np.random.seed(SEED)
	lgb_parmas = {}
	lgb_parmas['seed'] = SEED
	lgb_parmas['objective'] = 'regression'
	# 'tweedie'分布超参数p
	#     lgb_parmas['tweedie_variance_power'] = 1.1
	lgb_parmas['verbose'] = -1
	# 以下，模型超参数调整
	lgb_parmas['metric'] = 'rmse'
	lgb_parmas['learning_rate'] = 0.03
	lgb_parmas['subsample'] = 0.7
	lgb_parmas['feature_fraction'] = 0.7
	lgb_parmas['device'] = 'gpu'
	lgb_parmas['gpu_device_id'] = 0
	lgb_parmas['gpu_use_dp'] = True
	print("yes__________________")
	model = lgb.train(
		lgb_parmas,
		train_data,
		valid_sets=[train_data, valid_data],
		num_boost_round=800,
		callbacks=[
			lgb.early_stopping(20),
			lgb.log_evaluation(period=50),
			plot_evaluation
		],
		categorical_feature=cate_features,
	)
	# ax = lgb.plot_metric(model, metric='loss', figsize=(12, 6))
	# # plt.show()
	# lgb.plot_importance(model, importance_type='gain', max_num_features=20)
	# plt.show()

	valid['quantity_pred'] = model.predict(valid[feature_cols])
	valid_df.append(valid)

	best_iteration = model.best_iteration
	model = lgb.train(
		lgb_parmas,
			train_data_all,
			num_boost_round=best_iteration,
			categorical_feature = cate_features
		)

	test['quantity'] = model.predict(test[feature_cols])
	test = test[['store_id', 'sku_id', 'date', 'quantity', 'channel']]
	sub_df.append(test)

sub_df = pd.concat(sub_df, axis=0, ignore_index=True)
sub_df.to_csv(output_path+'submission_model.csv', index=False)
valid_df = pd.concat(valid_df, axis=0, ignore_index=True)
valid_df.to_pickle(output_path+'valid.pkl')

import pandas as pd
from datetime import datetime
import pytz
import math
import os


sub = pd.read_csv(output_path+'submission_model.csv')
sub = sub.query('channel <= 2')
sub = sub.pivot_table(
	index=['store_id', 'sku_id', 'date'],
	columns=['channel'],
	values=['quantity'],
	fill_value=0).reset_index()
sub.columns= ['store_id', 'sku_id', 'date', 'x_k', 'x_m']
sub['x_k'] = sub['x_k'].apply(lambda x: math.ceil(x) if x >0 else 0)
sub['x_m'] = sub['x_m'].apply(lambda x: math.ceil(x) if x >0 else 0)

print(sum(sub['x_k'])+sum(sub['x_m']))
sub['flag'] = sub.apply(lambda x: 1 if (x['x_k'] +x['x_m']>0)&(x['x_k']<=0) else 0, axis=1)
sub['x_k'] = sub.apply(lambda x: x['x_k'] +x['x_m'] if x['flag']==1 else x['x_k'], axis=1)
sub['x_m'] = sub.apply(lambda x: 0 if x['flag']==1 else x['x_m'], axis=1)
sub = sub.drop('flag', axis=1)

print(sum(sub['x_k'])+sum(sub['x_m']))
sub['x'] = sub['x_k'] + sub['x_m']
sub['x_m_flag'] = sub['x_m'].apply(lambda x:1 if x>0 else 0)
x_sum = sub.groupby(['store_id', 'date'], as_index=False).agg({'x': 'sum'})
sub = sub.drop(['x'], axis=1)
sub = sub.merge(x_sum, on=['store_id', 'date'], how='left')
sub = sub.sort_values(['store_id', 'date', 'x_m'], ascending=[True, True, False])
sub['x_m_cumsum'] = sub.groupby(['store_id', 'date'])['x_m'].cumsum()
sub['x_m_flag_cumsum'] = sub.groupby(['store_id', 'date'])['x_m_flag'].cumsum()
sub['flag'] = ((sub['x_m_cumsum'] <= (sub['x'] * 0.4)) & (sub['x_m_flag_cumsum'] <=200)).astype('int')
print(sum(sub['x_k'])+sum(sub['x_m']))
sub['x_k'] = sub.apply(lambda x: x['x_k'] +x['x_m'] if x['flag'] == 0 else x['x_k'], axis=1)
sub['x_m'] = sub.apply(lambda x: 0 if x['flag'] == 0 else x['x_m'], axis=1)
print(sum(sub['x_k'])+sum(sub['x_m']))
sub = sub.drop(['x_m_flag', 'x', 'x_m_cumsum', 'x_m_flag_cumsum', 'flag'], axis=1)
# print(sub.head())
# print(sub.info())
test_df = pd.read_csv(data_path + 'test_item.csv')
# print(test_df.head())
# print(test_df.info())
test_df = test_df.merge(sub, on=['store_id','sku_id', 'date'], how='left')
print(test_df.head())
print(sum(test_df['x_k'])+sum(test_df['x_m']))
total = sum(test_df['x_k'])+sum(test_df['x_m'])
test_df.to_csv(
	output_path + 'submission_final_{}_{}.csv'.format(total, datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%m%d%H%M')),
	index=False)



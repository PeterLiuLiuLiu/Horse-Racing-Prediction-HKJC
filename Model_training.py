# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:47:05 2020

@author: raymond-cy.liu
"""

import pandas as pd
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import operator

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import seaborn as sns
plt.rcParams.update({'font.size':3})



class Model:
	
	def __init__(self, df):
		self.df = df
		self.num_df = df.copy()
		
		
# 		COMMENT OUT FOR KEEPING IN MODEL TRAINNING
		self.columns = [
			'Year', 'Month', 'Day', 'Horse', 'Jockey', 'Trainer', 
			'Gear', 'Best Time', 'Over Wt.', 'Priority',
			
# =============================================================================
# # 			Factors specifically for particular horses
# 			'Actual Wt.', 
# 			'Declar. Horse Wt.', 
# 			'Draw', 
# 			'Age',
# 			'Last race day', 
# 			'Weight diff.', 
# 			'Last plc.',
# 			'new',
# 			'score', 
# 			'improve',
# 			'5_avg_score',
# 			'10_avg_score',
# =============================================================================
	
			# Newly added feature 2020-08-18
			'Wt.+/- (vs Declaration)',
			'Rtg.',
			'Jockey Rtg.+/-',
			
			'BT',
			'gallop',
			'swimming',
			'trackwork_total',
			'trotting',
			'Rtg.+/-', # THATS REALLY USEFUL/DOUBTFUL
			'Season Stakes', # THATS REALLY USEFUL/DOUBTFUL

# =============================================================================
#  			# Additional info for specific horse performance
# 			'Jockey_Elo', 
# 			'Trainer_Elo',
# =============================================================================
	
# 		Factors same for ALL horses
			'class',
			'distance', 
			'location', 
			'condition', 
			'track', 
			'N',
			
			# Weather
			'temp', 
			'baro', 
			'wind', 
			'wd', 
			'hum', 
			
# 			HMMMMMM use with CAUTION, difficult to obtain in real case
			'Win Odds',
			# Not to be used since its zero except 1, 2 and 3 places horses
			'Place Odds', 
			
# 			The results, not know beforehand, only use for analysis in the future
			'Plc.', 
			'Finish Time', 
			'per_km_time', 
			'TBW', 
			'Plc. score', 
			'race_diff',
			'public_est_diff',
			'run_pos_initial',
			'run_pos_max', 
			'run_pos_min', 
 			'run_pos_mean', 
 			'run_pos_range', 
			'run_pos_std', 
 			'run_pos_min_slope',
 			'run_pos_max_slope',
				]

	
	def get_encoding(self):
		# Categorical encoding or one hot?
		# One hot first
		columns = [
				'Priority',
				'location', 
				'condition', 
				'track',
				]
		for col in columns:
			if col not in self.columns:
				encode_df = pd.get_dummies(self.num_df[col], prefix = col)
				self.num_df = pd.concat([self.num_df, encode_df], axis = 1).drop(col, axis = 1)
	
	
	def drop_columns(self, df):
		return df.drop(self.columns, axis = 1)
		
	
	def scale_data(self, df, method = 'race'):
		columns = df.columns
		
		if method == 'race':
	# 		Avoid dropping encoded columns
			not_scale = df.nunique()[df.nunique() == 2].index
			not_scale_df = df[not_scale]
			not_scale = not_scale.append(pd.Index(['Race_no']))
			df_ = np.array([])
			scaler = preprocessing.StandardScaler()
	# 		Dropping Race_no now for better standardization of the data PER RACE
			print('\nScaling data per race...')
			for race in tqdm(df['Race_no'].unique(), position = 0):
				temp_df = scaler.fit_transform(df[df['Race_no'] == race].drop(not_scale, axis = 1))
				df_ = np.vstack((df_, temp_df)) if len(df_) > 0 else temp_df
			df_ = pd.DataFrame(df_, columns = columns.drop(not_scale))
			df_ = pd.concat([df_, not_scale_df], axis = 1)
			
		elif method == 'all':
			print('\nScaling data in all races...')
			not_scale = df.nunique()[df.nunique() == 2].index
			not_scale_df = df[not_scale]
			not_scale = not_scale.append(pd.Index(['Race_no']))
			df_train = df.drop(not_scale, axis = 1).iloc[ : self.test_race]
			df_test = df.drop(not_scale, axis = 1).iloc[self.test_race : ]
			scaler = preprocessing.MinMaxScaler()
			scaler.fit(df_train)
			df_train = scaler.transform(df_train)
			df_test = scaler.transform(df_test)
			df_ = np.vstack((df_train, df_test))
			df_ = pd.DataFrame(df_, columns = columns.drop(not_scale))
			df_ = pd.concat([df_, not_scale_df], axis = 1)
			
		return df_

	def check_na(self):
		if len(self.num_df[self.num_df.isnull().any(axis=1)]) == 0:
			return True
		else:
			print('Null value exists in data, please check!')
			return False
		
		
	def make_label(self):
		self.num_df['label'] = np.nan
		index = self.num_df[self.num_df['Plc.'] == 1].index
		self.num_df['label'].loc[index] = 1
		self.num_df.fillna(0, inplace = True)
		
		
	def train_test_split(self):	
		ratio = self.test_split
		race = self.num_df['Race_no'].unique()
		self.test_race = self.num_df[self.num_df['Race_no'] == math.floor(len(race) * (1 - ratio))].index[0]
		self.get_encoding()
		y = self.num_df['label']
		
# 		To make dates for reference
		a, b, c = self.num_df[['Year', 'Month', 'Day']].loc[self.test_race:].iloc[0].values
		self.st_date = f'{a}/{b}/{c}'
		a, b, c = self.num_df[['Year', 'Month', 'Day']].loc[self.test_race:].iloc[-1].values
		self.end_date = f'{a}/{b}/{c}'
		
# 		To create location of races 1st plc for distinguishing races
		test_index = y.loc[self.test_race:][y.loc[self.test_race:] == 1].index
		self.race_index = test_index.append(pd.Index([len(y)])) - test_index[0] # make sure it loops till the end later on
		self.win_odds = self.num_df[['Race_no', 'Plc.', 'Win Odds']].loc[self.test_race:].reset_index(drop = True)
		indexes = pd.Index([])
		index_to_remove = []
		
		for i in self.win_odds['Race_no'].unique():
			if self.win_odds[self.win_odds['Race_no'] == i]['Plc.'].value_counts().loc[1] > 1:
				# Divide the odds by two for actual rewards of bets while public odds maintains doubled
				indexes = indexes.append(self.win_odds[self.win_odds['Race_no'] == i].index[:2])
				index_to_remove.append(self.win_odds[self.win_odds['Race_no'] == i].index[1])
# 		Need to adjust the actual payout? not know before race
		self.win_odds['return'] = self.win_odds['Win Odds']
		self.win_odds['return'].loc[indexes] = self.win_odds['Win Odds'].loc[indexes] / 2
		
		self.race_index = self.race_index.drop(index_to_remove)

		self.public_result = self.num_df['Win Odds'].loc[self.test_race:].reset_index(drop = True).pow(-1) * (1-0.17) # 17% take from HKJC, jesus SO HIGH!
		self.actual = y.loc[self.test_race:].reset_index(drop = True)
		self.new_horse = self.num_df['new'].loc[self.test_race:].reset_index(drop = True)
		
		X = self.num_df.drop('label', axis = 1)
		yr, m, d = X.loc[self.test_race][['Year', 'Month', 'Day']]
		print(f'Testing from {yr}_{m}_{d}')
		X = self.drop_columns(X)
		X = self.scale_data(X, method = 'all')
		return X.loc[:self.test_race], X.loc[self.test_race:], y.loc[:self.test_race], y.loc[self.test_race:]
	
	def process_data(self):
		if self.check_na() == False:
			return
		self.make_label()
		X_train, X_test, y_train, y_test = self.train_test_split()
		return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
	
	
	def normalize_result(self, result, dim):
		norm_result = pd.DataFrame()
		result = [result[i][1] for i in range(len(result))] if dim == 2 else [result[i] for i in range(len(result))]
		st = self.race_index[0]
		for end in self.race_index[1:]:
			temp_result = result[st:end] / sum(result[st:end])
			pd.DataFrame(temp_result)
			norm_result = pd.DataFrame(temp_result) if norm_result.empty else norm_result.append(pd.DataFrame(temp_result))
			st = end
		return norm_result.reset_index(drop = True)
	
	
	def concat_bin(self, factor, df, compare_model_bias):
		cut_bins = [0, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 1]
		df['range'] = pd.cut(df[factor], bins = cut_bins)
		df['index'] = df.index
		df.set_index('range', inplace = True)
		data_dict = dict()
# 		NOT TO BE CHANGED
		if compare_model_bias:
			data_dict[1] = df[df['fundamental'] <= df['public']].sort_index()
			data_dict[-1] = df[df['fundamental'] > df['public']].sort_index()
		else:
			data_dict[0] = df.sort_index()
		return data_dict
	
	
	def r2_score(self, factor):
			fit = self.combined_df['actual'] - self.combined_df[factor]
			random = self.combined_df['actual'] - self.combined_df['random']
			LL_fit = fit.abs().sum()
			LL_mean = random.abs().sum()
			r2_score = 1 - (LL_fit / LL_mean)
			return r2_score


	def build_NNmodel(self, dim):
		model = keras.Sequential()
		model.add(layers.Dense(64, activation = 'relu', input_dim = dim))
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(64, activation = 'relu'))
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(1, activation = 'sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model


	def train(self, test_split):
		
		self.test_split = test_split
		X_train, X_test, y_train, y_test = self.process_data()
		
# =============================================================================
# 		clf = RandomForestClassifier(n_estimators = 1000)
# 		clf.fit(X_train, y_train)
# =============================================================================

		clf = LogisticRegression(max_iter = 50000, random_state = None)
		clf.fit(X_train, y_train)
		self.model_result = clf.predict_proba(X_test)
		self.model_result = self.normalize_result(self.model_result, dim = 2)
		
# =============================================================================
# 		model = self.build_NNmodel(X_train.shape[1])
# 		history = model.fit(X_train, y_train, epochs = 20, validation_split = 0.1)
# 		self.model_result = model.predict(X_test).flatten() # turn into dim 1
# 		self.model_result = self.normalize_result(self.model_result, dim = 1)
# =============================================================================

		self.public_result = self.normalize_result(np.array(self.public_result), dim = 1)
		
		
	def analysis_result(self, pub_weight = 0, incorp_pub_step = 0, adjust_Z = 0):
		
		self.pub_weight = pub_weight
		self.adjust_Z = adjust_Z
		
# 		Can tidy up by another function
		print(f'{len(self.model_result)} nos. of records in total') 
		dp = 3
		df_ = pd.DataFrame()
		# 1 = 'public LARGER THAN fundamental', -1 = 'public SMALLER THAN fundamental, which 1 makes betting liekly, treat with cautionTREAT WITH CAUTION'
		temp = pd.concat([self.model_result, self.public_result, self.actual, self.new_horse], axis = 1)
		temp.columns = ['fundamental', 'public', 'actual', 'new']
		temp['fundamental'] = temp['fundamental'].apply(lambda x: min(max(x, 0.001), 0.999)) # strict between 0, 1 for model predictions
		index = dict()
		model = self.concat_bin('fundamental', temp, compare_model_bias = True)
		for j, df in model.items():
			index[j] = df['index']
			ranges = df.index.unique()
			for i in ranges:
				temp_df = df[df.index == i]
				adjust_Z = adjust_Z if i != ranges[-1] else adjust_Z
				# It adjusts the error here, 0: model <= public, 1: model > public
				temp_df['fundamental'] += j *adjust_Z * temp_df['fundamental'].std()
				temp_df['fundamental'] = temp_df['fundamental'].apply(lambda x: min(max(x, 0.001), 0.999)) # strict between 0, 1
				df_ = temp_df if df.empty else df_.append(temp_df)
			
		print('\nComparing fundamental model and public estimates...')
		self.combined_df = df_.copy()
		self.combined_df['range'] = self.combined_df.index
		self.combined_df = self.combined_df.set_index('index').sort_index()
		new_horse_index = self.combined_df[self.combined_df['new'] == 1].index
# 		To assign the new horse with public expectation
		self.combined_df['fundamental'].loc[new_horse_index] = self.combined_df['public'].loc[new_horse_index]
		
		self.combined_df['fundamental'] = self.normalize_result(self.combined_df['fundamental'], 1)
		self.combined_df['race'] = 0
		self.combined_df['race'] = self.combined_df.groupby('race').cumsum()['actual']
		for j in model.keys():
			model[j] = self.combined_df.loc[index[j].values]
		for j, df in model.items():
			print('\npublic estimates LARGER THAN fundamental model estimates') if j == 1 else print('\n*****public estimates SMALLER THAN fundamental model estimates*****')
			data_dict = dict()
			benchmark = {1: 0.02, -1: -0.0159} # from bill benter (1994)
			df.set_index('range', inplace = True)
			for i in df.index.unique():
				temp_df = df[df.index == i]
				pub_exp_std = temp_df['public'].std()
				pub_exp_mean = temp_df['public'].mean()
				model_exp_std = temp_df['fundamental'].std()
				model_exp_mean = temp_df['fundamental'].mean()
				act_mean = temp_df['actual'].mean()
				Z_pub = (act_mean - pub_exp_mean) / pub_exp_std
				Z_model = (act_mean - model_exp_mean) / model_exp_std
				data_dict[i] = [len(temp_df), round(act_mean, dp), round(model_exp_mean, dp), round(Z_model, dp), round(pub_exp_mean, dp), round(Z_pub, dp)]
			result_df = pd.DataFrame.from_dict(data_dict, orient = 'index', columns = ['N', 'Actual_frequency', 'model_exp.', 'model_Z', 'public_exp.', 'public_Z']).sort_index()
			print(result_df)
			model = (result_df['Actual_frequency'] - result_df['model_exp.']) * result_df['N']
			public = (result_df['Actual_frequency'] - result_df['public_exp.']) * result_df['N']
			print(f'Model deviation from actual result: {round(model.sum() / len(df), dp+1)}, \nPublic deviation from actual result: {round(public.sum() / len(df), dp+1)} \nBenchmark before win odds: {benchmark[j]}')
			print(f'record length = {len(df)}')
			
		self.combined_df.drop('range', axis = 1, inplace = True)
		self.combined_df['random'] = 1
		self.combined_df['random'] = self.normalize_result(self.combined_df['random'], 1)
		if incorp_pub_step > 0:
			print('\nPublic estimation R2 score:\t', round(self.r2_score('public'), 4))
			print('Fundamental model R2 score:\t', round(self.r2_score('fundamental'), 4), '\n')
			score_dict = dict()
			print(f'\nAdjustment to z_score: {adjust_Z}\nPlotting R2 score with different degrees of incorporation based on public estimations...')
			for i in tqdm(np.arange(0, 1 + incorp_pub_step, incorp_pub_step), position = 0):
				self.combined_df['combined'] = np.exp(np.log(self.combined_df['fundamental']) * (1 - i) + np.log(self.combined_df['public']) * i) # equ 1, bill benter (1994)
				self.combined_df['combined'] = self.normalize_result(self.combined_df['combined'], 1)
				score_dict[i] = [self.r2_score('public'), self.r2_score('fundamental'), self.r2_score('combined')]
			score_df = pd.DataFrame().from_dict(score_dict, orient = 'index')
			score_df.columns = ['public', 'fundamental', 'combined']
			score_df.index.name = 'Pub weight'
			sns.lineplot(data = score_df, hue = score_df.index)
			plt.title(f'R2 score with different degree of public estimations incorporation\nAdj to z_score = {adjust_Z}')
			plt.xlabel('Weight of public estimate in combined model')
			plt.ylabel('pseudo-R2')
			plt.show()
			max_key = max(score_dict, key = score_dict.get)
			print(score_df)
			return round(max_key, 1), score_dict[max_key][2], score_dict[max_key][0]
		else:
			print(f'\nPublic estimation weight: {pub_weight}\nAdjustments to z_score = {adjust_Z}')
			self.combined_df['combined'] = np.exp(np.log(self.combined_df['fundamental']) * (1 - pub_weight) + np.log(self.combined_df['public']) * pub_weight) # equ 1, bill benter (1994)
			self.combined_df['combined'] = self.normalize_result(self.combined_df['combined'], 1)
			pub = self.r2_score('public')
			fun = self.r2_score('fundamental')
			com = self.r2_score('combined')
			print('Public estimation R2 score:\t', round(pub, 4))
			print('Fundamental model R2 score:\t', round(fun, 4))
			print('Combined model R2 score:\t', round(com, 4))
			return 0, com, pub

		models = ['fundamental', 'public']
		for factor in models:
			key = factor 
			df = self.concat_bin(key, self.combined_df, compare_model_bias = False)[0]
			data_dict = dict()
			for i in df.index.unique():
				temp_df = df[[key, 'actual']][df.index == i]
				exp_std = temp_df[key].std()
				exp_mean = temp_df[key].mean()
				act_mean = temp_df['actual'].mean()
				Z = (act_mean - exp_mean) / exp_std
				data_dict[i] = [len(temp_df), round(exp_mean, dp), round(act_mean, dp), round(Z, dp)]
			print(f'\n{key} model vs actual frequency:\n',
			pd.DataFrame.from_dict(data_dict, orient = 'index', columns = ['N', 'expected', 'actual', 'Z']).sort_index())

	
	def execute(self, er_threshold = 1, diff = 0.01, base = 0.1):
		
		self.combined_df['er'] = self.win_odds['Win Odds'] * self.combined_df['combined']
		self.combined_df['return'] = self.win_odds['return'] * self.combined_df['actual'] - 1
		
# 		Establish bet er_threshold and return on bets by model
# 		bet_index = self.combined_df[self.combined_df['er'] > er_threshold].index
		self.combined_df['diff'] = self.combined_df['combined'] - self.combined_df['public']
		bet_index = self.combined_df[(self.combined_df['diff'] > diff) & (self.combined_df['er'] > er_threshold) & (self.combined_df['public'] > base)].index
		self.combined_df['Bet'] = 0
		self.combined_df['Bet'].loc[bet_index] = 1
		self.combined_df['Model bet'] = self.combined_df['Bet'] * self.combined_df['return']
		self.combined_df['Model bet'] = self.combined_df['Model bet'].cumsum()
		
# 		To set a benchmark for profitr estimation, use the one with highest odds by public standard
		public_guess = []
		for race in self.combined_df['race'].unique():
			public_guess.append(self.combined_df[self.combined_df['race'] == race]['public'].idxmax())
		self.combined_df['Base public bet'] = 0
		self.combined_df['Base public bet'].loc[public_guess] = 1
		self.combined_df['Base public bet'] = self.combined_df['Base public bet'] * self.combined_df['return']
		self.combined_df['Base public bet'] = self.combined_df['Base public bet'].cumsum()
		
# 		store all win sitaution, reference its estimate to see how model performs in terms of ER
		self.win = self.combined_df[self.combined_df['actual'] == 1]
		
		# Result for the model
		print(f'\n---> Bet {len(bet_index)} out of {len(self.combined_df)} records ({round(len(bet_index) / len(self.combined_df) * 100, 1)})%')
		print(f'---> Win pct: {len(self.combined_df.loc[bet_index][self.combined_df.loc[bet_index]["actual"] == 1])} out of {len(bet_index)} bets ({round(len(self.combined_df.loc[bet_index][self.combined_df.loc[bet_index]["actual"] == 1]) / len(bet_index) * 100, 1)})%')
		
# 		Garph plotting
		sns.lineplot(data = self.combined_df[['Model bet', 'Base public bet']])
		plt.title(f'Profit on bet over {int(self.combined_df["public"].sum())} races from {self.st_date} to {self.end_date}\nAdj. to z_score = {self.adjust_Z} || {self.pub_weight} pub. est. || er_threshold for ER = {er_threshold}')
		plt.xlabel('Nos. of bets')
		plt.ylabel('Profit on bets')
		plt.show()
		
		er_value = self.win[self.win['er'] < 7]['er']
		er_profit = er_value[er_value > 1]
		sns.distplot(er_value)
		plt.title(f'Expected return for winning horses\nER {round(er_profit.count() / er_value.count() * 100, 2)}% > 1')
		plt.xlabel('Expected return')
		plt.xticks(np.arange(-1, 7))
		plt.show()
		
		



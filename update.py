# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 08:47:53 2020

@author: raymond-cy.liu
"""

import datetime
import os
import time
import pickle
from bs4 import BeautifulSoup
import pandas as pd
import math
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import deque

import operational

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 3})


class Update:
	
	def __init__(self, df, feature_df, avg, scoring_avg, horse_race_result_path, weather_history_path, update_pref = False, auto_save = False, cloud_path = None):
		st_time = time.time()
		time_now = datetime.datetime.fromtimestamp(st_time).strftime("%d-%b-%Y (%H:%M)")		
		with open('Update run_log.txt', 'a') as error_info:
			error_info.write(f'\n\nRunning log recording upadting for df ---> {time_now}\n')

		# Probably do not need deepcopy for all
		self.df = df
		self.feature_df = feature_df
		self.avg = avg
		self.scoring_avg = scoring_avg
		self.horse_path = horse_race_result_path
		self.weather_path = weather_history_path
		self.cloud_path = cloud_path
		self.auto_save = auto_save
		self.update_pref = update_pref
		
# 		See if the preference dict is to be renewed
		self.pref_update_no = len(self.feature_df)
# 		To override update pref decision
		if self.update_pref == False:
			if 'final_df.pickle' not in os.listdir():
				self.update_pref = True
		
		self.file_no = self.df.iloc[-1]['Race_no'] + 1
		self.files = os.listdir(self.horse_path)
		self._update_files = self.check_latest_dates(self.files)
		
# 		To fill in the tg. for those HKJC records missing Rtg in racecard
		self.class_score = {1:98, 2:87, 3:70, 4:51, 5:31, 0.8:110, 0.85:107, 0.9:103, 0.95:100, 1.5:93}
		
		update = 0
		
		while len(self._update_files) != 0:
			
			update += 1
			temp_time = time.time()
			
			slicing_num = 1000
						
			if len(self._update_files) > slicing_num:
				self.update_files = self._update_files[:slicing_num]
				self.update_files = self._update_files[:self.check_all_races(slicing_num)]
				print(f'Updating till {self.update_files[-1][0:10]}')
			
			else:
				self.update_files = self._update_files[:]
				print(f'Updating till {self.update_files[-1][0:10]}')
			
			self.new_df, self.new_combined_df = self.update_df()
			self.new_feature_df, self.new_combined_feature_df = self.update_feature_df()
			
			if auto_save:
				operational.save_data(self.new_combined_df, 'df.pickle')
				operational.save_data(self.new_combined_feature_df, 'feature_df.pickle')
				
				if cloud_path != None:
					cloud_path = cloud_path
					operational.save_data(self.new_combined_df, os.path.join(cloud_path, 'df.pickle'))
					operational.save_data(self.new_combined_feature_df, os.path.join(cloud_path, 'feature_df.pickle'))

			with open('Update_progress.txt', 'w') as update_info:
				update_info.write(f'Updated till {self.update_files[-1][0:10]}')
				
# 			Loops ENDS for that batch of files, and save file if needed
			
			with open('Update run_log.txt', 'a') as error_info:
				null_row = self.new_combined_feature_df[self.new_combined_feature_df.isnull().any(axis=1)]
				if len(null_row) != 0:
					error_info.write(f'{null_row.index} have null values!')
					print(f'{null_row.index} have null values!')
			
			print(f'Used {round(time.time() - temp_time, 0)} s')
				
# 			renew the df, feature
			self.df = self.new_combined_df
			self.feature_df = self.new_combined_feature_df
			
# 			See if there is any new files loaded in folder
			self.file_no = self.df.iloc[-1]['Race_no'] + 1
			self.files = os.listdir(self.horse_path)
			self._update_files = self.check_latest_dates(self.files)
		
		if update > 0: 
 			# if have updates, run this final tasks before preference
			self.class_score = {0.8: 113, 0.85: 110, 0.9: 105, 0.95: 102, 1: 100, 1.5: 87, 2: 80, 3: 60, 4: 40, 5: 20}
			self.horse_scoring()
			self.jockey_scoring()
			self.trainer_scoring()
			self.new_horse()
			self.new_gear(x = 3)
# 			self.cat_priority() # Replaced by one-hot encoding due to unscalaility
			self.combine_avg() # Auto save relevant file
			
		self.factor_dict = {
			'distance' : ['class', 'distance'] + [f'{avg}_distance' for avg in self.scoring_avg],
			'track' : ['track'],
			'condition' : ['condition'],
			'location' : ['location'],
			}
# 		To drop Horse back if go do multiple regression for all horses tgt
		self.others = ['Win Odds', 'Place Odds', 'Plc.', 'Finish Time', 'per_km_time', 'TBW', 'Plc. score', 'race_diff','temp', 'baro', 'wind', 'wd', 'hum', 'Year', 'Month', 'Day', 'Jockey', 'Trainer', 'N', 'public_est_diff', 'run_pos_max', 'run_pos_min', 'run_pos_mean', 'run_pos_range', 'run_pos_std', 'run_pos_min_slope', 'run_pos_max_slope', 'run_pos_initial', 'Gear', 'Best Time', 'Over Wt.'] #<<- Really drop priority??? # NOT dropping Race_no

		if self.update_pref:
			self.make_preference()
			self.pref_update_no = len(self.feature_df) - 100764 # 0.1: 100764, 0.05: 100764
			self.update_preference()
		
		print('\nNo new files for updates -------------------------\n')
		
		

	def check_latest_dates(self, files):
		all_dates = []
		latest_files = []
		latest_df_date = self.make_datetime(self.df.iloc[-1]['Year'], self.df.iloc[-1]['Month'], self.df.iloc[-1]['Day'])
		for file in files:
			year, month, day = list(map(int, file.split('_')[0:3]))
			all_dates.append(self.make_datetime(year, month, day))
		all_dates = list(set(all_dates))
		new_date = [date for date in all_dates if date > latest_df_date]
		new_date = list(map(self.return_date_string, new_date))
		latest_files = [file for file in files if file[0:10] in new_date]
		return latest_files
		
	
	def check_all_races(self, num):
# 		Check if there are more races
		for i in range(14): # assuming max 14 races per day
			if self.update_files[-1][:10] == self._update_files[num][:10]:
				self.update_files.append(self._update_files[num])
				num += 1
			else:
				return num
			
			
	def make_datetime(self, year, month, day):
		return datetime.date(year, month, day)
	
	
	def return_date_string(self, date):
		return date.strftime('%Y_%m_%d')
		
	
	def return_day(self, time_delta):
		return time_delta if type(time_delta) == int else time_delta.days


	def return_horse_age(self, time_delta):
		time_delta = math.floor(time_delta) if type(time_delta) == float else math.floor(time_delta.days/365)
		time_delta += 3
		return time_delta

	
	def assign_index(self, df, st, finish):
# 		have to pass len of df, or the intended num + 1
		df.index = np.arange(st, finish)
		return df
	
	
	def return_run_pos(self, pos, N):
		pos = np.array(pos)
		pos = 1 / (N - 1) * (pos - 1) * (10 - 1) + 1
		temp = dict()
		temp['run_pos_max'] =  pos.max()
		temp['run_pos_min'] = pos.min()
		temp['run_pos_mean'] = pos.mean()
		temp['run_pos_range'] = temp['run_pos_max'] - temp['run_pos_min']
		temp['run_pos_std'] = pos.std()
		temp['run_pos_min_slope'] = min(np.diff(pos).min(), 0) # To account for the rapid rise and drop of horse
		temp['run_pos_max_slope'] = max(np.diff(pos).max(), 0)
		temp['run_pos_initial'] = pos[:-1].mean() # To access the horse is laid back during the the initial period of the race
		return temp


	def make_new_df(self, update_files, file_no):
		
		_new_df = pd.DataFrame()
		with open('Update run_log.txt', 'a') as error_info:
		
			for file in self.update_files:
				print(file)
				year, month, day, location, _, _ = file.split('_')
				
				with open(os.path.join(self.weather_path, rf'{year}_{month}_{location}.pickle'), 'rb') as weather_info:
					weather = pickle.load(weather_info)
			
# 				try:
				with open(os.path.join(self.horse_path, file), 'r', encoding = 'utf-8') as source:
					html = source.read()
					soup = BeautifulSoup(html,'html.parser')
			# 		Race basic info scrapping
					basic_info = soup.find('tbody', attrs = {'class':'f_fs13'})
					_lv, _, condition = [i.text.strip() for i in basic_info.find_all('tr')[1].find_all('td')] # condition of track
					try:
						lv, distance, _ =  _lv.split(' - ')
						lv = int(lv.split()[1]) # Class 1 to 5
					except Exception as e:
						distance = _lv.split(' - ')[1]
						class_dict = {'Hong Kong Group One': 0.85, 'Hong Kong Group Two': 0.9, 'Hong Kong Group Three': 0.95, '4 Year Olds': 1.5, 'Group One': 0.8, 'Group Two': 0.9, 'Group Three': 0.95, 'Griffin Race': 4, 'Restricted Race': 2}
						lv = class_dict[_lv.split(' - ')[0]] # For international and unclassified events
					distance = int(distance.split('M')[0])/1000 # in kilometer
					track = [i.text.strip() for i in basic_info.find_all('tr')[2].find_all('td')][-1] # track type
					location = soup.find('td', attrs = {'class':'font_w7', 'style':'white-space:nowrap;'}).text.split(':')[0]
		
			# 		Main table scrapping
					table = soup.find('table', attrs = {'class' : 'f_tac table_bd draggable'})
					table_rows = table.find_all('tr')
					table_row = {}
					for i, tr in enumerate(table_rows):
						td = tr.find_all('td')
						row = [i.text.strip() for i in td]
						table_row[i] = row
						
					# To split and make the running positiion
					for i in range(1, len(table_row)):
						if table_row[i][-3] != '---':
							table_row[i][-3] = table_row[i][-3].split()
							table_row[i][-3] = list(map(int, table_row[i][-3]))
					

					new_df = pd.DataFrame(list(table_row[i] for i in range(1, len(table_row))), columns = table_row[0])
					new_df = new_df[new_df['Finish Time'] != '---'] # In case horse doesnt finish race
					sec = pd.to_datetime(new_df['Finish Time'], format='%M:%S.%f')
					new_df['Finish Time'] = sec.dt.minute * 60 + sec.dt.second + sec.dt.microsecond * 1e-6
					
			# 		Win/place odd scrapping and replace
					odds = soup.find('table', attrs = {'class' : 'table_bd f_tac f_fs13 f_fl'})
					# To prevent error due to the ',' sign when odds higher than 99
							# To account for the DH situation, and win odd will remanis same while the prize will be halved, therefore denominator is halved to reflect
					if '1 DH' in new_df['Plc.'].value_counts().index:
						new_df.iloc[0, -1] = float([i.text.strip() for i in odds.find_all('tr')[2].find_all('td')][-1].replace(',', '')) / 5
					else:
						new_df.iloc[0, -1] = float([i.text.strip() for i in odds.find_all('tr')[2].find_all('td')][-1].replace(',', '')) / 10
			# 		Likewise for place off and add
					new_df['Place Odds'] = 0
					for j in range(3, 6):
						# To prevent error due to the ',' sign when odds higher than 99
						new_df.iloc[j-3, -1] = float([i.text.strip() for i in odds.find_all('tr')[j].find_all('td')][-1].replace(',', '')) / 10
					
			# 		incoporate weather data
					infos = ['temp', 'baro', 'wind', 'wd', 'hum']
					for info in infos:
						new_df[info] = weather[info][int(day)]
					
					new_df.insert(0, 'Day', int(day))
					new_df.insert(0, 'Month', int(month))
					new_df.insert(0, 'Year', int(year))
					new_df.insert(0, 'Race_no', int(file_no))
					new_df['location'] = location
					new_df['class'] = lv
					new_df['distance'] = distance
					new_df['condition'] = condition
					new_df['track'] = track
					
					# In case the any of the stuff isnt included in HKJC website
					to_drop = ['LBW', 'Horse No.']
					for item in to_drop:
						try:
							new_df.drop(item, axis = 1, inplace = True)
						except Exception as e:
							print(f'{file} has no {item} info')
							error_info.write(f'{file} has no {item} info')
					
					new_df['Win Odds'] = new_df['Win Odds'].astype(float)
					
# 						----------------
# 						Trackwork scrapping 
					df_ = pd.DataFrame()
					tables = soup.findAll('table', attrs = {'class' : 'table_bd f_fs13'})
					for table in tables: # Possibily have backup for horse, therefore 2 tables
						if table.find('td').text.strip() == 'STAND-BY STARTER': # For backup table
							table_rows = table.find_all('tr')[1:]
						else:
							table_rows = table.find_all('tr')
						table_row = {}
						for i, tr in enumerate(table_rows):
							td = tr.find_all('td')
							row = [i.text.strip() for i in td]
							row[1] = row[1].split('\n')[0]
							table_row[i] = row
# 							Turn the paragraph into list
						for i in range(1, len(table_row)):
							for j in range(2, 7):
								table_row[i][j] = table_row[i][j].replace('  ', '').split('\n')
						
						temp = pd.DataFrame(list(table_row[i] for i in range(1, len(table_row))), columns = [i.replace('/', '') for i in table_row[0]])
						df_ = temp if df_.empty else df_.append(temp)
# 						Combine the two dfs together
					new_df = new_df.set_index(new_df['Horse'].apply(lambda name: name.split('(')[0])).join(df_.set_index('Name of Horse').iloc[:, 1:6]).reset_index(drop = True)
# 						----------------
						
# 						----------------
# 						Race card scrapping
					df_ = pd.DataFrame()					
# 						To parse main table
					tables = soup.find_all(lambda tag: tag.name == 'div' and tag.get('class') == ['rowDiv10'])[1] # Complicated since the script has been merged with noise
					table_row = {}
					table_row[0] = [i.text.strip() for i in tables.find('tr', attrs = {'class' : 'trBg01 boldFont13'}).find_all('td')]
					for i, tr in enumerate(tables.find('table', attrs = {'class' : 'draggable hiddenable'}).find('tbody').findAll('tr'), 1):
						table_row[i] = [i.text.strip() for i in tr.find_all('td')]
					temp = pd.DataFrame(list(table_row[i] for i in range(1, len(table_row))), columns = table_row[0])
# 						create a column for the jockey rtg change
					temp['Jockey Rtg.+/-'] = temp['Jockey'].apply(lambda name: name.split('(')[1].split(')')[0] if len(name.split('(')) > 1 else 0)
					df_ = temp if df_.empty else df_.append(temp)
					
# 						To parse the secondary stand-by table
					if len(soup.find_all(lambda tag: tag.name == 'div' and tag.get('class') == ['rowDiv10'])) > 2:
						tables = soup.find_all(lambda tag: tag.name == 'div' and tag.get('class') == ['rowDiv10'])[2]
						table_rows = tables.find_all('tr')[1:]
						table_row = {}
						for i, tr in enumerate(table_rows):
							td = tr.find_all('td')
							row = [i.text.strip() for i in td]
							table_row[i] = row
						temp = pd.DataFrame(list(table_row[i] for i in range(1, len(table_row))), columns = table_row[0])
						df_ = temp if df_.empty else df_.append(temp)
					df_['Rtg.'].fillna('-', inplace = True)
					df_ = df_[['Horse', 'Age', 'Gear', 'Wt.+/- (vs Declaration)', 'Priority', 'Rtg.', 'Rtg.+/-', 'Season Stakes', 'Best Time', 'Over Wt.', 'Jockey Rtg.+/-']].fillna(0)
					df_ = df_.replace('', 0)
					new_df = new_df.set_index(new_df['Horse'].apply(lambda name: name.split('(')[0])).join(df_.set_index('Horse')).reset_index(drop = True)
# 						----------------
				
				new_df[['Age', 'Wt.+/- (vs Declaration)', 'Rtg.+/-',  'Jockey Rtg.+/-']] = new_df[['Age', 'Wt.+/- (vs Declaration)', 'Rtg.+/-', 'Jockey Rtg.+/-']].replace('-', 0).astype(int)
				new_df['Rtg.'] = new_df['Rtg.'].replace('-', self.class_score[lv]).astype(int)
				new_df['Season Stakes'] = new_df['Season Stakes'].astype(float)
# 					Assume the first race, at least, do not have omission for infos
				try:
					new_df[['Actual Wt.', 'Declar. Horse Wt.', 'Draw']] = new_df[['Actual Wt.', 'Declar. Horse Wt.', 'Draw']].astype(int)
				except Exception as e:
					new_df[['Actual Wt.', 'Declar. Horse Wt.', 'Draw']] = new_df[['Actual Wt.', 'Declar. Horse Wt.', 'Draw']].replace('---', _new_df[['Actual Wt.', 'Declar. Horse Wt.', 'Draw']].mean()).astype(int)
			
					print(f'{file} has missing info on either Actual Wt., Declar. Horse Wt. or Draw')
					error_info.write(f'{file} has no missing info on either Actual Wt., Declar. Horse Wt. or Draw')
				
				if new_df.empty:
					_new_df = new_df
				else:
					_new_df = _new_df.append(new_df)
				file_no += 1
	
# =============================================================================
# 				except Exception as e:
# 					print(f'Record from {file} not captured')
# 					error_info.write(f'Record from {file} not captured')
# =============================================================================


		return _new_df
			
	
	def new_horse(self):
		print('\nUpdating new horse')
		new_horse_index = self.feature_df[self.feature_df['Last race day'] == 0].index
		self.feature_df['new'] = 0
		self.feature_df['new'].loc[new_horse_index] = 1

	
	
	def new_gear(self, x):	
		print('\nUpdating new gear')
		temp_ = pd.Series
		for horse in tqdm(self.feature_df['Horse'].unique()):
# 			Turn into cat and count
			temp = self.feature_df[self.feature_df['Horse'] == horse]['Gear'].astype('category').cat.codes
			temp = temp != temp.rolling(x+1).mean()
			temp.iloc[:x] = True
			temp_ = temp if temp_.empty else temp_.append(temp)
		self.feature_df['New_Gear'] = temp_.astype(int).sort_index()		
	
	
	def cat_priority(self):
		print('\nUpdating trainner priority on horses')
		self.feature_df[self.feature_df['Priority'] == 0]['Priority'] = '0'
		self.feature_df['Priority_cat'] = self.feature_df['Priority'].astype('category').cat.codes
	
	
	def combine_avg(self):
		
		temp_cols = ['Plc. score', 'TBW', 'Actual Wt.', 'Draw', 'per_km_time', 'race_diff', 'distance', 'public_est_diff', 'run_pos_max', 'run_pos_min', 'run_pos_mean', 'run_pos_range', 'run_pos_std', 'run_pos_min_slope', 'run_pos_max_slope', 'run_pos_initial', 'horse_avg_score']
		
		cols = list()
		[cols.append([f'{num}_' + s for s in temp_cols]) for num in self.scoring_avg]
		
		cols.append([
		 '5_jockey_avg_score', 
		 '15_jockey_avg_score', 
		 '40_jockey_avg_score', 
		 '10_trainer_avg_score', 
		 '30_trainer_avg_score', 
		 '100_trainer_avg_score', 
		   ])
	
		cols = [val for sublist in cols for val in sublist]
		
		cols_ = pd.Series(['_'.join(col.split('_')[1:]) for col in cols])
		print('\nFinally combining average scores')
		for item in cols_.unique():
			index = cols_[cols_== item].index
			col = pd.Series(cols).loc[index].to_list()
			self.feature_df[f'wavg_{item}'] = self.feature_df[col].sum(axis = 1) / len(col)
			
		operational.save_data(self.feature_df, 'feature_df.pickle')
	

	def horse_scoring(self):
		# Create score for every horse
		print('\nUpdating a continuous score for each horse')
		new_score_dict = dict()
		df = deepcopy(self.feature_df)
		temp_ = pd.Series()
		score_df = pd.Series()
		for horse in tqdm(df['Horse'].unique(), position = 0):
			first = df[df['Horse'] == horse].index
# 			Use the first score to fill all, and add with cumsum for ultimate score at every step
			temp = self.class_score[df['class'].loc[first[0]]] + df['Actual Wt.'].loc[first[0]] - 113
			temp_ = pd.Series(temp, index = first) if temp_.empty else temp_.append(pd.Series(temp, index = first))
		temp_ = pd.concat([temp_.sort_index(), df['Plc. score']], axis = 1)
		print('\nAdding Plc. scores and scoring averages for horses')
		for horse in tqdm(df['Horse'].unique(), position = 0):
			index = df[df['Horse'] == horse].index
			temp = temp_.loc[index]
# 			Leave the latest score increment into later use
			new_score_dict[horse] = temp['Plc. score'].iloc[-1]
			temp['horse_score'] = temp[0] + temp['Plc. score'].shift(1).fillna(0).cumsum()
# 			Count the horses rise from its very first apperance
			temp['horse_improve'] = temp['horse_score'] / temp.reset_index().index - temp['horse_score'].iloc[0] / temp.reset_index().index # Solely depend on overall slope?
			for num in self.scoring_avg:
				temp[f'{num}_horse_avg_score'] = temp['Plc. score'].rolling(window = num).mean().shift(1)
			temp.fillna(0, inplace = True) # Replace all the shift, and inf/inf with 0
			score_df = temp[['horse_score', 'horse_improve', '3_horse_avg_score', '8_horse_avg_score']] if score_df.empty else score_df.append(temp[['horse_score', 'horse_improve', '3_horse_avg_score','8_horse_avg_score']])
		self.feature_df[['horse_score', 'horse_improve', '3_horse_avg_score', '8_horse_avg_score']] = score_df.sort_index()
		operational.save_data(new_score_dict, 'horse_score_dict.pickle')
		
		
	def jockey_scoring(self):
		# Create score for every jockey
		print('\nUpdating a continuous score for each jockey')
		new_score_dict = dict()
		df = deepcopy(self.feature_df)
		temp_ = pd.Series()
		score_df = pd.Series()
		temp_ = pd.Series(0, index = df.index)
		temp_ = pd.concat([temp_.sort_index(), df['Plc. score']], axis = 1)
		print('\nAdding Plc. scores and scoring averages for jockeys')
		for jockey in tqdm(df['Jockey'].unique(), position = 0):
			index = df[df['Jockey'] == jockey].index
			temp = temp_.loc[index]
# 			Leave the latest score increment into later use
			new_score_dict[jockey] = temp['Plc. score'].iloc[-1]
			temp['jockey_score'] = temp[0] + temp['Plc. score'].shift(1).fillna(0).cumsum()
# 			Count the Jockey rise from its very first apperance
			temp['jockey_improve'] = temp['jockey_score'] / temp.reset_index().index - temp['jockey_score'].iloc[0] / temp.reset_index().index # Solely depend on overall slope?
			temp['5_jockey_avg_score'] = temp['Plc. score'].rolling(window = 5).mean().shift(1)
			temp['15_jockey_avg_score'] = temp['Plc. score'].rolling(window = 15).mean().shift(1)
			temp['40_jockey_avg_score'] = temp['Plc. score'].rolling(window = 40).mean().shift(1)
			temp.fillna(0, inplace = True) # Replace all the shift, and inf/inf with 0
			score_df = temp[['jockey_score', 'jockey_improve', '5_jockey_avg_score', '15_jockey_avg_score', '40_jockey_avg_score']] if score_df.empty else score_df.append(temp[['jockey_score', 'jockey_improve', '5_jockey_avg_score', '15_jockey_avg_score', '40_jockey_avg_score']])
		self.feature_df[['jockey_score', 'jockey_improve', '5_jockey_avg_score', '15_jockey_avg_score', '40_jockey_avg_score']] = score_df.sort_index()
		operational.save_data(new_score_dict, 'jockey_score_dict.pickle')
		
		
	def trainer_scoring(self):
		# Create score for every jockey
		print('\nUpdating a continuous score for each trainer')
		new_score_dict = dict()
		df = deepcopy(self.feature_df)
		temp_ = pd.Series()
		score_df = pd.Series()
		temp_ = pd.Series(0, index = df.index)
		temp_ = pd.concat([temp_, df['Plc. score']], axis = 1)
		print('\nAdding Plc. scores and scoring averages for trainers')
		for trainer in tqdm(df['Trainer'].unique(), position = 0):
			index = df[df['Trainer'] == trainer].index
			temp = temp_.loc[index]
# 			Leave the latest score increment into later use
			new_score_dict[trainer] = temp['Plc. score'].iloc[-1]
			temp['trainer_score'] = temp[0] + temp['Plc. score'].shift(1).fillna(0).cumsum()
# 			Count the Trainer rise from its very first apperance
			temp['trainer_improve'] = temp['trainer_score'] / temp.reset_index().index - temp['trainer_score'].iloc[0] / temp.reset_index().index # Solely depend on overall slope?
			temp['10_trainer_avg_score'] = temp['Plc. score'].rolling(window = 10).mean().shift(1)
			temp['30_trainer_avg_score'] = temp['Plc. score'].rolling(window = 30).mean().shift(1)
			temp['100_trainer_avg_score'] = temp['Plc. score'].rolling(window = 100).mean().shift(1)
			temp.fillna(0, inplace = True) # Replace all the shift, and inf/inf with 0
			score_df = temp[['trainer_score', 'trainer_improve', '10_trainer_avg_score', '30_trainer_avg_score', '100_trainer_avg_score']] if score_df.empty else score_df.append(temp[['trainer_score', 'trainer_improve', '10_trainer_avg_score', '30_trainer_avg_score', '100_trainer_avg_score']])
		self.feature_df[['trainer_score', 'trainer_improve', '10_trainer_avg_score', '30_trainer_avg_score', '100_trainer_avg_score']] = score_df.sort_index()
		operational.save_data(new_score_dict, 'trainer_score_dict.pickle')


	def trackwork_score(self, jockey, bt, gallop, trotting, swimming):
		agg_score = 0
		trackwork_dict = dict()
		
		interval = 1
# =============================================================================
# 		try:
# 			next_month = int(trotting[0].split('/')[0]) - int(trotting[-1].split('/')[0])
# 			interval = next_month + 1 if next_month > 0 else next_month + 1 + 30.5
# 		except:
# 			interval = 14
# 		interval = max(14, interval)
# =============================================================================
		
		score = 0
		for i in bt:
			num = 0
			try:
				plc, n = i.split('(')[0].split()[-1].split('/')
				num += 3 * (2 - int(plc) / int(n))
				a, b, c = i.split('(')[-1].split(')')[0].split('.')
				time = int(a) * 60 + int(b) + int(c) / 100
				distance = int(i.split()[3]) / 1000
				per_km_time = time / distance / np.power(1.075, distance - 1)
				num *= (np.power(1000, 58 / per_km_time) / 1000)
			except:
				pass
			if jockey in i:
				num *= 1.5
			score += num
		agg_score += score / interval
		trackwork_dict['BT'] = score / interval
		
		score = 0
		for i in gallop:
			time = []
			num = 0
			for word in i.split():
				try:
					float(word)
					num += 1.5
					time.append(word)
				except:
					pass
			try:
				time = np.array(time).astype(np.float)
				if (time.max() - time.min()) > 5.5 and time.min() < 24.5: # smaller than 24 & range > 5 is intense exercise
					num *= 1.25
				if jockey in i:
					num *= 1.5
			except:
				pass
			score += num
		agg_score += score / interval
		trackwork_dict['gallop'] = score / interval
					
		score = 0
		for i in trotting:
			try:
				num = int(i.split(' Round')[0][-1])
			except:
				num = 1
			if jockey in i:
				num *= 1.5
			score += num
		agg_score += score / interval
		trackwork_dict['trotting'] = score / interval
		
		score = (len(swimming)  - 1) / 2
		agg_score += score / interval
		trackwork_dict['swimming'] = score / interval
		
		trackwork_dict['trackwork_total'] = agg_score
		
		return trackwork_dict


	def make_new_feature_df(self):
		
# 		Call the previously made new df from new files
		new_feature_df = deepcopy(self.new_df)
		new_combined_feature_df = deepcopy(self.new_combined_df)
		updated_from = len(new_combined_feature_df) - len(new_feature_df)
		new_feature_df = self.assign_index(new_feature_df, updated_from, len(new_combined_feature_df))

		# Call previous record and use for estimate pass data
		previous_df = operational.load_data('feature_df.pickle')

		# Finishing time per 1000m
		print('\t Updating finishing time per 1000m')
		new_feature_df['per_km_time'] = new_feature_df['Finish Time'] / new_feature_df['distance'] / np.power(1.075, new_feature_df['distance'] - 1)
		
	# Convert the DH into actual place value, and calculate score for each plc, calculate time behind winner for races and No of participants in that race
		print('\t Updating TBW, nos. of participants, plcs and plc score')
		new_feature_df.replace('DISQ', '14', inplace = True)
		for plc in new_feature_df['Plc.'].unique():
			if len(plc) > 2:
				rank = int(plc.split(' ')[0])
				new_feature_df.replace(plc, rank - 0.5 if rank > 1 else 1, inplace = True)
		_temp_df = pd.DataFrame()
		for i in new_feature_df['Race_no'].unique():
			temp_df = new_feature_df[new_feature_df['Race_no'] == i][['per_km_time', 'Plc.']]
			temp_df.replace(temp_df['per_km_time'].values, temp_df['per_km_time'].iloc[0], inplace = True)
			temp_df.replace(temp_df['Plc.'].values, len(temp_df), inplace = True)
			_temp_df = temp_df if i == 1 else _temp_df.append(temp_df)
		new_feature_df['TBW'] = new_feature_df['per_km_time'] - _temp_df['per_km_time']
		new_feature_df['N'] = _temp_df['Plc.']
		new_feature_df['Plc.'] = new_feature_df['Plc.'].astype(float)
		new_feature_df['Plc. score'] = 1 + (new_feature_df['Plc.'] - 1) * 14 / new_feature_df['N'] # normalize to 14 horse race
		new_feature_df['Plc. score'] = np.exp(4 - new_feature_df['Plc. score'])
	
		# Add difference between public expectations and real result last time
		print('\t Updating publie error estimation deviation on horse')
		temp_ = pd.Series()
		for race in new_feature_df['Race_no'].unique():
			temp = new_feature_df[new_feature_df['Race_no'] == race][['Plc.', 'Win Odds']]
			temp = temp.sort_values('Win Odds').reset_index(drop = True).reset_index()
			temp['index'] += 1
	# 		To Adjust the magnitude of estimate by adding the std and diff to mean
			adj_fac = temp['Win Odds'].std() * (temp['Win Odds'] / temp['Win Odds'].mean() - 1)
			temp['public_est_diff'] = (1 / temp['Plc.'] - 1 / temp['index']) * adj_fac.abs()
			temp['public_est_diff'] = temp['public_est_diff'] / temp['public_est_diff'].abs().mean()
			temp['public_est_diff'].fillna(0, inplace = True)
			temp.sort_values('Plc.', inplace = True)
			temp_ = temp['public_est_diff'] if temp_.empty else temp_.append(temp['public_est_diff'])
		temp_.index = new_feature_df.index
		new_feature_df['public_est_diff'] = temp_
	
		# Add running position analysis for horses
		print('\t Updating Running Position on horse')
		run_pos = pd.DataFrame(map(self.return_run_pos, new_feature_df['RunningPosition'], new_feature_df['N']))
		run_pos.index = new_feature_df.index
		new_feature_df = new_feature_df.join(run_pos).drop('RunningPosition', axis = 1)
		
# 		Race_diff placed here NECESSARY!
	# 	How difficult is the race in terms of competition
		new_feature_df['race_diff'] = new_feature_df['per_km_time']
		for i in new_feature_df['Race_no'].unique():
			temp_df = new_feature_df[new_feature_df['Race_no'] == i]
			index = temp_df.index
			race_diff = temp_df['per_km_time'] * temp_df['Plc. score'] / temp_df['Plc. score'].sum()
			race_diff = race_diff.sum()
			new_feature_df['race_diff'].loc[index] = race_diff
			
		# Specify the score received per horse in each class race, to replace the Plc. score
		print('\t Replacing Plc. score')
		temp_ = pd.Series()
		for race in tqdm(new_feature_df['Race_no'].unique(), position = 0):
			temp = new_feature_df[new_feature_df['Race_no'] == race][['class', 'Plc.', 'TBW', 'N', 'Draw', 'Actual Wt.', 'public_est_diff']] # Public est to be included
			temp['score'] = 0
			temp['score'].iloc[0] = 6 + min((1 + temp['TBW'].diff().abs().iloc[1]), 1.4) ** 6
			temp['score'].iloc[1] = 14 - temp['score'].iloc[0]
			temp['score'].iloc[2] = temp['score'].iloc[1] - min((1 + temp['TBW'].diff().abs().iloc[2]), 1.3) ** 2.5
			N = temp['N'].mean()
			cl = temp['class'].mean()
			if N > 5:
				temp['score'].iloc[3] = temp['score'].iloc[2] / (1+temp['TBW'].diff().abs().iloc[3]) ** 10
				temp['score'].iloc[6:] = - temp['TBW'].iloc[6:] / temp['TBW'].iloc[6:].sum() * 14 * (N-6) / 4
			else:
				temp['score'].iloc[3:] = -temp['TBW'].iloc[3:]/ temp['TBW'].iloc[3:].sum() * 4 * (N-3) / 5
			temp['TBW'].loc[temp[temp['TBW'] > 3].index] = 3
			temp['score'].loc[temp.index[:3]] = temp['score'].iloc[:3] * np.power(1 + (4 - temp['Plc.'].iloc[:3]) / 30, 5-cl)
			temp['score'] = temp['score'] + temp['Draw']/temp['N'] - temp['Plc.']/temp['N'] + 20 * temp['Actual Wt.'] / temp['Actual Wt.'].mean() - 20 + np.power(1.1, temp['public_est_diff']) - 1
			temp_ = temp['score'] if temp_.empty else temp_.append(temp['score'])
		new_feature_df['Plc. score'] = temp_
		
		# Adding trackwork data
		print('\t Updating trackwork data of horse')
		new_feature_df[['BT', 'gallop', 'swimming', 'trackwork_total', 'trotting']] = pd.DataFrame(map(self.trackwork_score, new_feature_df['Jockey'], new_feature_df['Barrier Trial'], new_feature_df['Gallop'], new_feature_df['Trotting'], new_feature_df['Swimming']), index = new_feature_df.index) # Ignore Spelling for now
		new_feature_df.drop(['Barrier Trial', 'Gallop', 'Swimming', 'Trotting', 'Spelling'], axis = 1, inplace = True)

# 		Does not need rearranging since it adds on the original
		new_combined_feature_df = self.feature_df.append(new_feature_df, sort = True)
		# Only add weight diff and time since last race ON HORSE
		print('\t Updating weight diff and time since last race on horse')
		dates = pd.DataFrame(map(self.make_datetime, new_combined_feature_df['Year'], new_combined_feature_df['Month'], new_combined_feature_df['Day']))
		new_combined_feature_df['Last race day'] = dates
		new_combined_feature_df['Weight diff.'] = new_combined_feature_df['Declar. Horse Wt.']
		features_to_add = ['Weight diff.', 'Last race day']
		for feature in features_to_add:
			data_dict = dict()
			for horse in new_feature_df['Horse'].unique():
				temp_df = new_combined_feature_df[new_combined_feature_df['Horse'] == horse][feature] - new_combined_feature_df[new_combined_feature_df['Horse'] == horse][feature].shift(1)
				temp_df = temp_df[temp_df.index >= updated_from]
				temp_df.fillna(0 if temp_df.dtype == float else pd.Timedelta(0), inplace = True) # Why becomes float?
				data_dict.update(temp_df.to_dict())
			new_feature_df[feature] = pd.DataFrame.from_dict(data_dict, orient = 'index').sort_index()
# 		Reassign index num to the new _df to be appended
		new_feature_df['Last race day'] = self.assign_index(pd.DataFrame(map(self.return_day, new_feature_df['Last race day'])), updated_from, len(new_combined_feature_df))
		
		# Past place record on same track
		print('\t Updating past place record on same track')
		features_to_add = ['Last plc.']
		new_combined_feature_df['Last plc.'] = new_combined_feature_df['Plc.']
		locations = ['Happy Valley', 'Sha Tin']
		data_dict = dict()
		for feature in features_to_add:
			for location in locations:
				for horse in new_feature_df['Horse'].unique():
					temp_df = new_combined_feature_df[(new_combined_feature_df['Horse'] == horse) & (new_combined_feature_df['location'] == location)][feature].shift(1)
					temp_df = temp_df[temp_df.index >= updated_from]
					temp_df.fillna(14, inplace = True) # Assigned 14 as it is newbie, need review
					data_dict.update(temp_df.to_dict())
			new_feature_df[feature] = pd.DataFrame.from_dict(data_dict, orient = 'index').sort_index()

# =============================================================================
# 	 # Frozen for now, use the age from racecard
# 		# Calculate horse age (first race + 3)
# 		print('\t Updating horse age')
# 		new_combined_feature_df['Age'] = dates
# 		features_to_add = ['Age']
# 		data_dict = dict()
# 		for feature in features_to_add:
# 			for horse in new_feature_df['Horse'].unique():
# 				temp_df = new_combined_feature_df[new_combined_feature_df['Horse'] == horse][[feature]]
# 				temp_df['st_date'] = new_combined_feature_df[new_combined_feature_df['Horse'] == horse][feature].iloc[0]
# 				temp_df[feature] = temp_df[feature] - temp_df['st_date']
# 				data_dict.update(temp_df[feature].to_dict())
# 			new_feature_df[feature] = pd.DataFrame.from_dict(data_dict, orient = 'index').sort_index()
# 		new_feature_df['Age'] = self.assign_index(pd.DataFrame(map(self.return_horse_age, new_feature_df['Age'])), updated_from, len(new_combined_feature_df))
# =============================================================================
			
		# Add recent race data for the horses only
		# Load data from previous year for the first cols selections
		print('\n\t Updating average across recent races for horses')
		previous_dict = dict()
		for cl in previous_df['class'].unique():
			for loc in previous_df['location'].unique():
				previous_dict[f'class{cl}_{loc}'] = previous_df[(previous_df['class'] == cl) & (previous_df['location'] == loc)][['Plc. score', 'TBW', 'Actual Wt.', 'Draw', 'per_km_time', 'race_diff', 'distance', 'public_est_diff', 'run_pos_max', 'run_pos_min', 'run_pos_mean', 'run_pos_range', 'run_pos_std', 'run_pos_min_slope', 'run_pos_max_slope', 'run_pos_initial']].mean()
		for avg in tqdm(self.scoring_avg, position = 0):
			cols = ['Plc. score', 'TBW', 'Actual Wt.', 'Draw', 'per_km_time', 'race_diff', 'distance', 'public_est_diff'] + ['run_pos_max', 'run_pos_min', 'run_pos_mean', 'run_pos_range', 'run_pos_std', 'run_pos_min_slope', 'run_pos_max_slope', 'run_pos_initial']
			_x = pd.DataFrame()
			for horse in new_feature_df['Horse'].unique():
				num = deque(maxlen = avg)
				index = new_combined_feature_df[new_combined_feature_df['Horse'] == horse].index
				if len(index) > 1:
					if index[0] >= updated_from:
					# 	To fill up the first value first
						cl = new_combined_feature_df['class'].loc[index[0]]
						loc = new_combined_feature_df['location'].loc[index[0]]
						x = previous_dict[f'class{cl}_{loc}']
						x = x.to_frame().T
						x.index = np.arange(index[0], index[0] + 1)
						_x = x if len(_x) == 0 else _x.append(x)
					for i, value in enumerate(index[:-1]): # Make sure wont count current result
						num.append(value)
						x = new_combined_feature_df[new_combined_feature_df['Horse'] == horse][cols].loc[num].mean()
						x = x.to_frame().T
						if index[i+1] >= updated_from:
							x.index = np.arange(index[i+1], index[i+1] + 1)
							_x = x if len(_x) == 0 else _x.append(x)
				else:
					cl = new_combined_feature_df['class'].loc[index].values[0]
					loc = new_combined_feature_df['location'].loc[index].values[0]
					x = previous_dict[f'class{cl}_{loc}']
					x = x.to_frame().T
					x.index = np.arange(index[0], index[0] + 1)
					_x = x if len(_x) == 0 else _x.append(x)
			avg_cols = [f'{avg}_{col}' for col in cols]
			for col in range(len(cols)):
				_x.rename(columns = {cols[col]:avg_cols[col]}, inplace = True)
			_x.sort_index(inplace = True)
	# 		To estimate if its current distance is longer than its usual race or not
			_x[f'{avg}_distance'] = _x[f'{avg}_distance'] - new_feature_df['distance']
			new_feature_df = new_feature_df.join(_x)
		
	# 	Caluclate winning pct for two parties
	# 	Have to use so many for loops?!?!
		print('\n\t Updating win/plc percentage across recent races for jockeys and trainers')
		items = ['Jockey', 'Trainer']
		for item in items:
			data_dict = dict()
			for j in tqdm(self.avg, position = 0):
				data_dict[j] = dict()
				for plc in range(1, 4):
					data_dict[j][plc] = dict()
				for name in new_feature_df[item].unique():
					row_list = deque(maxlen = j)
					index = new_combined_feature_df[new_combined_feature_df[item] == name].index
					if len(index) > 1:
						if index[0] >= updated_from:
							for plc in range(1, 4):
								data_dict[j][plc][index[0]] = 0
							for i, row in enumerate(index[:-1]): # Make sure wont count current result
								row_list.append(row)
								x = new_feature_df[new_feature_df[item] == name]['Plc.'].loc[row_list].value_counts()
								for plc in range(1, 4):
									if plc in x.index:
										first = x.loc[plc]
										data_dict[j][plc][index[i+1]] = round(first / x.sum(), 4)
									else:
										data_dict[j][plc][index[i+1]] = 0

						else:
							for i, row in enumerate(index[:-1]): # Make sure wont count current result
								row_list.append(row)
								x = new_combined_feature_df[new_combined_feature_df[item] == name]['Plc.'].loc[row_list].value_counts()
								if index[i+1] >= updated_from:
									for plc in range(1, 4):
										if plc in x.index:
											first = x.loc[plc]
											data_dict[j][plc][index[i+1]] = round(first / x.sum(), 4)
										else:
											data_dict[j][plc][index[i+1]] = 0
					else:
						for plc in range(1, 4):
							data_dict[j][plc][index[0]] = 0
				for plc in range(1, 4):
					temp_df = pd.DataFrame.from_dict(data_dict[j][plc], orient = 'index').sort_index()
					# Try changing it to see how it affects performance
					new_feature_df[f'{item}_{plc}_plc_over_{j}_races'] = temp_df # -np.log(temp_df)?
		
		return new_feature_df


# 	Main function starts
	def update_df(self):
		
		print('\nUpdating df -------------')
		
		new_df = self.make_new_df(self.update_files, self.file_no)

		new_combined_df = self.df.append(new_df, sort = False)
		print('Updated df\n')
		return new_df, new_combined_df.reset_index(drop = True)

		
	def update_feature_df(self):
		
		print('Updating feature_df -------------')		
		time_now = datetime.datetime.fromtimestamp(time.time()).strftime("%d-%b-%Y (%H:%M)")
		with open('Update run_log.txt', 'a') as error_info:
			error_info.write(f'\n\nRunning log recording updating for feature_df ---> {time_now}\n')
		
		new_feature_df = self.make_new_feature_df()
		
		new_combined_feature_df = self.feature_df.append(new_feature_df, sort = False)
		
		print('Updated feature_df\n')
		return new_feature_df, new_combined_feature_df.reset_index(drop = True)

		
	def get_encoding(self, *factors_columns):
	# Categorical encoding or one hot?
	# One hot first
		columns = [
				'Priority',
				'location', 
				'condition', 
				'track',
				]
		
		df = self.temp_df.copy()
		
# 		Assemble columns to be dropped
		drop_cols = []
		for factor in factors_columns:
			drop_cols = drop_cols + factor
			
		for col in columns:
			if col not in drop_cols:
				encode_df = pd.get_dummies(self.temp_df[col], prefix = col)
				df = pd.concat([df, encode_df], axis = 1).drop(col, axis = 1)
						
		return df.drop(drop_cols, axis = 1)

# =============================================================================
# 	def build_NNmodel(self, dim):
# 		model = keras.Sequential()
# 		model.add(layers.Dense(64, activation = 'relu', input_dim = dim))
# 		model.add(layers.Dropout(0.5))
# 		model.add(layers.Dense(128, activation = 'relu'))
# 		model.add(layers.Dropout(0.5))
# 		model.add(layers.Dense(64, activation = 'relu'))
# 		model.add(layers.Dense(1, activation = 'linear'))
# 		
# 
# 		optimizer = tf.keras.optimizers.RMSprop(0.001)
# 
# 		model.compile(loss = 'mse',
# 				optimizer = 'adam',
# 				metrics = ['mae', 'mse'])
# 		
# 		return model
# =============================================================================

	def split_train(self, encoded_df):
		
		df = encoded_df.copy()
				
		temp = pd.Series()
		
		index = df.index
		x = df.drop(['Norm_plc.', 'Race_no', 'Horse'], axis = 1)
		y = df['Norm_plc.']
		x = (x - x.mean()) / x.std()
		
# =============================================================================
# 		reg_net = make_pipeline(PolynomialFeatures(1), LinearRegression())
# 		reg_net.fit(x, y)
# 		y_net = reg_net.predict(x)
# =============================================================================
		
		reg_ridge = make_pipeline(PolynomialFeatures(1), Ridge())
		reg_ridge.fit(x, y)
		y_ridge = reg_ridge.predict(x)

# =============================================================================
# 		from sklearn.metrics import r2_score
# 		import seaborn as sns
# 		df = pd.concat([pd.DataFrame(y_net), pd.DataFrame(y_ridge), y], axis = 1)
# 		df.columns = ['Net_y', 'Ridge_y', 'Real_y']
# 		sns.lineplot(data = df.iloc[6000:6040])
# 		plt.show()
# 		print(f'Linear reg. fit score: {r2_score(y.to_numpy(), y_net)}')
# 		print(f'Ridge reg. fit score: {r2_score(y.to_numpy(), y_ridge)}')
# =============================================================================
		
		diff =  - y + y_ridge # Negative residual implies the factor is benefitial, vice versa
		
# =============================================================================
# 		NN_model = self.build_NNmodel(x.shape[1])
# 		history = NN_model.fit(x.to_numpy(), y.to_numpy(), epochs = 25, validation_split = 0.2, verbose = 0)
# 		y_NN = NN_model.predict(x.to_numpy())
# =============================================================================
		
		temp = pd.Series(diff, index = index) if temp.empty else temp.append(pd.Series(diff, index = index))
	
		return temp.reset_index(drop = True)
	
	
	def fit_preference(self, factor, degree):
			df = pd.Series()
			temp_dict = dict()
			horses = self.temp_df['Horse'].unique()
			print(f'\nMaking {factor} preference for horses...')
			for horse in tqdm(horses, position = 0):
				x, y = [], []
				index = self.temp_df[self.temp_df['Horse'] == horse].index
				for row in index:
					x.append(self.temp_df[self.temp_df['Horse'] == horse][factor].loc[row])
					y.append(self.temp_df[self.temp_df['Horse'] == horse][f'{factor}_pref'].loc[row])
				
				x = np.array(x)
				y = np.array(y)
				y /= max((10/len(y)), 1) # To adjust the significance of data, if less than 10 then reduce its significance
				
				if degree != None:
					if len(np.unique(x)) <= 1:
						y = np.zeros(len(x))
						temp = pd.Series(y)
						temp.index = index
						df = temp if df.empty else df.append(temp)
					else:
# =============================================================================
# 						x = np.array(x)
# 						fit = np.poly1d(np.polyfit(x, y, degree))
# =============================================================================
						
						x = x.reshape(-1, 1)
						
# =============================================================================
# # For debugging purpose, plot with ployfit and sklearn polyfit
# model = make_pipeline(PolynomialFeatures(degree), Ridge())
# model.fit(x, y)
# plt.scatter(x, y, color = 'k', s = 3)
# plt.scatter(x, model.predict(x)/y.std(), color = 'blue', s = 10)
# plt.title('sklearn polyfit')
# plt.show()
# 
# fit = np.poly1d(np.polyfit(x.reshape(1, -1)[0], y, degree))
# plt.scatter(x.reshape(1, -1)[0], y, color = 'k', s = 3)
# plt.scatter(x.reshape(1, -1)[0], fit(x.reshape(1, -1)[0])/y.std(), color = 'red', s = 10)
# plt.title('numpy polyfit')
# plt.show()				
# =============================================================================
						
						model = make_pipeline(PolynomialFeatures(degree),
# 											preprocessing.StandardScaler(),
											Ridge())
						temp_dict[horse] = [model.fit(x, y), y.std()] # Put into dict
# 						print(temp_dict[horse] if y.std() == 0 else print(''))
						temp = pd.Series(model.predict(x) / y.std()) # Benter (1994) divide the predicted result by the std
						temp.index = index
						df = temp if df.empty else df.append(temp)
				else:
					if len(np.unique(x)) <= 1:
						y = np.zeros(len(x))
						temp = pd.Series(y)
						temp.index = index
						df = temp if df.empty else df.append(temp)
					else:
						temp = pd.Series(y)
						temp.index = x
						temp_dict[horse] = dict()
						for row in temp.index.unique():
							if type(temp.loc[row]) == np.float64:
								temp_dict[horse][row] = temp.loc[row].mean() / y.std() / temp.count() # Put into dict
							else:
								temp_dict[horse][row] = temp.loc[row].mean() / y.std() * temp.loc[row].count() / temp.count() # Put into dict
							temp.loc[row] = temp_dict[horse][row]
						temp.index = index
						df = temp if df.empty else df.append(temp)
			if degree == None:
				z = pd.DataFrame.from_dict(temp_dict, orient = 'columns').fillna(0)
			else:
				z = temp_dict
			return df.sort_index(), z
				
	
# 	Only to be used with large dataset, the preference based on end of 2019, NOT time dependent
	def make_preference(self):
		
		self.temp_df_ = deepcopy(self.feature_df)
		self.temp_df = self.temp_df_.iloc[:100764]
				
		A = self.temp_df['N'].astype(float) - 1
		A = A.pow(-1)
		B = self.temp_df['Plc.'].astype(float) - 1
		self.temp_df['Norm_plc.'] = 1 + (10-1) * A * B
		
		self.pref_dict = dict()
		self.final_df = self.feature_df.copy().iloc[:100764]
		
		for factor in self.factor_dict.keys():
			self.temp_df[f'{factor}_pref'] = self.split_train(self.get_encoding(self.factor_dict[factor], self.others))
			
			self.final_df[f'{factor}_pref'], self.pref_dict[factor] = self.fit_preference(factor, degree = 1 if factor == 'distance' else None)
			
			if self.auto_save:
				operational.save_data(self.final_df, 'final_df.pickle')
				operational.save_data(self.pref_dict, 'pref_dict.pickle')
					
				if self.cloud_path != None:
					operational.save_data(self.final_df, os.path.join(self.cloud_path, 'final_df.pickle'))
					operational.save_data(self.pref_dict, os.path.join(self.cloud_path, 'pref_dict.pickle'))

			self.temp_df.drop(f'{factor}_pref', axis = 1, inplace = True)
				
			
	def update_preference(self):
		
		if self.pref_update_no == 0:
			return
		
		# Use pref_dict to temporarily update the final_df, recommend to update the whole database of preference regularly for maximal accuracy
		
		pref_dict = operational.load_data('pref_dict.pickle')
		
		new_final_df = self.feature_df.iloc[-self.pref_update_no:]
		
		horses = self.feature_df['Horse'].iloc[-self.pref_update_no:].unique()
		
		for factor in self.factor_dict.keys():
			print(f'\nUpdating {factor}_pref....')
			df = pd.DataFrame()
			for horse in tqdm(horses, position = 0):
				index = new_final_df[new_final_df['Horse'] == horse].index
				if horse in pref_dict[factor].keys():
					if factor == 'distance':
# =============================================================================
# 						fit = pref_dict[factor][horse]
# 						x = new_final_df[factor].loc[index].to_numpy()
# =============================================================================
						model = pref_dict[factor][horse][0]
						x = new_final_df[factor].loc[index].to_numpy().reshape(-1, 1)
						temp = pd.Series(model.predict(x)/pref_dict[factor][horse][1])# working with series is WAY FASTER than working with dataframes
						temp.index = index
						df = temp if df.empty else df.append(temp)
					else:
						mean = pref_dict[factor][horse]
						x = new_final_df[factor].loc[index]
						temp = x.map(mean)
						df = temp if df.empty else df.append(temp)
				else:
					temp = pd.Series(np.zeros(len(index)), index = index)
					df = temp if df.empty else df.append(temp)
					
			new_final_df[f'{factor}_pref'] = df.sort_index()
		final_df = operational.load_data('final_df.pickle')
		self.obs_final_df = final_df.append(new_final_df)
							
# 		Not a lot of horses, loop through all then update and save
		if self.auto_save:
			print('Used pref_dict to temporarily update the final_df, recommend to udpate the whole database of preference regularly for maximal accuracy')
			operational.save_data(self.obs_final_df, 'final_df_(no_new_pref).pickle')
				
			if self.cloud_path != None:
				operational.save_data(self.obs_final_df, os.path.join(self.cloud_path, 'final_df_(no_new_pref).pickle'))
				
	
				

		
# Need to add a function to fit new manual input data into actual real betting
				
				
				

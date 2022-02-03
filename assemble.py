# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:23:00 2020

@author: raymond-cy.liu
"""

import pandas as pd
import os
from bs4 import BeautifulSoup
import pickle
import datetime
import numpy as np
import math
import time
from tqdm import tqdm
from collections import deque
import operational


def assemble_df(horse_race_result_path, weather_history_path, year = 'all'):
	print('\nAssembling data -------------')
	
	files = os.listdir(horse_race_result_path)
	if year == 'all':
		result_files = files
	else:
		result_files = [file for file in files if file[:4] == str(year)]
		
	combined_df = pd.DataFrame()
	file_no = 1
	time_now = datetime.datetime.fromtimestamp(time.time()).strftime("%d-%b-%Y (%H:%M)")
	
# 		To fill in the tg. for those HKJC records missing Rtg in racecard
	class_score = {1:98, 2:87, 3:70, 4:51, 5:31, 0.8:110, 0.85:107, 0.9:103, 0.95:100, 1.5:93}

	with open('Assemble run_log.txt', 'a') as error_info:
		error_info.write(f'\n\nRunning log recording assembling for df ---> {time_now}\n')
		for file in result_files:
			year, month, day, location, _, _ = file.split('_')
			
			with open(os.path.join(weather_history_path, rf'{year}_{month}_{location}.pickle'), 'rb') as weather_info:
				weather = pickle.load(weather_info)
# 				try:
				with open(os.path.join(horse_race_result_path, file), 'r', encoding = 'utf-8') as source:
					html = source.read()
					soup = BeautifulSoup(html, 'html.parser')

# 						----------------
# 						Trackwork scrapping 
			# 		Race basic info scrapping
					basic_info = soup.find('tbody', attrs = {'class':'f_fs13'})
					print('\t', file)
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
						
# 						To split and make the running positiion
					for i in range(1, len(table_row)):
						if table_row[i][-3] != '---':
							table_row[i][-3] = table_row[i][-3].split()
							table_row[i][-3] = list(map(int, table_row[i][-3]))
					
					df = pd.DataFrame(list(table_row[i] for i in range(1, len(table_row))), columns = table_row[0])
					df = df[df['Finish Time'] != '---'] # In case horse doesnt finish race
					sec = pd.to_datetime(df['Finish Time'], format='%M:%S.%f')
					df['Finish Time'] = sec.dt.minute * 60 + sec.dt.second + sec.dt.microsecond * 1e-6
					
			# 		Win/place odd scrapping and replace
	# 				See if the win/odds table is available for scrapping
					try:
						odds = soup.find('table', attrs = {'class' : 'table_bd f_tac f_fs13 f_fl'})
						# To prevent error due to the ',' sign when odds higher than 99
							# To account for the DH situation, and win odd will remanis same while the prize will be halved, therefore denominator is halved to reflect
						if '1 DH' in df['Plc.'].value_counts().index:
							df.iloc[0, -1] = float([i.text.strip() for i in odds.find_all('tr')[2].find_all('td')][-1].replace(',', '')) / 5
						else:
							df.iloc[0, -1] = float([i.text.strip() for i in odds.find_all('tr')[2].find_all('td')][-1].replace(',', '')) / 10
					
				# 		Likewise for place off and add
						df['Place Odds'] = 0
						for j in range(3, 6):
							# To prevent error due to the ',' sign when odds higher than 99
							df.iloc[j-3, -1] = float([i.text.strip() for i in odds.find_all('tr')[j].find_all('td')][-1].replace(',', '')) / 10
					except Exception as e:
						print(f'{file} has no win/odds info')
						error_info.write(f'{file} has no win/odds info\n')
					
	# 				incoporate weather data
					infos = ['temp', 'baro', 'wind', 'wd', 'hum']
					for info in infos:
						df[info] = weather[info][int(day)]
					
					df.insert(0, 'Day', int(day))
					df.insert(0, 'Month', int(month))
					df.insert(0, 'Year', int(year))
					df.insert(0, 'Race_no', int(file_no))
					df['location'] = location
					df['class'] = lv
					df['distance'] = distance
					df['condition'] = condition
					df['track'] = track
					
# 					In case the any of the stuff isnt included in HKJC website
					to_drop = ['LBW', 'Horse No.']
					for item in to_drop:
						try:
							df.drop(item, axis = 1, inplace = True)
						except:
							print(f'{file} has no {item} info')
							error_info.write(f'{file} has no {item} info\n')
					
					df['Win Odds'] = df['Win Odds'].astype(float)
# 						----------------

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
					df = df.set_index(df['Horse'].apply(lambda name: name.split('(')[0])).join(df_.set_index('Name of Horse').iloc[:, 1:6]).reset_index(drop = True)
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
					df_ = df_[['Horse', 'Age', 'Gear', 'Wt.+/- (vs Declaration)', 'Priority', 'Rtg.', 'Rtg.+/-', 'Season Stakes', 'Best Time', 'Over Wt.', 'Jockey Rtg.+/-', 'Import Cat.']].fillna(0)
					df_ = df_.replace('', 0)
					df = df.set_index(df['Horse'].apply(lambda name: name.split('(')[0])).join(df_.set_index('Horse')).reset_index(drop = True) # To join racecard into main namelist in case switched
# 						----------------
						
# 						----------------
# 						Other reports??


						
# 						----------------
					
				df[['Age', 'Wt.+/- (vs Declaration)', 'Rtg.+/-', 'Jockey Rtg.+/-']] = df[['Age', 'Wt.+/- (vs Declaration)', 'Rtg.+/-',  'Jockey Rtg.+/-']].replace('-', 0).astype(int)
				df['Rtg.'] = df['Rtg.'].replace('-', class_score[lv]).astype(int)
				df['Season Stakes'] = df['Season Stakes'].astype(float)
# 					Assume the first race, at least, do not have omission for infos
				try:
					df[['Actual Wt.', 'Declar. Horse Wt.', 'Draw']] = df[['Actual Wt.', 'Declar. Horse Wt.', 'Draw']].astype(int)
				except Exception as e:
					df[['Actual Wt.', 'Declar. Horse Wt.', 'Draw']] = df[['Actual Wt.', 'Declar. Horse Wt.', 'Draw']].replace('---', combined_df[['Actual Wt.', 'Declar. Horse Wt.', 'Draw']].mean()).astype(int)
				
					print(f'{file} has missing info on either Actual Wt., Declar. Horse Wt. or Draw')
					error_info.write(f'{file} has no missing info on either Actual Wt., Declar. Horse Wt. or Draw')
				
				if combined_df.empty:
					combined_df = df
				else:
					combined_df = combined_df.append(df)
				file_no += 1

# =============================================================================
# 				except Exception as e:
# 					print(f'Record from {file} not captured')
# 					with open('Assemble run_log.txt', 'a') as error_info:
# 						error_info.write(f'\nRecord from {file} not captured')
# 				
# =============================================================================
					
	print('Data assembled-------------\n')
	return combined_df.reset_index(drop = True)


def get_features(df, win_pct_avg, scoring_avg):
	print('\nMaking features -------------')
	
	def make_datetime(year, month, day):
		return datetime.date(year, month, day)
	
	
	def return_day(time_delta):
		return time_delta if type(time_delta) == int else time_delta.days


	def return_horse_age(time_delta):
		time_delta = math.floor(time_delta) if type(time_delta) == int else math.floor(time_delta.days/365)
		time_delta += 3
		return time_delta
	
	
	def trackwork_score(jockey, bt, gallop, trotting, swimming):
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
		
	
	def return_run_pos(pos, N):
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
		
	
	# Call previous record and use for estimate pass data
	previous_df = operational.load_data('2009_feature_df.pickle')
		
	# Finishing time per 1000m
	print('\t Featuring finishing time per 1000m')
	df['per_km_time'] = df['Finish Time'] / df['distance'] / np.power(1.075, df['distance'] - 1)
				
	# Convert the DH into actual place value, and calculate score for each plc, calculate time behind winner for races and No of participants in that race
	print('\t Featuring TBW, nos. of participants, plcs and plc score')
	df.replace('DISQ', '14', inplace = True)
	for plc in df['Plc.'].unique():
		if len(plc) > 2:
			rank = int(plc.split(' ')[0])
			df.replace(plc, rank - 0.5 if rank > 1 else 1, inplace = True)
	_temp_df = pd.DataFrame()
	for i in df['Race_no'].unique():
		temp_df = df[df['Race_no'] == i][['per_km_time', 'Plc.']]
		temp_df.replace(temp_df['per_km_time'].values, temp_df['per_km_time'].iloc[0], inplace = True)
		temp_df.replace(temp_df['Plc.'].values, len(temp_df), inplace = True)
		_temp_df = temp_df if i == 1 else _temp_df.append(temp_df)
	df['TBW'] = df['per_km_time'] - _temp_df['per_km_time']
	df['N'] = _temp_df['Plc.']
	df['Plc.'] = df['Plc.'].astype(float)
	df['Plc. score'] = 1 + (df['Plc.'] - 1) * 14 / df['N'] # normalize to 14 horse race
	df['Plc. score'] = np.exp(4 - df['Plc. score'])
	
	# Add difference between public expectations and real result last time
	print('\t Featuring publie error estimation deviation on horse')
	temp_ = pd.Series()
	for race in df['Race_no'].unique():
		temp = df[df['Race_no'] == race][['Plc.', 'Win Odds']]
		temp = temp.sort_values('Win Odds').reset_index(drop = True).reset_index()
		temp['index'] += 1
# 		To Adjust the magnitude of estimate by adding the std and diff to mean
		adj_fac = temp['Win Odds'].std() * (temp['Win Odds'] / temp['Win Odds'].mean() - 1)
		temp['public_est_diff'] = (1 / temp['Plc.'] - 1 / temp['index']) * adj_fac.abs()
		temp['public_est_diff'] = temp['public_est_diff'] / temp['public_est_diff'].abs().mean()
# 		For races that the place follows entirely with public estimates
		temp['public_est_diff'].fillna(0, inplace = True)
		temp.sort_values('Plc.', inplace = True)
		temp_ = temp['public_est_diff'] if temp_.empty else temp_.append(temp['public_est_diff'])
	df['public_est_diff'] = temp_.reset_index(drop = True)
	
	# Add running position analysis for horses
	print('\t Featuring Running Position on horse')
	run_pos = pd.DataFrame(map(return_run_pos, df['RunningPosition'], df['N']))
	run_pos.index = df.index
	df = df.join(run_pos).drop('RunningPosition', axis = 1)
	
	# Only add weight diff and time since last race ON HORSE
	print('\t Featuring weight diff and time since last race on horse')
	df['Last race day'] = pd.DataFrame(map(make_datetime, df['Year'], df['Month'], df['Day']))
	df['Weight diff.'] = df['Declar. Horse Wt.']
	features_to_add = ['Weight diff.', 'Last race day']
	for feature in features_to_add:
		data_dict = dict()
		for horse in df['Horse'].unique():
			temp_df = df[df['Horse'] == horse][feature] - df[df['Horse'] == horse][feature].shift(1)
			temp_df.fillna(0 if temp_df.dtype == float else pd.Timedelta(0), inplace = True) # To be updated?
			data_dict.update(temp_df.to_dict())
		df[feature] = pd.DataFrame.from_dict(data_dict, orient = 'index').sort_index()
	df['Last race day'] = pd.DataFrame(map(return_day, df['Last race day']))
	
# =============================================================================
# 	# label whether the horse is a new horse
# 	print('\t Featuring new horse')
# 	index = df[df['Last race day'] == 0].index
# 	df['new'] = 0
# 	df['new'].loc[index] = 1
# =============================================================================
	
	# Past place record on same track
	print('\t Featuring past place record on same track')
	features_to_add = ['Last plc.']
	df['Last plc.'] = df['Plc.']
	locations = ['Happy Valley', 'Sha Tin']
	data_dict = dict()
	for feature in features_to_add:
		for location in locations:
			for horse in df['Horse'].unique():
				temp_df = df[(df['Horse'] == horse) & (df['location'] == location)][feature].shift(1)
				temp_df.fillna(14, inplace = True) # Assigned 14 as it is newbie, need review
				data_dict.update(temp_df.to_dict())
		df[feature] = pd.DataFrame.from_dict(data_dict, orient = 'index').sort_index()
		
# =============================================================================
# # 	# Frozen for now, use the age from racecard
# 	# Calculate horse age (first race + 3)
# 	print('\t Featuring horse age')
# 	df['Age'] = pd.DataFrame(map(make_datetime, df['Year'], df['Month'], df['Day']))
# 	features_to_add = ['Age']
# 	data_dict = dict()
# 	for feature in features_to_add:
# 		for horse in df['Horse'].unique():
# 			temp_df = df[df['Horse'] == horse][[feature]]
# 			temp_df['st_date'] = df[df['Horse'] == horse][feature].iloc[0]
# 			temp_df[feature] = temp_df[feature] - temp_df['st_date']
# 			data_dict.update(temp_df[feature].to_dict())
# 		df[feature] = pd.DataFrame.from_dict(data_dict, orient = 'index').sort_index()
# 	df['Age'] = pd.DataFrame(map(return_horse_age, df['Age']))
# =============================================================================
	
# 	How difficult is the race in terms of competition
	print('\t Featuring difficulty of race')
	df['race_diff'] = df['per_km_time']
	for i in df['Race_no'].unique():
		temp_df = df[df['Race_no'] == i]
		index = temp_df.index
		race_diff = temp_df['per_km_time'] * temp_df['Plc. score'] / temp_df['Plc. score'].sum()
		race_diff = race_diff.sum()
		df['race_diff'].loc[index] = race_diff
		
	# Specify the score received per horse in each class race, to replace the Plc. score
	print('\t Replacing Plc. score')
	temp_ = pd.Series()
	for race in tqdm(df['Race_no'].unique(), position = 0):
		temp = df[df['Race_no'] == race][['class', 'Plc.', 'TBW', 'N', 'Draw', 'Actual Wt.', 'public_est_diff']] # Public est to be included
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
# 		Adjust for the draw and weight carried as well as public estimation error
		temp['score'] = temp['score'] + temp['Draw']/temp['N'] - temp['Plc.']/temp['N'] + 20 * temp['Actual Wt.'] / temp['Actual Wt.'].mean() - 20 + np.power(1.1, temp['public_est_diff']) - 1
		temp_ = temp['score'] if temp_.empty else temp_.append(temp['score'])
	df['Plc. score'] = temp_
	
	# Adding trackwork data
	print('\t Updating trackwork data of horse')
	df[['BT', 'gallop', 'swimming', 'trackwork_total', 'trotting']] = pd.DataFrame(map(trackwork_score, df['Jockey'], df['Barrier Trial'], df['Gallop'], df['Trotting'], df['Swimming'])) # Ignore Spelling for now
	df.drop(['Barrier Trial', 'Gallop', 'Swimming', 'Trotting', 'Spelling'], axis = 1, inplace = True)
	
	# Add recent race data for the horses only
	# Load data from year 2009 for the first cols selections
	print('\n\t Featuring average across recent races for horses')
	previous_dict = dict()
	for cl in previous_df['class'].unique():
		for loc in previous_df['location'].unique():
			previous_dict[f'class{cl}_{loc}'] = previous_df[(previous_df['class'] == cl) & (previous_df['location'] == loc)][['Plc. score', 'TBW', 'Actual Wt.', 'Draw', 'per_km_time', 'race_diff', 'distance', 'public_est_diff', 'run_pos_max', 'run_pos_min', 'run_pos_mean', 'run_pos_range', 'run_pos_std', 'run_pos_min_slope', 'run_pos_max_slope', 'run_pos_initial']].mean()
	for avg in tqdm(scoring_avg):
		cols = ['Plc. score', 'TBW', 'Actual Wt.', 'Draw', 'per_km_time', 'race_diff', 'distance', 'public_est_diff'] + ['run_pos_max', 'run_pos_min', 'run_pos_mean', 'run_pos_range', 'run_pos_std', 'run_pos_min_slope', 'run_pos_max_slope', 'run_pos_initial']
		_x = pd.DataFrame()
		for horse in df['Horse'].unique():
			num = deque(maxlen = avg)
			index = df[df['Horse'] == horse].index
			if len(index) > 1:
# 				To fill up the first value first
				cl = df['class'].loc[index[0]]
				loc = df['location'].loc[index[0]]
				x = previous_dict[f'class{cl}_{loc}']
				x = x.to_frame().T
				x.index = np.arange(index[0], index[0] + 1)
				_x = x if len(_x) == 0 else _x.append(x)
				for i, value in enumerate(index[:-1]): # Make sure wont count current result
					num.append(value)
					x = df[df['Horse'] == horse][cols].loc[num].mean()
					x = x.to_frame().T
					x.index = np.arange(index[i+1], index[i+1] + 1)
					_x = x if len(_x) == 0 else _x.append(x)
			else:
				cl = df['class'].loc[index].values[0]
				loc = df['location'].loc[index].values[0]
				x = previous_dict[f'class{cl}_{loc}']
				x = x.to_frame().T
				x.index = np.arange(index[0], index[0] + 1)
				_x = x if len(_x) == 0 else _x.append(x)
		avg_cols = [f'{avg}_{col}' for col in cols]
		for col in range(len(cols)):
			_x.rename(columns = {cols[col]:avg_cols[col]}, inplace = True)
		_x.sort_index(inplace = True)
# 		To estimate if its  current distance is longer than its usual race or not
		_x[f'{avg}_distance'] = df['distance'] - _x[f'{avg}_distance']
		df = df.join(_x)
		
# 	Caluclate winning pct for two parties
# 	Have to use so many for loops?!?!
	print('\n\t Featuring win/plc percentage across recent races for jockeys and trainers')
	items = ['Jockey', 'Trainer']
	for item in items:
		data_dict = dict()
		for j in tqdm(win_pct_avg):
			data_dict[j] = dict()
			for plc in range(1, 4):
				data_dict[j][plc] = dict()
			for name in df[item].unique():
				row_list = deque(maxlen = j)
				index = df[df[item] == name].index
				if len(index) > 1:
					for plc in range(1, 4):
						data_dict[j][plc][index[0]] = 0
					for i, row in enumerate(index[:-1]): # Make sure wont count current result
						row_list.append(row)
						x = df[df[item] == name]['Plc.'].loc[row_list].value_counts()
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
		# 		Try changing it to see how it affects performance
				df[f'{item}_{plc}_plc_over_{j}_races'] = temp_df # -np.log(temp_df)?
		
	print('\nFeatures made -------------')
	return df


	
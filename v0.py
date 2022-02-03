# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import time
import pandas as pd
import os
import numpy as np
from web_scrapping import parse_hkjc_racesum, parse_weather, parse_remedial_hkjc, parse_hkjc_ind
from assemble import assemble_df, get_features
import operational, visualize
import combine_hkjc_record 
from update import Update
import Model_training

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',500)
pd.set_option('display.expand_frame_repr', True)
plt.rcParams.update({'font.size':3})
sns.set_context(rc={"lines.linewidth": 1})


# File creation, can be more elegant
# =============================================================================
horse_race_result_path = 'horse race result history'
weather_history_path = 'weather history'
error_path = 'Error'
chrome_path = r'C:\Users\HipHingUser\Dropbox\Horse betting\chromedriver.exe' # to be changed for diff computers
cloud_path = r'C:\Users\HipHingUser\Dropbox\Horse betting\result dfs'
try:
	os.mkdir(horse_race_result_path)
except Exception as e:
	pass
try:
	os.mkdir(weather_history_path)
except Exception as e:
	pass


# Hyperparameter definition
# =============================================================================
WIN_PCT_AVG = [
			20, 
			50,
			] # for Jockey and Trainer respectively

SCORING_AVG = [
 			3, 
 			8, 
			 ] # For Horse ONLY


st = time.time()
temp_time = st

# =============================================================================
# # Parse horse result and weather at hkjc sites
# years = range(2009, 2020)
# months = range(1, 13) 
# locations = ['HV', 'ST'] # Manual input for weather parsing only
# for year in years:
# 	for month in months:
# 		temp_time = time.time()
# # 		parse_hkjc_racesum(year, month, chrome_path) # At least parse a month
# # 		for location in locations:
# # 			parse_weather(year, month, location, weather_history_path)
# # 			pass
# 		print(f'---> {year}_{month} has used {round(time.time() - temp_time,0)}s to parse')
# =============================================================================

# =============================================================================
# # Try capturing other webpages or whats left
# # files = os.listdir(horse_race_result_path)
# parse_hkjc_ind(files, 'trackwork', chrome_path) # racecard, past_incident, exp_factors
# =============================================================================

# =============================================================================
# # Error remeidal for HKJC reuslts, put in Error file and will recapture source code
# parse_remedial_hkjc(error_path, chrome_path)
# =============================================================================

# =============================================================================
# # To combine the race webpage into one source for each race
# combine_hkjc_record.combine_hkjc_records(horse_race_result_path)
# =============================================================================

try:
	df = operational.load_data('df.pickle')
except:
	print('No data file, calculating original df\n')
	# Assemble all basic infos
	temp_time = time.time()
	df = assemble_df(horse_race_result_path, weather_history_path, year = 2009)
	operational.save_data(df, 'df.pickle')
	print(f'---> Used {round(time.time() - temp_time,0)}s to assemble df\n\n')
	
try:
	feature_df = operational.load_data('feature_df.pickle')
except:
	print('No feature file, calculating feature_df\n')
	# Feature engineering for the data
	temp_time = time.time()
	df = operational.load_data('df.pickle')
	feature_df = get_features(df, WIN_PCT_AVG, SCORING_AVG)
	operational.save_data(feature_df, 'feature_df.pickle')
	operational.check_column_type(feature_df)
	print(f'---> Used {round(time.time() - temp_time,0)}s to create new feature')


# Access the original df with  the above only, update class only for updates due to shallow copying
update = Update(df, feature_df, WIN_PCT_AVG, SCORING_AVG, horse_race_result_path, weather_history_path, 
				update_pref = False,
				auto_save = True, 
# 				cloud_path = cloud_path,
				)


# =============================================================================
# win_rate = visualize.win_rate_per_race(feature_df, corr_to = 1, dict_return = True)
# =============================================================================

drop_cols = [
 # 'score',
 'distance_pref',
 'track_pref',
 'condition_pref',
 'location_pref',
 # 'score',
 # 'improve',
 # 'Jockey_Elo',
 # 'Trainer_Elo',
	
 # '2_Plc. score','2_TBW','2_Actual Wt.','2_Draw','2_per_km_time','2_race_diff','2_distance','2_public_est_diff','2_run_pos_max','2_run_pos_min','2_run_pos_mean','2_run_pos_range','2_run_pos_std','2_run_pos_min_slope','2_run_pos_max_slope','2_run_pos_initial',
 # '5_Plc. score','5_TBW','5_Actual Wt.','5_Draw','5_per_km_time','5_race_diff','5_distance','5_public_est_diff','5_run_pos_max','5_run_pos_min','5_run_pos_mean','5_run_pos_range','5_run_pos_std','5_run_pos_min_slope','5_run_pos_max_slope','5_run_pos_initial',
 # '10_Plc. score','10_TBW','10_Actual Wt.','10_Draw','10_per_km_time','10_race_diff','10_distance','10_public_est_diff','10_run_pos_max','10_run_pos_min','10_run_pos_mean','10_run_pos_range','10_run_pos_std','10_run_pos_min_slope','10_run_pos_max_slope','10_run_pos_initial',
 # '15_Plc. score','15_TBW','15_Actual Wt.','15_Draw','15_per_km_time','15_race_diff','15_distance','15_public_est_diff','15_run_pos_max','15_run_pos_min','15_run_pos_mean','15_run_pos_range','15_run_pos_std','15_run_pos_min_slope','15_run_pos_max_slope','15_run_pos_initial',
 # '2_Jockey_Elo_avg','5_Jockey_Elo_avg','10_Jockey_Elo_avg','15_Jockey_Elo_avg',
 # '5_avg_score', '10_avg_score', # Ti add more
 
 # 'wavg_Plc. score','wavg_TBW','wavg_Actual Wt.','wavg_Draw','wavg_per_km_time','wavg_race_diff','wavg_distance','wavg_public_est_diff','wavg_run_pos_max','wavg_run_pos_min','wavg_run_pos_mean','wavg_run_pos_range','wavg_run_pos_std','wavg_run_pos_min_slope','wavg_run_pos_max_slope','wavg_run_pos_initial','wavg_Jockey_Elo_avg','wavg_avg_score',
		]

final_df = operational.load_data('final_df_(no_new_pref).pickle').drop(drop_cols, axis = 1)
# final_df = operational.load_data('elo_df.pickle').drop(drop_cols, axis = 1)
model = Model_training.Model(final_df)
model.train(test_split = 0.1)
pos, model_score, public_score = model.analysis_result(pub_weight = 0, incorp_pub_step = 0, adjust_Z = 0)
model.execute(er_threshold = 0.8, diff = 0.03, base = 0.00)


# =============================================================================
# ## For visiualizing the change of fit in the console, preference
# horse = 'FLYING GODSPELL(V376)'
# new_final_df = operational.load_data('final_df_(no_new_pref).pickle')
# final_df = operational.load_data(r'C:\Users\raymond-cy.liu\Dropbox\Horse betting\result dfs\final_df.pickle')
# pref_dict = operational.load_data('pref_dict.pickle')
# new_pref_dict = operational.load_data(r'C:\Users\raymond-cy.liu\Dropbox\Horse betting\result dfs\pref_dict.pickle')
# 
# new_horses = new_final_df['Horse'].iloc[100694:].unique()
# for horse in new_final_df['Horse'].iloc[100694:].value_counts()[new_final_df['Horse'].iloc[90000:].value_counts() > 12].index:
# 	index = new_final_df[new_final_df['Horse'] == horse].index
# 	if horse in new_horses:
# 		length = index > 100694
# 		if horse == horse: #'FLYING GODSPELL(V376)':
# 			print(index)
# 			new = new_final_df[new_final_df['Horse'] == horse]
# 			old = final_df[final_df['Horse'] == horse]
# # 			print(new.loc[index].iloc[:, :10])
# 			old = final_df[final_df['Horse'] == horse]
# 
# 			x = old['distance'].to_numpy()
# 			new_x = new['distance'].to_numpy()
# 			y = old['distance_pref'].to_numpy()
# 			new_y = new['distance_pref'].to_numpy()
# 			
# 			plt.scatter(x, pref_dict['distance'][horse](x), s = 2)
# 			plt.scatter(new_x, new_pref_dict['distance'][horse](new_x), s = 5, marker = 'x')
# 			plt.title(f'{horse} ({round(length.sum()/len(index), 2)})\n{length.sum()}/{len(index)} new')
# 			plt.show()
# 			time.sleep(1)
# =============================================================================

# =============================================================================
# ## To visualize the betting criteria 
# for i in range(8500, 8510):
#     print(i)
#     print(final_df[['Plc.', 'Horse', 'score', 'improve', '5_avg_score', '5_Jockey_Elo_avg', 'Jockey_Elo', 'Win Odds']][final_df['Race_no'] == i].reset_index(drop = True))
#     index = final_df[final_df['Race_no'] == i].index - 100764
#     print(model.combined_df.loc[index].reset_index(drop = True), '\n\n')
# =============================================================================


print(f'\nTotal running time: {round(time.time() - st, 0)}s')





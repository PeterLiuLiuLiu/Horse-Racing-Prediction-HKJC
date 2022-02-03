# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 00:27:34 2020

@author: raymond-cy.liu
"""

import pickle
import os

def save_data(df, name):
	with open(f'{name}', 'wb') as handle:
		pickle.dump(df, handle)
		print(f'Record saved ({name})')
		

def load_data(name, path = None):
	with open(name,'rb') as handle:
		df = pickle.load(handle)
		print(f'Record loaded ({name})')
	return df

		
def check_column_type(df):
	print('\nChecking type of values in each column')
	for x, y in zip(df.columns, df.values[0]):
		print(x, type(y))
	print('\n')


def check_all_data(path):
	files = os.listdir(path)
	datas = []
	years = []
	for file in files:
		year, month = file.split('_')[0:2]
		data = f'{year}_{month}' # current data in file
		if data not in datas:
			datas.append(data)
		if year not in years:
			years.append(year)
# 	Check all available years
	count = 0
	for year in years:
		for month in range(1, 13):
			month = f'0{month}' if month < 10 else str(month)
			record = f'{year}_{month}'
			if record not in datas:
				print(f'{record} record does not exist')
				count += 1
	if count == 0:
		print(f'All records from {years[0]} to {years[-1]} saved')
		print('\n')

	
	

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:59:00 2020

@author: raymond-cy.liu
"""

import matplotlib.pyplot as plt
import time
import numpy as np
plt.style.use('ggplot')

# feature_df.set_index('Horse').loc[x]


def win_rate(df, plot = True, dict_return = False):
	
	print(df.columns)
	item = str(input(f'Columns to calculate against win_pct: '))
	corr_to = float(input(f'{item} to be corrected to the nearest: '))
	column = np.round(df[item] / corr_to, 0) * corr_to
	win_dict = {i: 0 for i in sorted(column.unique())}
	for plc, value in zip(df['Plc.'], np.round(df[item] / corr_to, 0) * corr_to):
		if plc == 1:
			win_dict[value] += 1
	win_dict = {i: win_dict[i] / len(df) * 100 for i in sorted(column.unique())}
	bottom = 0
	top = max(win_dict.values()) * 1.2
	plt.xlabel(f'{item}')
	plt.ylabel('Win_pct (%)')
	plt.title(f'Win_pct plot against {item}')
	plt.ylim(bottom, top)
	plt.scatter(win_dict.keys(), win_dict.values(), s = 4)
	return win_dict

	
def plot_acc_win_odds(df):
	acc = {year:0 for year in df['Year'].unique()}
	for race in df['Race_no'].unique():
		temp_df = df[df['Race_no'] == race].set_index('Plc.').sort_index()
		year = temp_df['Year'].iloc[0]
		if temp_df['Win Odds'].iloc[0] == temp_df['Win Odds'].min():
			acc[year] += 1
	for year in acc.keys():
		acc[year] /= len(df[df['Year'] == year]['Race_no'].unique())
		acc[year] *= 100
	plt.xlabel('Year')
	plt.ylabel('Accuracy (%)')
	plt.title('Accuracy plot against Year')
	plt.bar(acc.keys(), acc.values())
	return acc


def plot_relationship(df):
	print(df.columns)
	x = input('x axis ?')
	y = input('y axis ?')
	for i in df[x].unique():
		plt.scatter(i, df[df[x] == i][y].mean())
	plt.show()	
		
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:42:23 2020

@author: raymond-cy.liu
"""

import os
from tqdm import tqdm

def difference_content(horse_race_result_path, item):
	original = r'horse race result history (race)'
	lost_files = list(set(os.listdir(original)) - set(os.listdir(f'{horse_race_result_path} ({item})')))
	return lost_files
	

def combine_hkjc_records(horse_race_result_path):
	combine_file = dict()
	for file in os.listdir():
		if horse_race_result_path in file:
			combine_file[file] = len(os.listdir(file))
	original = r'horse race result history (race)'
	items = []
	diff = []
	for i in list(combine_file.keys()):
		if i.split(horse_race_result_path)[1] != '':
			item = i.split(horse_race_result_path)[1].split('(')[1].split(')')[0]
			items.append(item)
			diff.append(len(difference_content(horse_race_result_path, item)))
			if sum(diff) != 0:
				print(f'Mismatch between files: (race) & {item}, please check')
# 				return
	if len(os.listdir(horse_race_result_path)) > 0:
		replace = input('Resultant path has files already, replace? press Y/y\n')
	if replace.upper() == 'Y':
		for file in tqdm(os.listdir(original)):
			with open(os.path.join(f'{horse_race_result_path}', file), 'wb') as fw:
				for item in items:
					with open(os.path.join(f'{horse_race_result_path} ({item})', file), 'rb') as fr:
						fw.write(fr.read())
	else:
		print('Files in resultant path kept untouched.')
			
		
	




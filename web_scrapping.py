# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 09:41:09 2020

@author: NG
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import pandas as pd
import os
import re
import datetime
import urllib.request as url
import numpy as np
import pickle
from tqdm import tqdm


# MAKE A CLASS OBJECT
def parse_hkjc_racesum(year, month, chrome_path):
	result_path = 'horse race result history (race)'
	year = str(year)
	month = str(month) if month >= 10 else f'0{month}'
	error_path = os.path.join(result_path, rf'~error_web_parsing.txt')

	# Finding the racing dates in certain month
	check_date_url = rf'https://racing.hkjc.com/racing/information/english/Racing/Fixture.aspx?CalYear={year}&CalMonth={month}'
	try:
		options = Options()
		options.add_argument('--no-sandbox')
		browser = webdriver.Chrome(executable_path = chrome_path, options = options)
		browser.get(check_date_url)
		time.sleep(1)
		html = browser.page_source
		soup = BeautifulSoup(html,'lxml')
		days = dict() 
		race_days = soup.find_all('p', attrs = {'class':'f_clear'})
		for i in range(len(race_days)):
			temp_day = race_days[i].find_all('span')[0].text.strip()
			temp_day = temp_day if len(temp_day) == 2 else f'0{temp_day}'
			temp_location = re.findall(r'<img alt="(\w+)', str(race_days[i].find_all('span')[1]))[0]
			days[temp_day] = temp_location
	except Exception as e:
		print(f'\n{year}_{month} cannot read race schedule on HKJC!')
		with open(error_path, 'a') as error_file: # Save error months for record
			error_file.write(f'\n{year}_{month} cannot read race schedule on HKJC!')
			browser.close()
			return

	browser.close()
	print('\n', year, month, days)

	
	for day in days:
		location = days[day]
		url = rf'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={year}/{month}/{day}&Racecourse={location}&RaceNo=1' 
        
# 		Try if it is a raceday and count how many races on that day and parse 1st race
		loop_race_num = True
		error_count = 0
		while loop_race_num == True :
			loop_race_num = False
			options = Options()
			options.add_argument('--no-sandbox')
			browser = webdriver.Chrome(executable_path = chrome_path, options = options)
			browser.get(url)
			time.sleep(1)
			html = browser.page_source
			soup = BeautifulSoup(html,'html.parser')
			race_num = soup.find('table', attrs = {'class': 'f_fs12 f_fr js_racecard'})

			# Accounting for overseas at the bottom messing up the race_num
			try:
				overseas_race = int(race_num.find_all('a')[-1].get('href').split('RaceNo=')[1])
			except Exception as e:
				overseas_race = 0

# 			Still got error for some reason, cannot obtain race_num
			try:
				race_num = int(race_num.find_all('a')[-2-overseas_race].get('href').split('RaceNo=')[1])
				print(f'{year}_{month}_{day}_{location} going to create {race_num} nos. of files')
			except Exception as e:
				loop_race_num = True
				error_count += 1
                
            # Return and check further if cannot parse race_num
			if error_count > 3:
				loop_race_num = False
				print(f'{year}_{month} has no info on HKJC website!')
				with open(error_path, 'a') as error_file: # Save error months for record
					error_file.write(f'\n{year}_{month} has no info on HKJC website!')
					browser.close()
				return

			browser.close()
	
		i = 1
		while i <= race_num:
			url = rf'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={year}/{month}/{day}&Racecourse={location}&RaceNo={i}'
			options = Options()
			options.add_argument('--no-sandbox')
			browser = webdriver.Chrome(executable_path = chrome_path, options = options)
			browser.get(url)
			time.sleep(1)
			html = browser.page_source
		
			# Add condition if there is NO INFORMATION
			
			num = str(i) if i >= 10 else f'0{i}'
			file_path = os.path.join(result_path, rf'{year}_{month}_{day}_{location}_race_{num}.txt')
			with open(file_path, 'w', encoding = "utf-8") as html_file:
				html_file.write(html)
				if os.path.getsize(file_path) > 50000: # Check if source is captured correcvtly
					i += 1
			browser.close()
		print(f'{year}_{month}_{day}_{location}, {race_num} files have been created')


def parse_hkjc_ind(names, item, chrome_path):
	result_path = rf'horse race result history ({item})'
	try:
		os.makedirs(result_path)
	except Exception as e:
		pass
	for name in tqdm(names, position = 0):
		year, month, day, location,_, i = name.split('_')
		i = i[:2]
		x = {'racecard':'RaceCard',
				'past_incident':'RaceReportExt',
				'exp_factors':'ExceptionalFactors',
				'vet': 'VeterinaryRecord'
				}
		if item != 'trackwork':
			url = rf'https://racing.hkjc.com/racing/info/meeting/{x[item]}/English/Local/{year}{month}{day}/{location}/{i}'
		else:
			url = rf'https://racing.hkjc.com/racing/information/English/Racing/Localtrackwork.aspx?RaceDate={year}/{month}/{day}&Racecourse={location}&RaceNo={i}'
			
		
		# Add condition if there is NO INFORMATION
		num = 0
		while num < 1:
			options = Options()
			options.add_argument('--no-sandbox')
			browser = webdriver.Chrome(executable_path = chrome_path, options = options)
			browser.get(url)
# 			time.sleep(1)
			html = browser.page_source
		
			file_path = os.path.join(result_path, rf'{year}_{month}_{day}_{location}_race_{i}.txt')
			with open(file_path, 'w', encoding = "utf-8") as html_file:
				html_file.write(html)
				if os.path.getsize(file_path) > 46500: # Check if source is captured correctly
					num += 1
			browser.close()



def parse_weather(year, month, location, result_path):
	link = rf'https://www.timeanddate.com/weather/{"@1818920" if location == "ST" else "@1819783"}/historic?month={month}&year={year}'
	response = url.urlopen(link)
	html = str(response.read())
	soup = BeautifulSoup(html,'lxml')
	main_text = soup.find_all('script', attrs = {'type':"text/javascript"})[1].text.strip()
	
	content = dict()
	infos = ['temp', 'baro', 'wind', 'wd', 'hum']
	for info in infos:
		content[info] = np.zeros(31) # max 31 days
	for day in range(1, 32):
		try:
			# assume the racing takes place at noon to 6pm
			target = [f'{day} {datetime.date(year, month, day).strftime("%B %Y")}, {time}' for time in ['12:00', '18:00']]
			content['temp'][day] = int(re.findall(rf'{target[0]}.*"temp":(\d+).*{target[1]}', main_text)[0])
			content['baro'][day] = int(re.findall(rf'{target[0]}.*"baro":(\d+).*{target[1]}', main_text)[0])
			content['wind'][day] = int(re.findall(rf'{target[0]}.*"wind":(\d+).*{target[1]}', main_text)[0])
			content['wd'][day] = int(re.findall(rf'{target[0]}.*"wd":(\d+).*{target[1]}', main_text)[0])
			content['hum'][day] = int(re.findall(rf'{target[0]}.*"hum":(\d+).*{target[1]}', main_text)[0])
		except Exception as e:
			pass

# 	Save file to pickle
	month = str(month) if month >= 10 else f'0{month}'
	file_name =f'{year}_{month}_{location}'
	file_path = os.path.join(result_path, rf'{year}_{month}_{location}.pickle')
	with open(file_path, 'wb') as handle:
		pickle.dump(content, handle)



# MAKE A CLASS OBJECT
def parse_remedial_hkjc(error_path, chrome_path):
	files = os.listdir(error_path)
	result_path = os.path.join(error_path, 'Replaced')
	try:
		os.mkdir(result_path)
	except Exception as e:
		pass

	try:
		for file in files:
			year, month, day, location, _, race = file.split('_')
			race = race[:2]
			print(f'Working on {year}_{month}_{day}_{location}_{race}.txt...')

			i = 0
			while i < 4:
				
				url = rf'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={year}/{month}/{day}&Racecourse={location}&RaceNo={race}'
				options = Options()
				options.add_argument('--no-sandbox')
				browser = webdriver.Chrome(executable_path = chrome_path, options = options)
				browser.get(url)
# 				time.sleep(1)
				html = browser.page_source
				file_path = os.path.join(result_path, rf'{year}_{month}_{day}_{location}_race_{race}.txt')
				with open(file_path, 'w', encoding = "utf-8") as html_file:
					html_file.write(html)
					if os.path.getsize(file_path) < 1500: # Check if source is captured correctly
						i += 1
					else:
						i = 4
				browser.close()
			print(f'{year}_{month}_{day}_{location}_{race}.txt has been created')
	except Exception as e:
		print('Remedial parsing on HKJC results done!')
		return






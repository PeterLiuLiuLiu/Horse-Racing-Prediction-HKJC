# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:12:59 2020

@author: raymond-cy.liu
"""

import pandas as pd

def read_pickle_file(file):
	pickle_data = pd.read_pickle(file)
	return pickle_data


'''
code for creating rough directories for storing intermediate code and data files
'''

import os

print(os.path.exists('data/rough_data'))

print(os.path.exists('code/rough_code'))

if not os.path.exists('data/rough_data'):
    os.makedirs('data/rough_data')
    
if not os.path.exists('code/rough_code'):
    os.makedirs('code/rough_code')

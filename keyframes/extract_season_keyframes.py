import os
import cv2
import pandas as pd
from extract_episode_keyframes import extract_keyframes

path = os.getcwd()

# Get all files with .mkv extension
mkv_files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.mkv')]

# Create a dictionary of matching .xlsx and .mkv files
file_dict = {}
for f in os.listdir(path):
    if os.path.isfile(os.path.join(path, f)):
        if f.endswith('.xlsx'):
            excel_file = os.path.join(path, f)
            matching_mkv = os.path.splitext(excel_file)[0] + '.mkv'
            if matching_mkv in mkv_files:
                file_dict[excel_file] = matching_mkv

extract_keyframes(file_dict)
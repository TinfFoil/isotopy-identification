import os
import re

# Get the current working directory
folder_path = os.getcwd()

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Define a regular expression to match the video file names
regex = re.compile(r'^.*[Ss](\d{2})[Ee](\d{2})\..*$')

# Iterate over the files
for filename in file_list:

    # If the file is not an Excel file and matches the regular expression
    if not filename.endswith('.xlsx') and regex.match(filename):
        # Extract the season and episode numbers
        season_num, episode_num = regex.findall(filename)[0]
        
        # Construct the new filename
        new_filename = f'S{season_num}E{episode_num}.mkv'
        
        # Rename the file
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

regex_excel = re.compile(r'^GAS(\d{2})[Ee](\d{2}).*\.xlsx$')

# Iterate over the files
for filename in file_list:

    # If the file is an Excel file and matches the regular expression
    if regex_excel.match(filename):
        # Extract the season and episode numbers
        season_num, episode_num = regex_excel.findall(filename)[0]
        
        # Construct the new filename
        new_filename = f'S{season_num}E{episode_num}.xlsx'
        
        # Rename the file
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
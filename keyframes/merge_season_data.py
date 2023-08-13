import pandas as pd
import os

# get a list of all xlsx files in the directory
file_list = [filename for filename in os.listdir() if filename.endswith('.xlsx')]

# read the first file and create a new dataframe
df = pd.read_excel(file_list[0])

# iterate through the remaining files and append them to the dataframe
for file in file_list[1:]:
    temp_df = pd.read_excel(file)
    df = df.append(temp_df, ignore_index=True)

# get the first three characters of the first file name
final_name = file_list[0][:3]

# write the dataframe to a new file with the final name
df.to_excel(f'{final_name}.xlsx', index=False)
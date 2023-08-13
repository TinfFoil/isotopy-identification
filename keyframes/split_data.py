from openpyxl import Workbook
import pandas as pd

def split_data(path): # Takes the path of an Excel file with all the data

    # Read the Excel file into a dataframe
    data = pd.read_excel(path)

    # Select rows where PP, SP, and MC are not all 0
    data = data[(data['PP'] != 0) | (data['SP'] != 0) | (data['MC'] != 0)]

    # Reset the index
    data = data.reset_index(drop=True)

    # Create a dictionary to hold the separate dataframes
    data_dict = {}

    # Iterate over the sorted data
    for i, (code, group) in enumerate(data.groupby('Code')):
        # Split the data and add to the dictionary
        data_dict[code] = group

        # Save the dataframe as an .xlsx file
        filename = f'{code}.xlsx'
        with pd.ExcelWriter(filename) as writer:
            group.to_excel(writer, index=False)

        # Clear the data from the dictionary to save memory
        del data_dict[code]

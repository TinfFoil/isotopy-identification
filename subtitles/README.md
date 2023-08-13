# Subtitles

This folder contains the software for the automatic alignment of temporal annotations and subtitles.

## Contents

1. [Episode_data_preparation.ipynb](https://github.com/TinfFoil/dar_tvseries/blob/main/episode_data_preparation.ipynb): description of the approach followed to align the subtitles with the corresponding segments.
2. [Season_data_preparation.ipynb](https://github.com/TinfFoil/dar_tvseries/blob/main/season_data_preparation.ipynb): an example of how the alignment works on larger amounts of data.
3. [Data_preprocessing.ipynb](https://github.com/TinfFoil/dar_tvseries/blob/main/data_preprocessing.ipynb): preprocessing the data from season_data_preparation.ipynb.

## Usage

All files are in Jupyter notebook format and can be opened directly in Google Colab or executed locally. The expected input files are specified at the beginning of each notebook. When executing the software, an input box will appear asking the user for the path of the files. In the case of *episode_data_preparation.ipynb* and *season_data_preparation.ipynb*, the user is also given the option of exporting the data in Excel format.

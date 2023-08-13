# Expanding the corpus by extracting a keyframe for each segment

This folder contains the software used to extract the keyframes, which are required to train the multimodal model.

## Requirements

A folder with:

1. .xlsx files with the data of each episode (subtitles already included, see folder [/isotopy-identification/subtitles]([https://github.com/TinfFoil/dar_tvseries](https://github.com/TinfFoil/isotopy-identification/tree/main/subtitles))).
2. .avi or .mkv episodes of a season.

## Procedure

1. Run [rename_videos_and_data.py](https://github.com/ffedox/isotopy-identification/blob/main/keyframes/rename_videos_and_data.py) in the directory of the season.
2. Run [extract_season_keyframes.py](https://github.com/ffedox/isotopy-identification/blob/main/keyframes/extract_season_keyframes.py).
3. Run [merge_season_data.py](https://github.com/ffedox/isotopy-identification/blob/main/keyframes/merge_season_data.py) to merge all dataframes into a single dataframe for the whole season.

## Notes

To split the whole dataset into episodes, use [split_data.py](https://github.com/ffedox/isotopy-identification/blob/main/keyframes/split_data.py).


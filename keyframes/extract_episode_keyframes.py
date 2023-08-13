# One episode

import cv2
import pandas as pd
from pathlib import Path
import numpy as np

def extract_keyframes(data_dict):

    for data_path, video_path in data_dict.items():
        # Load the data

        data_path = Path(data_path)
        data = pd.read_excel(data_path)

        # Open the video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception("Could not open video file")

        # Extract keyframes for each midpoint
        img_names = []
        for index, row in data.iterrows():
            start_time = pd.to_timedelta(row['Segment start'])
            end_time = pd.to_timedelta(row['Segment end'])
            duration = end_time - start_time
            midpoint = start_time + duration / 2

            video.set(cv2.CAP_PROP_POS_MSEC, midpoint.total_seconds() * 1000)
            success, image = video.read()
            if success:
                filename = f"{data_path.stem}_{index}.jpg"
                cv2.imwrite(filename, image)
                img_names.append(filename)
            else:
                img_names.append('Midpoint not in video')

        # Release the video
        video.release()

        # Append the keyframe filename column to the excel file
        data["img_name"] = img_names
        data.to_excel(data_path, index=False)

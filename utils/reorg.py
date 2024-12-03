import json
import os
import shutil
import numpy as np
import pandas as pd

wlasl_dir='/home/tkg5kq/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5/'

## move videos to their own directories
def move_videos(wlasl_dir, wlasl_df, new_dir='videos_org', threshold=10):
    idx = 0
    for row in wlasl_df.iterrows():
        word = row[1][0]
        instance_len = len(row[1][1])
        video_ids_exists = row[1][2][0]
        video_ids_no_exists = row[1][2][1]
        if len(video_ids_exists) > threshold: 
            print(word, len(video_ids_exists), len(video_ids_no_exists), len(video_ids_exists) + len(video_ids_no_exists), instance_len)
    
            dst_path = f'{wlasl_dir}{new_dir}/{word}'
            os.makedirs(dst_path, exist_ok=True)
            for idx in video_ids_exists:
                src = f'{wlasl_dir}videos/{idx}.mp4'
                dst = dst_path + f'/{idx}.mp4'
                shutil.copyfile(src, dst)

def get_video_ids(json_list):
    video_ids = []
    no_exist = []
    for ins in json_list:
        video_id = ins['video_id']
        if os.path.exists(f'{wlasl_dir}videos/{video_id}.mp4'):
            video_ids.append(video_id)
        else:
            no_exist.append(video_id)
    return video_ids, no_exist

import sys
def main():
    file = sys.argv[1]
    threshold = int(sys.argv[2])
    wlasl_dir='/home/tkg5kq/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5/'
    json_file = wlasl_dir + 'WLASL_v0.3.json'

    wlasl_df = pd.read_json(json_file)

    with open(json_file, 'r') as data_file:
        json_data = data_file.read()

    instance_json = json.loads(json_data)

    video_ids = get_video_ids(instance_json[0]['instances'])

    # index 0 exists
    # index 1 does not exist
    wlasl_df["video_ids"] = wlasl_df["instances"].apply(get_video_ids)

    move_videos(wlasl_dir, wlasl_df, file, threshold)

    print(f'{wlasl_dir}videos_org/')
main()
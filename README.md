# UVDAR-future-pose-estimation-
Link to data: 
https://drive.google.com/drive/folders/1w9zaxZLLrvIT-LrW3HTRXhg3Q1Xx1Tbu?usp=sharing

Structure Overview:

./data
    - flight rosbags
    - csv data from rosbags
    - bag_parser.py to convert rosbags to csv

./trajectories
    - .txt files to be used as trajectories in mrs_uav_trajectory_loader
    - trajectory_generator.py to generate various trajectory text files

./neural_networks
    - current_pose_estimation

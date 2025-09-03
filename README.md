# UVDAR-future-pose-estimation-
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
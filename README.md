# UVDAR-future-pose-estimation-
Link to data: 
https://drive.google.com/drive/folders/1w9zaxZLLrvIT-LrW3HTRXhg3Q1Xx1Tbu?usp=sharing

Structure Overview:


TOOLS:
- python3 bag_parser.py <path_to_bag> to convert rosbags to csv
- python3 visualize_flight.py <path_to_csv> to generate graphs for a flight
- run_data_collection.sh <dataset> <flight_number/name> (after starting two drones sim.)

./data
    - flight rosbags
    - csv data from rosbags
    - bags and csv files are sorted into the dataset/type of trajectory, and then the individual flights 

./trajectories
    - .txt files to be used as trajectories in mrs_uav_trajectory_loader
    - trajectory_generator.py to generate various trajectory text files

./neural_networks
    - current_pose_estimation

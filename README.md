# UVDAR-future-pose-estimation-

MRS TOOLS:
- start_mrs_system.sh starts docker and connects it to xwindow to enable visualisation from docker
- mrs (in docker), sets up the system

DATA TOOLS:
- python3 bag_parser.py <path_to_bag> to convert rosbags to csv
- python3 visualize_flight.py <path_to_csv> to generate graphs for a flight
- start_simulation.sh (in docker), start two drone simulation
- run_data_collection.sh <dataset> <flight_number/name> (in docker) runs all necessary commands and collects odom and uvdar position data of a flight trajectory, in trajectory1.txt in uav_trajectory_loader. It turns off automatically after the data is collected.

 
ML TOOLS: 
- python3 current_pose_estimation_nn.py <path_to_csv_folder>

Structure Overview:

./data
    - flight rosbags
    - csv data from rosbags
    - bags and csv files are sorted into the dataset/type of trajectory, and then the individual flights 

./trajectories
    - .txt files to be used as trajectories in mrs_uav_trajectory_loader
    - trajectory_generator.py to generate various trajectory text files

./neural_networks
    - current_pose_estimation

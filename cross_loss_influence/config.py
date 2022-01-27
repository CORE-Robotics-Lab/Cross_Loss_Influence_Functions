# Created by Andrew Silva
import json
import os
path_to_here = os.path.abspath(__file__).split('config.py')[0]
con_file = open(os.path.join(path_to_here, "config.json"))
config = json.load(con_file)
con_file.close()
DATA_DIR = config['data_directory']
PROJECT_HOME = config['project_home']
MODEL_SAVE_DIR = config['model_save_directory']
PROJECT_NAME = config['project_name']
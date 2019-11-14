import numpy as np
import pandas as pd 
import argparse as ap
import matplotlib as plt

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_yaml

class gasDetector:
    def __init__(self, model_path, weights_path):

        '''
        Initialize Gas Detector CNN Model.
        Arguments:
            model_path    = path to .yaml model file. [str]
            weights_path  = path to .h5 weights file. [str]
        '''
        
        # Read Model YAML file.
        yaml_file = open(model, 'r')
        loaded_yaml_model = yaml_file.read()
        yaml_file.close()
        
        # Load model from YAML file.
        self.model = model_from_yaml(loaded_yaml_model)

        # Load CNN weights from .h5 file.
        self.model.load_weights(weights)

    def analyze_video(self, video_path, gps_stream):

        '''
        Analyze the video for Gas Leaks.
        Arguments:
            video_path  = path to target video frames directory. [str]
            gps_stream  = dataframe containing stream of gps coordinates from drone. [pd.DataFrame]
        Returns:
            detected_leaks = dataframe containing the timestamp of the detected leaks and corresponding gps coordinate. [pd.DataFrame]
        '''
        video = video_path

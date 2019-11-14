import os
import cv2
import numpy as np
import argparse as ap 

class preprocessor:
    
    def __init__(self, img_shape, img_format='jpg'):

        ''' 
        Initialize the preprocessor instance.
        Arguments:
            img_shape   = tuple containing (width, height) of the output frames. [int]
            img_format  = string for the output image format. [str] [default=jpg]
        '''

        self.img_shape_  = img_shape
        self.img_format_ = img_format

    def preprocess_video(self, path_to_video, export_path, frame_rate, echo=False):

        '''
        Preprocess the input video and save it as frames in export path.
        Arguments:
            path_to_video = path to video file. [str]
            export_path   = path to export directory. [str]
            frame_rate    = video frame rate. [float]
            echo          = toggle to output processing information to terminal. [bool] [default=False]
        Returns:
            frame_count   = total frames processed. [int]
            total_seconds = duration of the video in seconds. [float]
        '''
        
        # Process path string to obtain file name.
        self.video_path_      = path_to_video  
        index_name            = self.video_path_.rfind('/') + 1
        full_name             = self.video_path_[index_name:]
        index_extension       = full_name.find('.')
        file_name_            = full_name[:index_extension]

        self.export_path_     = export_path
        
        # Create OpenCV Capture object instance.
        vid                   = cv2.VideoCapture(self.video_path_)

        # Create MOG Substractor object instance.
        fg_bf                 = cv2.createBackgroundSubtractorMOG2()

        # Initialize frame count to 0.
        frame_count = 0

        while(True):
            
            # Read frame from capture object (video stream).
            ret, frame   = vid.read()

            # Check for correct reading of frame.
            if not ret:
                break
            
            # Apply MOG Substractor to the current frame.
            masked_frame       = fg_bf.apply(frame)

            # Invert the frame colors (white background).
            inv_masked_frame   = cv2.bitwise_not(masked_frame)

            # Resize frame to the specified shape.
            resized_frame      = cv2.resize(inv_masked_frame, self.img_shape_, interpolation = cv2.INTER_AREA)

            # Generate frame file name.
            frame_name         = self.export_path_ + file_name_ + '_' + str(index) + self.img_format_

            if echo:
                print ('Creating...' + frame_name)
            
            # Write frame to image file.
            cv2.imwrite(frame_name, resized_frame)    
            frame_count += 1
        
        # Release capture device/stream.
        vid.release()

        # Calculate total video duration.
        total_seconds = frame_count / float(frame_rate)

        if echo:
            print('Processed ' + frame_count + ' frames in total.')
            print('Video duration: ' + total_seconds +  ' seconds.')
        
        return frame_count, total_seconds
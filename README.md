# About

The aim of this application is to detect based on a stream of data if there are present any military forces. Due to the events that are happening in Ukraine, this could be a good tool in identifying enemy forces either inside the cities with surveilance cameras or outside with satelite image or drones.

In the current implementation, you can select to use either a video format or an image to check the results of the application.

# Current Progress

## Implemented
	* Face recognition using RetinaFace -> Persons detection
	* Image normalization
	* CNN for detecting if the person is a military or not

## Not Implemented
	* Vehicle detection based on shape (tried by movement, the problem with that is that I won't be knowing which CNN to use in order to determine if it is a person or a vehicle)
	* CNN for detecting if the vehicle is a military or not (it is actually implemented, but not trained)

## Attempted but failed
	* Human detection using HOG Descriptor
	* Human detection using DeepFace

## Work in progress
	* Completing the training database for Military Personnel detection (currently ~ 300 pictures, aiming for 2000)
	* Completing the training database for Vehicle Personnel detection (currently 0 pictures, aiming for 2000)

# Project Structure

## Subfolders

models: location for saving the brain of the models
saves: location for saving the predictions made by the models
training_dataset_personnel: location for the training of the MilitaryPersonnelModel

# Modules

## VideoCapture module

The aim of this module is to convert a video into frames.

### Required parameters

video_path: a path for of the video file
frames_no: the number of frames that will be taken for every occurance (1 for native framerate, 10 for example to take
every 10 frames into account)

### Functions

update_video_path: update the current path of the video
update_frames_no: update the current number of frame occurances
read_by_frame: take from the video each frame based on the number of occurances set and return a list of all the frames

## ImageReader

The aim of this module is to read an image from a path.

### Required parameters

image_path: a path for the image

### Functions

update_image_path: update the current path of the image
read_by_image: read the image and return it

## RetinaFace_HumanDetection

The aim of this module is to detect human faces from an input, compute and crop based on the size of their head, the whole body.

### Functions

part_normalize: function designed to lightly blur the image given and make it gray. returns the result
detect_by_video: function designed to detect all faces from a given list of frames, and return a list of them
detect_by_image: function designed to detect all faces from an image, and return a list of them

## MilitaryPersonnelModel module

### Functions

normalize: normalizes a given image and returns the results
create_model: creates a ML model and returns it
gather_training_data: from the data-training folder, gets all the date and categorizes it with the answers. returns a tuple of image - (0/1 for non-military/military)
train_and_save: based on the generated model and dataset, train the model, save the brain, and plot the loss
evaluate_frames: based on a saved brain, parse a list of people from a video and save them to the corresponding folders based on the prediction made
evaluate_image: based on a saved brain, parse a list of people from an image and save them to the corresponding folders based on the prediction made

# Setup

1. Create a new environment: `python3 -m venv <name_of_the_virtualenv>`
2. Activate the environment: `source <name_of_the_virtualenv>/bin/activate`
3. Install requirements.txt: `pip install -r requirements.txt`
4. Make adjustments in main.py based on what you want to do.
5. Run main.py: `python3 main.py`
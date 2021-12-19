# ultimate-rock-paper-scissors-player

2021 Fall KAIST CS492I Introduction to Deep Learning Team Project

## Contributors

The contributers are ordered in alphabetical order.

- Team 19
  - Dohyeong Kim (@lastnone)
  - Woongyeong Yeo (@astroywg)
  - Yunghee Lee (@iv-y)

## Files

- `webcam_input.py`
  - Main routine for playing rock-paper-scissors with the AI. There are functions `main_3d` and `main_2dlstm`, to select which neural network to use.
- `train_model.py`
  - Routine for training and testing the models. There are functions `main_3d` and `main_2dlstm`, to select which neural network to train. Also, `load_from_directory` processes the `dataset` folder to load the video into tensors.
- `hand_shape_finder.py`
  - Routine for finding the hand shapes from the video file's audio with FFT. The function `read_find_hands` will process the audio data and return the found frames. The main logic shows steps to save this into a `.npy` file.
- `simple_classifier.py`
  - This is a train/test pipeline for classifying 3-second videos in the `dataset/Paper`, `dataset/Rock`, `dataset/Scissors` folders with 3D CNN. This was a proof-of-concept for testing if 3D CNNs can recognize motions. 
- `models/simple_classify.py`
  - This file contains the models we used.
- `notebooks/*`
  - These files are Colab notebooks that we actually used for finding the hand shapes and training the model. 
- `requirements.txt`
  - This is the result of `pip freeze` in a development machine.

## Directory structure

- `dataset`
  - Folder containing the original video and frame postitions. 
  - Example: `paper_a.mp4`, `paper_a.mp4.npy`, ...
- `dataset/Paper`
- `dataset/Rock`
- `dataset/Scissors`
  - Folder containing the 3-second clips of the original video for each hand shapes. 
- `data`
  - Folder containing the processed tensors at `train_model.py`'s `load_from_directory` function.
- `ckpts`
  - Folder containing train checkpoints.
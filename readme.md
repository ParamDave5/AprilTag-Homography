# April Tag Detection and Image Superimposition
This repository contains code to detect a custom April tag which is a fiducial marker. The program detects the April Tag using Fast Fourier Transform, detects its corner using Shi Thomasi corner detection. It utilizes the concepts of homography to superimpose an image over the detected April Tag. A virtual cube is also drawn over the tag using the concepts of projection and calibration matrices. Note that no inbuilt OpenCV functions were used except for FFT and Shi Thomas.

## Project Description
The project consists of two parts:
1. In part 1, the goal was to superimpose a custom image (in this case, Testudo, which is University of Maryland's mascot) over an April Tag. 
2. In part 2, the goal was to draw a 3-D virtual cube over the detected April Tag.

## Requirement:
  - Python 2.0 or above

## Dependencies:
  - OpenCV
  - NumPy
  
## Instructions to run the code:
Run the following command in the terminal for the corresponding problem.

To detect edges of the April Tag using FFT:
```
python p1part1.py
```
To count boxes :
```
python p1part2.py
```
To superimpose an image over the April Tag:
```
python p2part1.py
```
To draw 3-D virtual cube over the April Tag:
```
python p2part2.py
```


## Results
Fast Fourier Transform followed by Low pass filter to detect edges of April Tag
<p align="center">
  <img src=https://github.com/AbhijitMahalle/AR_tag_detection/blob/master/results/fft.png>
<p align="center">
  
### Input Video

<p align="center">
  <img src=https://github.com/ParamDave5/AprilTag-Homography/blob/776a07dee24ca85ab4a0c4f346f06ea6147639b3/outputs/input.gif> 
<p align="center">
  
### Part 1
<p align="center">
  <img src=https://github.com/ParamDave5/AprilTag-Homography/blob/776a07dee24ca85ab4a0c4f346f06ea6147639b3/outputs/cube.gif> 
<p align="center">
  
### Part 2
<p align="center">
  <img src=https://github.com/ParamDave5/AprilTag-Homography/blob/776a07dee24ca85ab4a0c4f346f06ea6147639b3/outputs/testudo.gif>
<p align="center">


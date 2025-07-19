## Car Plate Recognition project

This project is part of the course of study of Computer Vision, degree of Artificial Intelligence and Robotics, from the university of Sapienza (Rome). 
In this project we're faced with the complex problem of developing a model for the complex task of license plates recognition in urban enviroments.\
The data used to train and evaluate the model is from the [CCPD2019](https://github.com/detectRecog/CCPD) dataset. Moreover, we're following the approach
defined in the following paper:
> Tao, L., Hong, S., Lin, Y., Chen, Y., He, P. and Tie, Z. (2024). A Real-Time License Plate Detection and Recognition Model in Unconstrained Scenarios. Sensors, 24(9), 2791

### About the model

The model is composed of two models: 
1. resnet18, deployed as a detection algorithm
2. LPRnet, used fo the recognition part of the task

The counterparts of the these models in the original work are Yolov5 and PDLPRnet. (+ results)

### Running the code

The model created in this project is purely educational, but if you want to run it, there are only to main steps to follow:
- download the CCPD2019 dataset and place it in the same directory as this project.
- read installs.txt for the needed libraries
Once prepared the enviroment, you may execute the code in the 'train' and 'evaluate' files to analyze the training and testing of our models.

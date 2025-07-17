## Car Plate Recognition project

This project is part of the course of study for Computer Vision, degree of Artificial Intelligence and Robotics, from the university of Sapienza (Rome). 
The whole project consists on the choice and training of a model for the non-trivial job of license plates recognition in urban enviroments.
The model in particular is composed of 2 models: a detection model to find and crop a license plate from an image, and a recognition model to
decipher the characters on the plate itself.
This approach is based on the following paper: 
Tao, L., Hong, S., Lin, Y., Chen, Y., He, P. and Tie, Z. (2024). A Real-Time License Plate Detection and
Recognition Model in Unconstrained Scenarios. Sensors, 24(9), 2791
credits to them.
The models are both trained on the [CCPD2019](https://github.com/detectRecog/CCPD) dataset.

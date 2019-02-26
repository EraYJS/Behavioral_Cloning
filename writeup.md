# Behavioral Cloning 


The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Files Submitted

The submission files can be used to run the simulator in autonomous mode, including the following files:

* `Behaviour_Cloing.ipynb`: all codes to create the dataset from raw data, and to build and train the model.
* `drive.py`: for driving the car in autonomous mode, unchanged.
* `model.h5`: the trained deep convolutional neural network 
* `testrun.mp4`: the video recording the test run of the model
* `writeup.md`: the report summarizing the work

## Dataset Preparation

I drove the car in the simulator for two laps, and kept it in the track, but the steering angles recorded were relatively smaller than the sample data provided by Udacity. Also, there are some tips from the web saying that using a joystick generates better data than using a keyboard, but my PS4 controller seems incompatible with the software as it can only control acceleratation or deceleration, but not steering.

Hence I ended up using the sample data. To augment the data, I flipped the images and inversed the corresponding steering angle, which essentially double the data. 

## Model Architecture and Training

### 1. Final Structure

The final network structure is as follows: 

| Layer (type)                                | Output Shape        | Param   | Setting |
|:--------------------------------------------|:-------------------:|:-------:|:------------------------|
|`batch_normalization_v1` (BatchNormaliztion) | (None, 160, 320, 3) | 12      | batch_size=256          |
|`conv2d` (Conv2D)|(None, 160, 320, 8)|608|kernel_size=5, stride=1， activation='relu', padding='same'|
|`conv2d_1` (Conv2D)|(None, 78, 158, 16)| 3216|kernel_size=5, stride=2, activation='relu', padding='valid'|
|`max_pooling2d` (MaxPooling2D)|(None, 77, 157, 16)|0|pool_size=2, stride=2, padding='valid'|
|`conv2d_2` (Conv2D)|(None, 37, 77, 32)|12832|kernel_size=5, stride=2, activation='relu', padding='valid'|
|`conv2d_3` (Conv2D)|(None, 18, 38, 32)|9248|kernel_size=3, stride=2, activation='relu', padding='valid'|
|`max_pooling2d_1` (MaxPooling2D)|(None, 8, 18, 32)|0|pool_size=4, stride=2, padding='valid'|
|`conv2d_4` (Conv2D)|(None, 6, 16, 64)|18496|kernel_size=3, stride=1, activation='relu', padding='valid'|
|`conv2d_5` (Conv2D)|(None, 4, 14, 64)|36928|kernel_size=3, stride=1, activation='relu', padding='valid'|
|`dropout` (Dropout)                          | (None, 4, 14, 64)   | 0       | dropout=0.4             |
|`flatten` (Flatten)                          | (None, 3584)        | 0       |                         |
|`dense` (Dense)                              | (None, 1024)        | 3671040 |                         |
|`dense_1` (Dense)                            | (None, 256)         | 262400  |                         |
|`dense_2` (Dense)                            | (None, 64)          | 16448   |                         |
|`dense_3` (Dense)                            | (None, 16)          | 1040    |                         |
|`dense_4` (Dense)                            | (None, 1)           | 17      |                         |

The network have 4 million trainable parameters, which are very suffient, if not overkill, for the purpose. In the following sections, I will introduce my initial thoughts and follow up modifications.

### 2. Initial Thoughts

In the previous traffic sign classifier project, the neural network takes in `32*32*1` images, and outputs the probabilities of the given sample being all 43 kinds of signs. A shallow network like LeNet works really well.

But in this project, the input size is significantly larger, of `160*320*3`, requiring a deeper network to learn the features of elevated quantity. I doubled the number of convolutional layers, and added some more dense layer on the top of the network. Also, initially, every convolutional layer has more filters than the final version. 

For the dense layers, I set the size of output to let it decreases approximately the same ratio at each layer.

### 3. Training Strategy

The loss function I chose is  Mean Square Error (MSE), as it very straightforward to express the difference of two mumbers. As for optimiser, Adam, which tunes learning rate dynamically, is still in use. As for epochs, I set a large number (10) initially, and observe at which epoch the model seemed like overfitted. Then finetune the setting to reach the maximum epoch that fits just fine. As the result turns out, 5 epochs is the sweet setting.

### 4. Modifications and Improvements

The initial network outputs an unconverged model which performed terribly, with a final loss of 0.126. To address this, I decreased the number of filters in convolutional layers, and modified the size of dense layers correspondingly. In addition to that, I also increase the size of pooling layer as well as the strides of some convolutional layers.

### 5. Other Thoughts

The project materials encourage us to crop the input, so as to rid the model of distractions such as landscapes other than the road. I managed to achieve the same purpose without doing so. Perhaps one may argue that the model merely memorized the track, and indeed, the model that succeeded in track 1, performs poorly in the harder one. But in my opinion, end-to-end deep networks only performs agreeably in familar conditions due to its statistical nature, hence I think the best way to autonomously steer a vehicle is to integrate visual geometry and deep learning detection, i.e. lane lines and other traffic participants.

## Results

The mp4 file is composed by `video.py` with testrun images and the default setting of 60fps. In the video, we can see the the car stays on the road at almost all times. However, it does, though rarely, draw very close the the edge of the road, after which it will correct itself to get back to the center of the raod. Hence the model successfully cloned my behavior on this track.

Another thing worth mentioning is that this project does not have a objective and precise performance metric such as a classification accuracy, we judge the performance of the model by roughly observe how well it finish the track. Hence to prevent the performance from being damaged, the speed setting of `drive.py` should be similar to the speed of the dataset.



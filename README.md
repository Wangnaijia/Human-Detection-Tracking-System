# Python Implementation of human detector and tracking system 
## First part: system GUI
this real time system is suitable for embedded platform or robotic platform 

code tested on ubuntu 16.04 and python 2.7, intel i5-8265 CPU

you can start with the main program of tkGUI, which is a GUI of human detecting and tracking system

run the code by:
```
$ cd TrackingDemo

$ python tkGUI.py
```
The detector is by hog+svm and the tracker is by eco-hc

Camera imput is available in this system, and stereo camera is captured at 1280x480 and both left and right image is 640x480, the disparity map by SGBM is showed as well 

You can refer to <help> in GUI menu to get more information

`detect_for_tracker.py` is used in detector

`measure.py` is used to get depth by stereo camera

stereoCam.py is a module of stereo camera

The environment you need to prepare is :

sudo apt-get install python-tk

pip install numpy scipy python-opencv glob pandas pillow PIL imutils

cd pyECO/eco/features/

python setup.py build_ext --inplace

if you want to use deep feature 
```
pip install mxnet-cu80(or 90 according to your cuda version)

pip install cupy-cu80(or 90 according to your cuda version)

```
Convert to deep feature version

uncomment `eco/conpytfig/config.py` at line5 and comment `eco/config/config.py` at line 6

for deep version, here use ResNet50 feature instead of the original imagenet-vgg-m-2048

## Second part: an implementation of HOG+SVM train and test
this part is in folder /TrackingDemo/detector

detector_demo is a main program, set flag to 'True' to run different script

1. `picture_cut.py` is for image cropping to get samples of 64x128 from common pictures, the cropped images is saved to `../data/images`

2. `extract_features.py` is for extract hog features from samples in ./data/images, and saved in `../data/features` 

3. `train_svm.py` is for training your own svm classifier based on features pos&neg, the model saved in `../data/models`

4. `PSO_PCA.py` is for optimal the C and gamma parameter of SVM classifier using PSO and PCA, where PCA is to reduce the dimension of HOG features, here the classifier is imported in `../lib/libsvm/`

5. `cross_train.py` is for cross training to reduce the error detection rate by adding the hard example into negative samples folder, eg:`../data/features/neg`, prepared for retraining

6. `classifier_result.py` is for testing the accuracy and fps of your model, all the test samples is from INRIAN data, the results containing the FP\FN\TP\TN will be saved in `output.txt`

7. `detectNMS.py` is for using the SVM model to detect human in image, using non-max-suppression and sliding window

8. `detector_config` is for parameters and path configure

by this package, you can use HOG+SVM to train your own classifier and apply in specific environment

the results is showed below:

## Third part: an implementation of trackers  
note: the interface of ECO tracker is different from vot trackers

python wrapper script file named `XXXtracker.py`，such as `KCFtracker.py`. These trackers can be integrated into the VOT evaluation process. There is a demo file `vot_demo_tracker.py` representing how to write the wrapper script file.

Trackers that have been implemented are as follows:

- `KCFtracker.py` High-Speed Tracking with Kernelized Correlation Filters (KCF) [[PDF]](http://home.isr.uc.pt/~henriques/publications/henriques_tpami2015.pdf)

- `DSSTtracker.py` Accurate Scale Estimation for Robust Visual Tracking (DSST) [[PDF]](http://www.cvl.isy.liu.se/en/research/objrec/visualtracking/scalvistrack/ScaleTracking_BMVC14.pdf)

- `HCFtracker.py` Hierarchical Convolutional Features for Visual Tracking (HCF) [[PDF]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Ma_Hierarchical_Convolutional_Features_ICCV_2015_paper.pdf)

the environment is as followed:

Python 2.7.12 scikit-image 0.13.0 scipy 0.19.1 matplotlib 1.5.3 numpy 1.13.1 pytorch 0.1.12
opencv 2.4.11

If the algorithm requires hog features, make sure `pyhog` is in directory.

`pyhog` folder includes a implementation of HOG feature. This implementation is copied from https://github.com/dimatura/pyhog

If the algorithm requires deep features (we use pretrained vgg19 in general), you need to download model file 'vgg19.pth' by url:

https://download.pytorch.org/models/vgg19-dcbb9e9d.pth

and place it in project directory.

Then you can use this tool:

If you want to evaluate on VOT, use [vot-toolkit](https://github.com/votchallenge/vot-toolkit) to evaluate the tracking algorithm on VOT datasets.
and download Visual Object Tracking (VOT) challenge datasets through the following links:

[VOT2015](http://data.votchallenge.net/vot2015/vot2015.zip), [VOT2016](http://data.votchallenge.net/vot2016/vot2016.zip), [VOT2014](http://data.votchallenge.net/vot2014/vot2014.zip), [VOT2013](http://data.votchallenge.net/vot2013/vot2013.zip)

To run your own video, use `examples.py` to understand how to use and make sure that your video has been decomposed into image sequences and each image is named with a number
(if the current image corresponds to the ith frame in the video, then the name is i.jpg or 0000i.jpg, adding 0 in front of the i is OK). For example:
Here you can use the script VideoConvertFrame.py in /TrackingDemo/
Except for the image sequence, you need to provide `groundtruth.txt` file which represents the boundingbox infromation.
The boundingbox of the first frame must be give, so there are at least one line in the `groundtruth.txt` file. 
 For example:
```
20.0,30.0,50.0,100.0(x,y,w,h)
```
Of course, if there are other frames of boundingbox information, it can also be written in `groundtruth.txt`.

To run algorithms on video, refer to `PC_demo_tracker.py`

## Reference
[1] Danelljan, Martin and Bhat, Goutam and Shahbaz Khan, Fahad and Felsberg, Michael
​    ECO: Efficient Convolution Operators for Tracking
​    In Conference on Computer Vision and Pattern Recognition (CVPR), 2017

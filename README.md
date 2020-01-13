# KFNet
This is a Tensorflow implementation of "Learning Temporal Camera Relocalization using Kalman Filtering".


## About


|| DSAC++ | KFNet |
|:--:|:--:|:--:|
|7scenes-fire       | ![Alt Text](doc/fire_DSAC++_pip.gif)       | ![Alt Text](doc/fire_KFNet_pip.gif)      |
|12scenes-office2-5a| ![Alt Text](doc/office2_5a_DSAC++_pip.gif) | ![Alt Text](doc/office2_5a_KFNet_pip.gif)|
|Description | Blue - ground truth poses   | Red - estimatd poses |

## Usage

### File format

The input folder of a project should contain the files below.
* `image_list.txt` comprising the sequential full image paths in lines, 
* `label_list.txt` comprising the full label paths in lines corresponding to the images. The label files are generated by the `tofile()` function of numpy matrices which store the scene coordinates of pixels. The label maps have a resolution 8 times lower than the images. For example, for the [7scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/), the images have a resolution of 480x640, while the label maps have a resolution of 60x80,
* `transform.txt` recording the 4x4 Euclidean transformation matrix which decorrelates the scene point cloud to give zero mean and correlations.

### Environment

The codes are tested along with 
* python 2.7,
* tensorflow-gpu 1.12.

### Testing

The testing program outputs a 3-d scene coordinate map (in meters) and a 1-d confidence map into a 4-channel numpy matrix for each input image. The confidences are the inverse of predicted Gaussain variances / uncertainties. Thus, the larger the confidences, the smaller the variances are.

* Test SCoordNet
```
git checkout SCoordNet
python SCoordnet/eval.py --input_folder <input_folder> --output_folder <output_folder> --model_folder <model_folder> --scene <scene>
# <scene> = chess/fire/heads/office/pumpkin/redkitchen/stairs, i.e., one of the scene names of 7scenes dataset
```

* Test KFNet
```
git checkout KFNet
python KFNet/eval.py --input_folder <input_folder> --output_folder <output_folder> --model_folder <model_folder> --scene <scene>
```

### Training

* Train SCoordNet
```
git checkout SCoordnet
python SCoordnet/train.py --input_folder <input_folder> --model_folder <model_folder> --scene <scene>
```

* Train OFlowNet
```
git checkout OFlowNet
python OFlowNet/train.py --input_folder <input_folder> --model_folder <model_folder>
```

* Train KFNet



## Credit

This implementation was developed by [Lei Zhou](https://zlthinker.github.io/). Feel free to contact Lei for any enquiry.
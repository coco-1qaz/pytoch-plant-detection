# Pytoch Plant Healthy Detection

A plant health program on a laptop or jetson nano

# Environment

1.[python](https://python.org)(python3.6.9=<)

2.[torch](https://pytorch.org)([jetson nano torch dowmload](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048))

3.[torchvision](https://github.com/pytorch/vision)(only jetson nano)

4.camera(usb camera or the laptop has a built-in camera)

---
# Use
## Step 1
download the [dataset file](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset) and move the dataset file to the input folder

## Step 2
Go to the src folder
```
cd src
```
Run train.py Python file
```
python train.py -epochs 10
```
**Note:**```-epochs: The number of training sessions is set here to 10, which can be modified by yourself```

## Step 3
Now we can run the Camera Video Detection Plant Health program
Go to the src folder second
```
cd src
```
Run test.py Python file
```
python test.py
```
**Note:**```After running, the camera will be turned on. Please make sure that the opencv video settings are correct. For video settings, please jump to the Additional settings location```

Now you can see the recognition result and probability in the upper left corner of the screen, and have fun

# Additional settings

If you report an error when displaying the video screen, it may be because OpenCV cannot find the video display, and you need to modify the src/test.py file
## camera
If using an external camera,please modify cv2 for 40 lines. VideoCapture(0), change 0 to 1





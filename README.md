# COMP9517-PROJECT

### File Structure

```python
model_Resnet.py         # Core ResNet34 architecture (this file)
train.py                # Script for training the model
test.py                 # Script for evaluating the model
main.py                 # Entry point (e.g. training pipeline)
image_split.py          # Utility to split dataset into train/val
class_indices.json      # Mapping between class indices and labels
HOG_SVM.py				# HOG + SVM traditional method
model_Mobilenetv2.py    # MobileNetV2 model definition
```

### Features

* Custom ResNet class using BasicBlock


* Supports dynamic block depth and downsampling


* include_top toggle to remove the final classification layer for feature extraction


* Built-in support for 1000-class classification (can be changed)

### Evaluation

Each method is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### Traditional Method

#### HOG + SVM

* Implemented in HOG_SVM.py
* Extracts HOG descriptors using `skimage`, trains a multi-class SVM using `scikit-learn`
* Saves models as `.pkl` files for reuse

### Deep Learning Methods

#### ResNet34

- Implemented in `model.py`
- Trained via `train.py`, evaluated via `test.py` and `batch_predict.py`
- Uses PyTorch and torchvision, includes ImageNet pre-trained weight support

### MobileNetV2

- Implemented in `model_Mobilenetv2.py`
- Lighter architecture for fast training and inference
- Can be substituted in `train.py` by importing `mobilenet_v2`

### How to use

1. **image_split**

   * Image_split is a Separate tool file, help you to split your own dataset.
   * If you have prepared data set then you do not need to use this.
   * If you need to use this tool, don't forget to update the data path and output location according to your own setup.

2. **Train the model**

   ```python
   python train.py
   # Modify the model function import for MobileNetV2 or Resnets
   ```

3. **Evaluate the model**

   ```python
   python test.py
   # Evaluate on a structured folder of validation/test images
   ```

4. **Use the pretrained weights(optional)**

   ```python
   model.load_state_dict(torch.load("resnet34-pre.pth"))
   #https://download.pytorch.org/models/resnet34-333f7ec4.pth
   ```

5. **Run SVM Classifier**

   ```python
   python HOG_SVM.py  
   # Trains and evaluates HOG + SVM pipeline
   ```



#### **Reference**

This project was developed with reference to the following open-source implementations:

- [[WZMIAOMIAO/deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)](https://github.com/username/repo-name): Used as a reference for model architecture and training pipeline.
- [pytorch/vision](https://github.com/pytorch/vision): Official PyTorch models were used for comparison and weight loading.
- Dataset courtesy of Kaggle contributor [ankit1743](

#### **Notes**

* Tested on PyTorch ≥ 1.12, Python 3.8+
* Dataset should be split into `train/` and `val/` folders beforehand (80/20 split)
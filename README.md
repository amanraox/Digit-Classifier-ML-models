# Digit Classifier

A simple digit classifier trained in **PyTorch** using the MNIST database. This classifier is built using several models.

1. [Softmax Classifier](https://github.com/amanraox/Digit-Recognition/tree/master/models/Softmax) : Accuracy of 92.47%
2. [kernel PCA Classifier](https://github.com/amanraox/Digit-Recognition/tree/master/models/kernelPCA) : Accuracy of 92.71%
3. [1-Hidden Layer Neural Network](https://github.com/amanraox/Digit-Recognition/tree/master/models/NN) : Accuracy of 95.77%
4. [Deep Neural Network](https://github.com/amanraox/Digit-Recognition/tree/master/models/DNN) : Accuracy of 96.54%
5. [Convolutional Neural Network](https://github.com/amanraox/Digit-Recognition/tree/master/models/CNN) : Accuracy of 98.38% 

## Code Structure:

The code structure for all the models is quite similar. Each model has the following files:

- **helpers.py** : Contains all essential imports and functions
- **train.py** : Used for training and testing the accuracy of a model
- **predict.py** : Used for predicting the digit in a given image
- **model/** : Has a pre-trained model

Apart from this, a few sample images are located in the [Utils](https://github.com/amanraox/Digit-Recognition/tree/master/utils/samples) folder.

## Usage of Code:

1. To train any model:

   ```bash
   python3 train.py # Note that this will replace the pre existing model
   ```

2. To check the accuracy of any model (present in the 'model' directory):

   ```bash
   python3 train.py -use_trained_model
   ```

3. To predict the digit from an image:

   ```bash
   python3 predict.py --path <Path To Image>
   ```


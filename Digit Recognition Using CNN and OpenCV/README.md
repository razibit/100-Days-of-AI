---

# Handwritten Digit Recognition using Deep Learning

This project demonstrates how to build a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) using the MNIST dataset. The model achieves an accuracy of ~98-99% on the test set.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dataset](#dataset)
5. [Model Architecture](#model-architecture)
6. [Results](#results)

---

## **Project Overview**
The goal of this project is to build a deep learning model that can recognize handwritten digits from the MNIST dataset. The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9), each of size 28x28 pixels. We use a Convolutional Neural Network (CNN) to classify these images into their respective digit classes.

---

## **Installation**
To run this project, you need the following libraries installed:

```bash
pip install tensorflow numpy matplotlib
```

---

## **Usage**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. Run the Jupyter Notebook or Python script:
   ```bash
   python handwritten_digit_recognition.py
   ```

3. The script will:
   - Load and preprocess the MNIST dataset.
   - Build and train a CNN model.
   - Evaluate the model on the test set.
   - Save the trained model as `mnist_cnn.h5`.

---

## **Dataset**
The MNIST dataset is a collection of 70,000 grayscale images of handwritten digits (0–9). It is split into:
- **Training set**: 60,000 images
- **Test set**: 10,000 images

Each image is of size 28x28 pixels.

---

## **Model Architecture**
The CNN model consists of the following layers:
1. **Convolutional Layer 1**: 32 filters of size 3x3, ReLU activation.
2. **Max Pooling Layer 1**: Pool size of 2x2.
3. **Convolutional Layer 2**: 64 filters of size 3x3, ReLU activation.
4. **Max Pooling Layer 2**: Pool size of 2x2.
5. **Flatten Layer**: Converts 2D feature maps to 1D.
6. **Fully Connected Layer**: 64 units, ReLU activation.
7. **Output Layer**: 10 units (one for each digit), Softmax activation.

---

## **Results**
- **Test Accuracy**: ~98-99%
- **Sample Prediction**:
  ```
  Predicted Label: 7
  Actual Label: 7
  ```
---

## **Contributing**
Contributions are welcome! If you find any issues or want to improve the project, feel free to open a pull request.

---

## **Acknowledgments**
- The MNIST dataset is provided by [Yann LeCun](http://yann.lecun.com/exdb/mnist/).
- This project is inspired by various deep learning tutorials and resources.

---

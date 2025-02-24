# Cat vs Dog Classification using CNN

## 📌 Project Overview
This project implements a **Convolutional Neural Network (CNN)** to classify images of cats and dogs using the **Dogs vs Cats dataset** from Kaggle. The model is trained to distinguish between cats and dogs based on their images and predicts new images accordingly.
.
## 🚀 Technologies Used
- Python
- TensorFlow & Keras
- OpenCV
- NumPy
- Matplotlib

## 📂 Dataset Structure
Ensure the dataset is structured as follows:
```
📂 Project_Folder/
   ├── 📜 train.py  (Training Script)
   ├── 📜 predict.py  (Prediction Script)
   ├── 📂 Dogs_VS_Cats_Datasets/
       ├── 📂 train/
       │   ├── 📂 Dog/
       │   ├── 📂 Cat/
       ├── 📂 test/
       │   ├── 📂 Dog/
       │   ├── 📂 Cat/
```

## 🏗️ Model Architecture
The CNN model consists of:
- **Conv2D layers** with ReLU activation
- **MaxPooling layers** for downsampling
- **Flatten layer**
- **Dense layers** for classification
- **Softmax activation** in the output layer

## ⚡ Installation & Dependencies
Run the following to install required dependencies:
```sh
pip install tensorflow opencv-python numpy matplotlib
```

## 🏋️‍♂️ Training the Model
Execute the following script to train the model:
```sh
python train.py
```

## 🔍 Making Predictions
Run the following command to classify a new image:
```sh
python predict.py --image path_to_image.jpg
```

## 🎯 Performance Metrics
- Training Accuracy: **~98%**
- Validation Accuracy: **~95%**
- Loss: **Minimal Overfitting**

## 📌 Future Improvements
- Implement **data augmentation** for better generalization
- Experiment with **Transfer Learning** (e.g., ResNet, VGG16)
- Deploy the model using **Flask or FastAPI**

## 📜 License
This project is open-source under the **MIT License**.

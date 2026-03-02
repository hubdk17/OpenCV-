# OpenCV-
Dog Breed Classification with ResNet50
This project implements a dog breed classification model using a pre-trained ResNet50 convolutional neural network. The model is fine-tuned on a custom dataset of dog breeds and evaluated on a validation set. Additionally, it demonstrates how to use a pre-trained ResNet50 (on ImageNet) for general image classification using OpenCV.

Project Structure
The project consists of the following main steps:

Data Download: Downloads the dog breed classification dataset from KaggleHub.
Data Preparation: Defines image transformations and creates data loaders for training and validation.
Model Definition: Initializes a pre-trained ResNet50 model and modifies its final classification layer to match the number of dog breeds for transfer learning.
Training: Trains (fine-tunes) the ResNet50 model using Cross-Entropy Loss and the Adam optimizer on the custom dataset.
Evaluation: Evaluates the fine-tuned model on a validation set, reporting accuracy, precision, recall, and F1-score.
Breed Identification (using general ImageNet ResNet50): Demonstrates image classification using a standard ImageNet pre-trained ResNet50 and cv2 for image processing.
Dataset
The dataset used for fine-tuning this project is hosted on KaggleHub: kabilan03/dogbreedclassification. It is structured into train, val, and test directories, each containing subdirectories for different dog breeds.

Model
The primary model used is ResNet50, pre-trained on ImageNet. For the dog breed classification task, the final fully connected layer is replaced to classify 93 distinct dog breeds (transfer learning). A separate instance of the ImageNet pre-trained ResNet50 is also used for general image classification demonstration.

Setup and Installation
To run this code, you need to have Python and the following libraries installed:

kagglehub
torch
torchvision
scikit-learn (for evaluation metrics)
opencv-python (for image processing in breed identification)
You can install them using pip:

pip install kagglehub torch torchvision scikit-learn opencv-python
Usage
1. Download Data
The dataset is automatically downloaded to a local cache directory using kagglehub.dataset_download.

import kagglehub
base_path = kagglehub.dataset_download("kabilan03/dogbreedclassification")
2. Train the Model (Transfer Learning)
Run the training cell (CGjKnssbdVJa) in the notebook. This will:

Load the dataset.
Apply image transformations (resize, random horizontal flip, normalization).
Initialize a pre-trained ResNet50 model and fine-tune its classification layer for the specific dog breeds for EPOCHS (default: 12) epochs.
Save the trained model's state dictionary to dogbreed_resnet50.pth.
# Snippet from training cell
# ...
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
# ...
for epoch in range(EPOCHS):
    # Training loop
    # ...
torch.save(model.state_dict(), "dogbreed_resnet50.pth")
print("Training complete!")
3. Evaluate the Fine-tuned Model
Run the evaluation cell (ycu2K0apS_LC) in the notebook. This will:

Load the validation dataset.
Load the previously saved fine-tuned model (dogbreed_resnet50.pth).
Perform inference on the validation set.
Calculate and print accuracy, precision, recall, and F1-score.
# Snippet from evaluation cell
# ...
model.load_state_dict(torch.load("dogbreed_resnet50.pth", map_location=device))
# ...
# Compute metrics and print results
4. Breed Identification (using cv2 and ImageNet pre-trained ResNet50)
Run the cell (U6xIFjdTty-w) that uses cv2 to load and process an image. This demonstrates a general image classification using a standard ResNet50 model pre-trained on ImageNet. This part of the code:

Loads a pre-trained ResNet-50 model from torchvision (pre-trained on ImageNet).
Downloads ImageNet labels.
Reads a sample image (sample_dog.jpg) using cv2 and converts it to RGB.
Applies standard ImageNet preprocessing transformations.
Performs inference to predict the image's category and confidence.
# Snippet from breed identification cell
# ...
import cv2
from PIL import Image
# ...
model = models.resnet50(weights='IMAGENET1K_V1') # Loads ImageNet pre-trained weights
# ...
image_cv = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
# ...
# Prediction and printing results
print(f"Prediction: {labels[category_id]}")
print(f"Confidence: {confidence:.2f}%")
Results
After fine-tuning for 12 epochs, the model achieved the following performance on the custom dog breed validation set:

Validation Accuracy : 0.7493
Validation Precision: 0.7826
Validation Recall : 0.7493
Validation F1 Score : 0.7412
These metrics indicate a reasonably good performance for classifying 93 different dog breeds. The separate demonstration using a general ImageNet pre-trained ResNet50 for 'sample_dog.jpg' yielded:

Prediction: pug
Confidence: 88.96%

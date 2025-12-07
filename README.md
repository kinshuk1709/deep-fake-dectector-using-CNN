This project implements a deepfake image detection pipeline using a convolutional neural network (CNN) to classify face images as real or fake.​

The code expects a ./data directory with train/ and val/ subfolders, each containing class-wise subdirectories for real and fake images.​

It uses Keras ImageDataGenerator to perform image preprocessing and data augmentation (rescaling, rotations, flips, zoom, shifts, shearing) to improve model generalization.​

The CNN model consists of stacked Conv2D and MaxPooling2D layers followed by dense layers with dropout, and is trained with the Adam optimizer and binary cross-entropy loss for binary classification.​

Early stopping and model checkpoint callbacks are used to prevent overfitting and automatically save the best-performing model as best_model.h5.​

After training, the script evaluates the model on the validation set and prints a detailed classification report and confusion matrix.​

Training and validation accuracy/loss curves are plotted using Matplotlib to visualize learning behavior across epochs.​

The script saves prediction results (file name, true label, predicted label) to predictions.csv for further analysis or debugging

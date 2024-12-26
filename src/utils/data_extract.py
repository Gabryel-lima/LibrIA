import cv2
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # Importando SMOTE para balanceamento
import traceback
from tqdm import tqdm

def error_log():
    """Função para registrar o erro em um arquivo log."""
    with open('error_log_extract.txt', 'w') as f:
        f.write('An exception occurred:\n')
        f.write(traceback.format_exc())
    print('An exception occurred. Check error_log.txt for details.')

def process_image(image_path, size=(32, 32)):
    """
    Process an image by resizing and converting it to grayscale.
    Args:
    - image_path: Path to the image file.
    - size: Tuple indicating the size to which the image should be resized.
    Returns:
    - Flattened array of the processed image, or None if the image could not be processed.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None  # Skip if the image could not be loaded
    image_resized = cv2.resize(image, size)
    return image_resized.flatten()

def images_to_csv(image_folder, output_csv, size=(32, 32)):
    """
    Converts all images from a folder (including subfolders) into a CSV file with flattened pixel values.
    Args:
    - image_folder: Path to the folder containing images (can have subfolders).
    - output_csv: Path to save the output CSV file.
    - size: Size to which images are resized.
    """
    data = []

    # Iterate over all files in the directory, including subdirectories
    files = []
    for root, _, filenames in os.walk(image_folder):
        label = os.path.basename(root)
        for file in filenames:
            if file.endswith(('.jpg', '.png')):  # Supports jpg and png images
                image_path = os.path.join(root, file)
                files.append((label, image_path))
    
    # Use tqdm to show the progress
    for label, image_path in tqdm(files, desc="Processing Images"):
        pixels = process_image(image_path, size)

        if pixels is None:
            continue  # Skip images that could not be processed

        # Append label and pixel data
        data.append([label] + pixels.tolist())

    # Create a DataFrame and save to CSV
    if data:
        columns = ['label'] + [f'pixel_{i}' for i in range(size[0] * size[1])]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_csv, index=False)
        print(f"CSV saved at: {output_csv}")
    else:
        print(f"No valid images were found in the specified folder {image_folder}. CSV not created.")

def balance_and_split_dataset(csv_path, train_csv_path, test_csv_path, max_samples=50000, test_size=0.2, random_state=42):
    """
    Balance and split a CSV dataset into training and test sets.
    Args:
    - csv_path: Path to the CSV containing the complete dataset.
    - train_csv_path: Path to save the training set CSV.
    - test_csv_path: Path to save the test set CSV.
    - max_samples: Maximum number of samples in the output CSV.
    - test_size: Proportion of the dataset to include in the test split (default: 0.2).
    - random_state: Random seed for reproducibility (default: 42).
    """
    df = pd.read_csv(csv_path)

    # Check if there are enough samples in each class
    class_counts = df['label'].value_counts()
    if class_counts.min() < 2:
        print("Error: No class has at least two samples. The DataFrame is empty.")
        return

    # Separate features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Create a balanced DataFrame
    columns = ['label'] + [f'pixel_{i}' for i in range(X_resampled.shape[1])]
    balanced_df = pd.DataFrame(data=np.column_stack((y_resampled, X_resampled)), columns=columns)

    # Split into training and test sets
    try:
        df_train, df_test = train_test_split(balanced_df, test_size=test_size, random_state=random_state, stratify=balanced_df['label'])
        df_train.to_csv(train_csv_path, index=False)
        df_test.to_csv(test_csv_path, index=False)
        print(f"Training set saved at: {train_csv_path}")
        print(f"Test set saved at: {test_csv_path}")
    except ValueError as e:
        error_log()
        print(f"Error while splitting the dataset: {e}")

# Example Usage
if __name__ == '__main__':
    # Convert images to CSV and balance/split datasets
    images_to_csv(image_folder="E:\\libria\\data\\asl_hands\\ASL_Alphabet_Dataset\\asl_alphabet_train",
                  output_csv="E:\\libria\\data\\asl_signals.csv")

    # balance_and_split_dataset(csv_path="E:\\libria\\data\\asl_signals.csv",
    #                           train_csv_path="E:\\libria\\data\\asl_signals_train.csv",
    #                           test_csv_path="E:\\libria\\data\\asl_signals_test.csv")

    images_to_csv(image_folder="E:\\libria\\data\\hand_keypoint_dataset_26k\\images\\train",
                  output_csv="E:\\libria\\data\\random_hands.csv")

    # balance_and_split_dataset(csv_path="E:\\libria\\data\\random_hands.csv",
    #                           train_csv_path="E:\\libria\\data\\random_hands_train.csv",
    #                           test_csv_path="E:\\libria\\data\\random_hands_test.csv")

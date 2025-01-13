import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # Importando SMOTE para balanceamento
import traceback
from tqdm import tqdm
import cv2

def error_log():
    """Função para registrar o erro em um arquivo log."""
    with open('error_log_extract.txt', 'w') as f:
        f.write('An exception occurred:\n')
        f.write(traceback.format_exc())
    print('An exception occurred. Check error_log.txt for details.')

def images_to_csv(image_folder, output_csv, batch_size=100):
    """
    Converts images from a folder into a CSV without modifying their format.
    Args:
    - image_folder: Path to the folder containing images.
    - output_csv: Path to save the output CSV file.
    - batch_size: Number of images to process before writing to the CSV.
    """
    files = []
    for root, _, filenames in os.walk(image_folder):
        label = os.path.basename(root)  # Use the folder name as the label
        for file in filenames:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):  # Supports common formats
                image_path = os.path.join(root, file)
                files.append((label, image_path))

    batch = []
    for idx, (label, image_path) in enumerate(tqdm(files, desc="Processing Images")):
        image = cv2.imread(image_path)  # Load the image as-is (BGR format)
        if image is None:
            print(f"Warning: Unable to load image {image_path}")
            continue

        # Flatten the image and append the label
        image_flattened = image.flatten()  # Preserve original resolution and channels
        batch.append([label, image_path] + image_flattened.tolist())

        # Write in batches to avoid memory overflow
        if len(batch) >= batch_size or idx == len(files) - 1:
            columns = ['label', 'image_path'] + [f'pixel_{i}' for i in range(len(image_flattened))]
            batch_df = pd.DataFrame(batch, columns=columns)

            # Write in append mode
            batch_df.to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))
            batch = []

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
    images_to_csv(image_folder="E:\\Projects\\\libria\\data\\asl_hands\\ASL_Alphabet_Dataset\\asl_alphabet_train",
                  output_csv="E:\\Projects\\\libria\\data\\asl_signals.csv")

    # balance_and_split_dataset(csv_path="E:\\Projects\\\libria\\data\\asl_signals.csv",
    #                           train_csv_path="E:\\Projects\\\libria\\data\\asl_signals_train.csv",
    #                           test_csv_path="E:\\Projects\\\libria\\data\\asl_signals_test.csv")

    images_to_csv(image_folder="E:\\Projects\\\libria\\data\\hand_keypoint_dataset_26k\\images\\train",
                  output_csv="E:\\Projects\\\libria\\data\\random_hands.csv")

    # balance_and_split_dataset(csv_path="E:\\Projects\\\libria\\data\\random_hands.csv",
    #                           train_csv_path="E:\\Projects\\\libria\\data\\random_hands_train.csv",
    #                           test_csv_path="E:\\Projects\\\libria\\data\\random_hands_test.csv")

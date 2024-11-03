import cv2
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from functools import lru_cache

@lru_cache(maxsize=None)
def process_image(image_path, size=(28, 28)):
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

def images_to_csv(image_folder, output_csv, size=(28, 28)):
    """
    Converts all images from a folder (including subfolders) into a CSV file with flattened pixel values.
    Args:
    - image_folder: Path to the folder containing images (can have subfolders).
    - output_csv: Path to save the output CSV file.
    - size: Size to which images are resized.
    """
    data = []

    # Iterate over all files in the directory, including subdirectories
    for root, _, files in os.walk(image_folder):
        label = os.path.basename(root)  # Use the folder name as the label

        for file in files:
            if file.endswith(('.jpg', '.png')):  # Supports jpg and png images
                image_path = os.path.join(root, file)
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

def balance_and_split_dataset(csv_path, train_csv_path, test_csv_path, max_samples=5000, test_size=0.2, random_state=42):
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
    if df['label'].value_counts().min() < 2:
        print("Error: No class has at least two samples. The DataFrame is empty.")
        return

    # Balance the dataset by ensuring each class has roughly equal representation
    balanced_df = df.groupby('label').apply(lambda x: x.head(max(max_samples // len(df['label'].unique()), 2))).reset_index(drop=True)

    # Split into training and test sets
    try:
        df_train, df_test = train_test_split(balanced_df, test_size=test_size, random_state=random_state, stratify=balanced_df['label'])
        df_train.to_csv(train_csv_path, index=False)
        df_test.to_csv(test_csv_path, index=False)
        print(f"Training set saved at: {train_csv_path}")
        print(f"Test set saved at: {test_csv_path}")
    except ValueError as e:
        print(f"Error while splitting the dataset: {e}")

# Example Usage
if __name__ == '__main__':
    # Convert images to CSV and balance/split datasets
    images_to_csv(image_folder="E:\\libria\\data\\asl_hands\\ASL_Alphabet_Dataset\\asl_alphabet_train",
                  output_csv="E:\\libria\\data\\signals.csv")

    balance_and_split_dataset(csv_path="E:\\libria\\data\\signals.csv",
                              train_csv_path="E:\\libria\\data\\signals_train.csv",
                              test_csv_path="E:\\libria\\data\\signals_test.csv")

    images_to_csv(image_folder="E:\\libria\\data\\hand_keypoint_dataset_26k\\images\\train",
                  output_csv="E:\\libria\\data\\hands.csv")

    balance_and_split_dataset(csv_path="E:\\libria\\data\\hands.csv",
                              train_csv_path="E:\\libria\\data\\hands_train.csv",
                              test_csv_path="E:\\libria\\data\\hands_test.csv")

    images_to_csv(image_folder="E:\\libria\\data\\landmarks\\characters",
                  output_csv="E:\\libria\\data\\landmarks.csv")

    balance_and_split_dataset(csv_path="E:\\libria\\data\\landmarks.csv",
                              train_csv_path="E:\\libria\\data\\landmarks_train.csv",
                              test_csv_path="E:\\libria\\data\\landmarks_test.csv")

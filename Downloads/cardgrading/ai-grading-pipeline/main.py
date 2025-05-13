"""
Main Python script consolidating the cardgod.ipynb notebook.
This script covers:
1. Data loading and initial exploration with Pandas.
2. Image downloading and basic enhancement with OpenCV.
3. Local parallel feature extraction from images.
4. Visualization of extracted features.
5. Example snippets for PySpark, Docker, Flask, and Google Cloud (commented out or requiring setup).
"""

import os
import cv2
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Dependencies & Setup Notes ---
# The following !pip install commands were in the notebook.
# It's recommended to manage dependencies using a requirements.txt file or environment.yml.
# !pip install datasets
# !pip install pandas
# !pip install transformers
# !pip install opencv-python opencv-python-headless matplotlib --quiet

# For PySpark (requires a Spark installation and environment):
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import pandas_udf

# For Flask (requires Flask installation):
# from flask import Flask, request, jsonify


# --- Global Variables & Configuration (modify as needed) ---
DATASET_URL = "hf://datasets/TheFusion21/PokemonCards/train.csv"

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assume the project root is the parent directory of SCRIPT_DIR if SCRIPT_DIR is 'src'
# Otherwise, SCRIPT_DIR is assumed to be the project root.
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR

# Adjusted paths to be relative to the project root and use data/raw and data/processed
SAMPLE_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "downloaded_card_images")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PROCESSED_DATASET_CSV = os.path.join(PROCESSED_DATA_DIR, 'processed_dataset.csv')

# Create directories if they don't exist
os.makedirs(SAMPLE_IMAGE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- Function Definitions ---

def download_image(img_url, img_path):
    """Downloads an image from a URL and saves it to a path."""
    if not os.path.exists(img_path):
        try:
            img_data = requests.get(img_url, timeout=10).content
            with open(img_path, 'wb') as handler:
                handler.write(img_data)
            # print(f"Downloaded: {img_path}")
            return img_path
        except requests.RequestException as e:
            print(f"Error downloading {img_url}: {e}")
            return None
    # print(f"Exists: {img_path}")
    return img_path

def basic_image_processing_and_display(image_path, title_prefix=""):
    """Reads an image, performs basic processing, and displays it."""
    img = cv2.imread(image_path)
    if img is None:
        print(f'Failed to load image: {image_path}')
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'{title_prefix}Original')
    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title(f'{title_prefix}Grayscale')
    plt.subplot(1, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title(f'{title_prefix}Edges')
    plt.suptitle(f"Processing for: {os.path.basename(image_path)}")
    plt.show()

def extract_image_features(image_path):
    """Extracts quantitative features from a single image."""
    img = cv2.imread(image_path)
    if img is None:
        # print(f'Failed to load image for feature extraction: {image_path}')
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Centering deviation
    moments = cv2.moments(gray)
    if moments['m00'] != 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
    else:
        cx = width / 2
        cy = height / 2
    centering_deviation = np.sqrt((cx - width / 2)**2 + (cy - height / 2)**2)
    
    # Edge gradient profiles (mean of Canny edges)
    edges = cv2.Canny(gray, 100, 200)
    edge_gradient = np.mean(edges) if edges is not None else 0

    # Corner-sharpness metrics (count of good features to track)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corner_sharpness = len(corners) if corners is not None else 0

    # Surface-texture descriptors (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_descriptor = laplacian.var()

    return {
        'centering_deviation': centering_deviation,
        'edge_gradient': edge_gradient,
        'corner_sharpness': corner_sharpness,
        'texture_descriptor': texture_descriptor,
        'image_path': os.path.basename(image_path)
    }

def process_images_in_parallel(df_pandas, num_images_to_process=None):
    """Downloads and extracts features from images listed in the DataFrame in parallel."""
    features_list = []
    
    if num_images_to_process is not None:
        df_subset = df_pandas.head(num_images_to_process)
    else:
        df_subset = df_pandas

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_url_idx = {}
        for idx, row in df_subset.iterrows():
            img_url = row['image_url']
            # Sanitize filename from URL or use index
            base_name = os.path.basename(img_url)
            sanitized_base_name = "".join([c if c.isalnum() or c in ('.', '_') else '_' for c in base_name])
            img_filename = f'image_{idx}_{sanitized_base_name}'[-100:] # Limit length
            
            img_path = os.path.join(SAMPLE_IMAGE_DIR, img_filename)
            future = executor.submit(download_image, img_url, img_path)
            future_to_url_idx[future] = (img_url, idx)

        processed_count = 0
        for future in as_completed(future_to_url_idx):
            img_url, idx = future_to_url_idx[future]
            try:
                downloaded_img_path = future.result()
                if downloaded_img_path and os.path.exists(downloaded_img_path):
                    features = extract_image_features(downloaded_img_path)
                    if features:
                        features['original_image_url'] = img_url
                        features['original_index'] = idx
                        features_list.append(features)
                        processed_count +=1
                        if processed_count % 50 == 0: # Print progress
                            print(f"Processed {processed_count}/{len(df_subset)} images for feature extraction...")
                else:
                    print(f"Skipping feature extraction for image {idx} (download failed or path invalid).")
            except Exception as e:
                print(f"Error processing image {idx} from URL {img_url}: {e}")

    if features_list:
        dataset = pd.DataFrame(features_list)
        dataset.to_csv(PROCESSED_DATASET_CSV, index=False)
        print(f'Processed dataset with {len(features_list)} images saved as {PROCESSED_DATASET_CSV}')
        return dataset
    else:
        print("No features were extracted.")
        return pd.DataFrame()

# --- Main Execution ---
def main():
    print("--- Starting Card Image Processing Pipeline ---")

    # --- 1. Load Initial Data with Pandas ---
    print(f"\n--- 1. Loading dataset from {DATASET_URL} ---")
    df_pandas = pd.DataFrame() # Initialize to empty DataFrame
    try:
        # The datasets library is needed to load from hf://
        from datasets import load_dataset 
        # Load the dataset, assuming it's CSV-like and can be converted to pandas
        # This might download to a cache directory first
        print("Attempting to load dataset using Hugging Face 'datasets' library...")
        # Make sure the dataset name and config are correct.
        # For "TheFusion21/PokemonCards", data_files="train.csv" might be needed if it's not the default.
        hf_dataset = load_dataset("TheFusion21/PokemonCards", data_files="train.csv", split="train")
        df_pandas = hf_dataset.to_pandas()
        print("Dataset loaded successfully via Hugging Face and converted to Pandas.")
        
        print("\nDataFrame Info:")
        df_pandas.info()
        print("\nDataFrame Head:")
        print(df_pandas.head())
        print("\nDataFrame Description:")
        print(df_pandas.describe(include='all'))
        print(f"\nNull values per column:\n{df_pandas.isnull().sum()}")
        print(f"\nNumber of duplicated rows: {df_pandas.duplicated().sum()}")
        if 'name' in df_pandas.columns:
            print(f"\nNumber of unique card names: {df_pandas['name'].nunique()}")
            # print(f"Value counts for card names:\n{df_pandas['name'].value_counts().head()}")
    except ImportError:
        print("The 'datasets' library is not installed. Please add 'datasets' to your environment.yml and rebuild the Docker image.")
        print("Attempting to load directly with pandas (this will likely fail for hf:// URLs).")
        try:
            # This direct read will fail for "hf://" URLs.
            # df_pandas = pd.read_csv(DATASET_URL) 
            print("Pandas read_csv for hf:// URLs requires the 'datasets' library. Skipping direct pandas load.")
        except Exception as e_pd:
            print(f"Error loading dataset with pandas: {e_pd}")
            return # Exit if data loading fails
    except Exception as e:
        print(f"Error loading or performing initial analysis on DataFrame: {e}")
        return # Exit if data loading fails

    if df_pandas.empty:
        print("Failed to load data. Exiting.")
        return

    # --- IPython HTML Display (Commented out for .py script) ---
    # This part is for Jupyter Notebooks.
    # from IPython.display import display, HTML
    # print("\n--- Displaying first 5 card images (HTML - Jupyter specific) ---")
    # if 'image_url' in df_pandas.columns:
        # for url in df_pandas['image_url'].head(5):
            # display(HTML(f'<img src="{url}" width="250"/>'))

    # --- 2. Example Image Enhancement with OpenCV ---
    print("\n--- 2. OpenCV Image Processing Examples ---")
    # Example 1: Process the first image from the dataset
    if not df_pandas.empty and 'image_url' in df_pandas.columns:
        first_img_url = df_pandas['image_url'].iloc[0]
        # Sanitize filename
        base_name = os.path.basename(first_img_url)
        sanitized_base_name = "".join([c if c.isalnum() or c in ('.', '_') else '_' for c in base_name])
        first_img_filename = f"sample_card_from_df_{sanitized_base_name}"[-100:] # Limit length
        first_img_path = os.path.join(SAMPLE_IMAGE_DIR, first_img_filename)
        
        print(f"Downloading and processing first image from dataset: {first_img_url}")
        if download_image(first_img_url, first_img_path):
            if os.path.exists(first_img_path):
                 basic_image_processing_and_display(first_img_path, "Dataset Img: ")
            else:
                print(f"Failed to find downloaded image at {first_img_path}")
        else:
            print(f"Failed to download {first_img_url}")


    # Example 2: Process a specific hardcoded image URL
    # specific_img_url = 'https://images.pokemontcg.io/pl3/1_hires.png'
    # specific_img_filename = "hardcoded_sample_card_1.jpg"
    # specific_img_path = os.path.join(SAMPLE_IMAGE_DIR, specific_img_filename)
    # print(f"\nDownloading and processing specific image: {specific_img_url}")
    # if download_image(specific_img_url, specific_img_path):
    #     basic_image_processing_and_display(specific_img_path, "Specific Img 1: ")

    # --- 3. Parallel Feature Extraction ---
    # Set num_images_to_process to a small number for testing, or None to process all
    NUM_IMAGES_FOR_FEATURE_EXTRACTION = 20 # Or None for all, or a larger number like 1000
    print(f"\n--- 3. Parallel Feature Extraction (Processing {NUM_IMAGES_FOR_FEATURE_EXTRACTION if NUM_IMAGES_FOR_FEATURE_EXTRACTION is not None else 'all'} images) ---")
    if not df_pandas.empty:
        processed_features_df = process_images_in_parallel(df_pandas, num_images_to_process=NUM_IMAGES_FOR_FEATURE_EXTRACTION)
        if not processed_features_df.empty:
            print("\nSample of extracted features:")
            print(processed_features_df.head())
    else:
        print("Skipping parallel feature extraction as initial DataFrame is empty.")
        # processed_features_df = pd.DataFrame() # Already initialized if df_pandas was empty


    # --- 4. Data Visualization of Extracted Features ---
    print("\n--- 4. Visualizing Extracted Features ---")
    if os.path.exists(PROCESSED_DATASET_CSV):
        try:
            vis_df = pd.read_csv(PROCESSED_DATASET_CSV)
            if not vis_df.empty:
                if 'edge_gradient' in vis_df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(x=vis_df['edge_gradient'].dropna(), bins=30, kde=True)
                    plt.title('Edge Gradient Distribution')
                    plt.xlabel('Edge Gradient')
                    plt.ylabel('Frequency')
                    plt.show()
                else:
                    print(f"Column 'edge_gradient' not found in {PROCESSED_DATASET_CSV} for visualization.")

                if 'centering_deviation' in vis_df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(x=vis_df['centering_deviation'].dropna(), bins=30, kde=True)
                    plt.title('Centering Deviation Distribution')
                    plt.xlabel('Centering Deviation')
                    plt.ylabel('Frequency')
                    plt.show()
                else:
                    print(f"Column 'centering_deviation' not found in {PROCESSED_DATASET_CSV} for visualization.")
            else:
                print(f"'{PROCESSED_DATASET_CSV}' is empty.")
        except Exception as e:
            print(f"Error during visualization: {e}")
    else:
        print(f"'{PROCESSED_DATASET_CSV}' not found. Run feature extraction first.")

    print("\n--- Pipeline Execution Finished ---")

if __name__ == "__main__":
    main()
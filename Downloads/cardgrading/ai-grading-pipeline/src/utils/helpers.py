def download_image(img_url, img_path):
    import os
    import requests

    if not os.path.exists(img_path):
        img_data = requests.get(img_url).content
        with open(img_path, 'wb') as handler:
            handler.write(img_data)
    return img_path

def load_data(file_path):
    import pandas as pd

    return pd.read_csv(file_path)

def save_data(df, file_path):
    df.to_csv(file_path, index=False)

def preprocess_image(image):
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def calculate_edge_gradient(image):
    import cv2
    import numpy as np

    edges = cv2.Canny(image, 100, 200)
    return np.mean(edges)

import os
import pandas as pd
import logging

def ingest_data(csv_path):
    """Load and validate a dataset from a CSV file."""
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found at {csv_path}")
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Successfully loaded dataset with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise
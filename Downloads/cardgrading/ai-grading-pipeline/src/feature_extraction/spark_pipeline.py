from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
import pandas as pd
import cv2
import numpy as np
import requests

spark = SparkSession.builder.appName("CardFeatureExtraction").getOrCreate()

def extract_features_udf(pdf: pd.DataFrame) -> pd.DataFrame:
    features = []
    for url in pdf['image_url']:
        try:
            img_data = requests.get(url).content
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_gradient = np.mean(edges)
            features.append({'edge_gradient': edge_gradient})
        except Exception:
            features.append({'edge_gradient': None})
    return pd.DataFrame(features)

def process_images(df):
    features_df = df.mapInPandas(extract_features_udf, schema="edge_gradient double")
    features_df.write.csv("s3://your-bucket/processed_features.csv")
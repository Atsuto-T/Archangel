from google.cloud import storage
from archangel.params import *

def save_model():

    model_filename = "exp8/weights/last.pt"
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename('yolov5/runs/train/exp8/weights/last.pt')

    print("Model saved to GCS")

    return None

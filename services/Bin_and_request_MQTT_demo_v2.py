import os
import cv2
import json
import time,glob
import logging
import numpy as np
import requests
import warnings
import ssl
from tqdm import tqdm
from pathlib import Path
from shutil import move, rmtree
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from paho.mqtt import client as mqtt_client

# Suppress warnings and TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# MQTT Configuration
BROKER = 'oomcardiodev.projectkmt.com'
PORT = 8883
CLIENT_ID = f'python-mqtt-{np.random.randint(1, 20)}'
USERNAME = 'kmt'
PASSWORD = 'Kmt123'
TOPIC_X = "oom/ecg/templateCreateRpm"
client = None

# Initialize the global model
global_model = MobileNetV3Small(weights="imagenet", include_top=False)

def connect_mqtt() -> mqtt_client:
    """Connect to MQTT broker."""
    global client

    def on_connect(client, userdata, flags, rc, protocol):
        if rc == 0:
            logging.info("Connected to MQTT")
            client.subscribe(TOPIC_X, qos=2)
        else:
            logging.error(f"Failed to connect, return code {rc}")

    client = mqtt_client.Client(CLIENT_ID, protocol=mqtt_client.MQTTv5)
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    client.tls_set_context(context)
    client.enable_shared_subscription = True
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.connect(BROKER, PORT)
    return client

def preprocess_image(image_path):
    """Loads and preprocesses an image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

def batch_image_embedding(image_paths):
    """Processes images in batches for faster embedding extraction."""
    images, valid_paths = [], []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(preprocess_image, image_paths))
    for img, path in zip(results, image_paths):
        if img is not None:
            images.append(img)
            valid_paths.append(path)
    if not images:
        return np.array([]), []
    images = np.vstack(images)
    embeddings = global_model.predict(images, verbose=0)
    return embeddings.reshape(len(valid_paths), -1), valid_paths

def hierarchical_clustering(embeddings, threshold=0.36):
    """Performs hierarchical clustering using cosine similarity."""
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix  # Convert to distance
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, affinity='precomputed', linkage='average')
    return clustering.fit_predict(distance_matrix)

def image_clustering(dd, input_folder, output_folder, patient):
    """Clusters images based on extracted features and filters by timestamp."""
    start = time.time()
    os.makedirs(output_folder, exist_ok=True)

    # Extract date range from the JSON object
    from_date = int(time.mktime(time.strptime(dd['fromDate'], '%Y-%m-%dT%H:%M:%S.%fZ'))) * 1000
    to_date = int(time.mktime(time.strptime(dd['toDate'], '%Y-%m-%dT%H:%M:%S.%fZ'))) * 1000

    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".jpg")]
    if not image_paths:
        dd.update({"template": {"status": "completed"}})
        logging.info("No images found.")
        client.publish("oom/ecg/templateUpdateRpm", json.dumps(dd), qos=2)
        return

    # Filter images based on timestamp
    filtered_image_paths = []
    for filename in image_paths:
        try:
            timestamp = int(filename.split("_")[-1].split(".jpg")[0])
            if from_date <= timestamp <= to_date:
                filtered_image_paths.append(filename)
        except ValueError:
            logging.warning(f"Filename {filename} does not have a valid timestamp format. Skipping.")

    if not filtered_image_paths:
        dd.update({"template": {"status": "completed"}})
        logging.info("No images within the date range.")
        client.publish("oom/ecg/templateUpdateRpm", json.dumps(dd), qos=2)
        return

    logging.info("Extracting embeddings...")
    image_embeddings, valid_paths = batch_image_embedding(filtered_image_paths)
    if image_embeddings.size == 0:
        logging.info("No valid embeddings generated.")
        return

    logging.info("Clustering images...")
    cluster_labels = hierarchical_clustering(image_embeddings)
    
    clusters = {}
    for path, label in zip(valid_paths, cluster_labels):
        clusters.setdefault(label, []).append(path)
    
    for label, paths in clusters.items():
        cluster_folder = os.path.join(output_folder, f"T_{label}")
        os.makedirs(cluster_folder, exist_ok=True)
        for path in paths:
            move(path, os.path.join(cluster_folder, os.path.basename(path)))

    [f.unlink() for f in Path(f"pvcs/{patient}").glob("*") if f.is_file()]
    logging.info(f"Clustering complete in {time.time() - start:.2f} seconds.")
    request_web(None,output_folder, patient, dd)

def request_web(destination_path, output_folder, patient, dd):
    """Send a request to a web server with clustering results."""
    patient_folder_name = output_folder
    image_list_data = []

    for bin_list in os.listdir(patient_folder_name):
        image_list_data = []
        for imagelist in glob.glob(patient_folder_name + "/" + bin_list + "/" + "*.jpg"):
            timestamp = imagelist.split("/")[-1].split("_")[-1].split(".jpg")[0]
            image_data = {
                "startTime": int(timestamp),
                "endTime": int(timestamp)
            }
            image_list_data.append(image_data)

        url = 'http://191.169.1.9:5000/api/v1/template/images?'
        params = {
            'patient': patient,
            'templateName': 'Ventricular',
            'name': bin_list
        }
        payload = {
            "name": bin_list,
            "patient": patient,
            "templateName": "Ventricular",
            "images": image_list_data
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.post(url, params=params, headers=headers, json=payload)
        print(payload)
    if response.status_code == 201:
        logging.info("Request successful: %s", response.text)
        dd.update({"template": {"status": "completed"}})
        client.publish("oom/ecg/templateUpdateRpm", json.dumps(dd), qos=2)
        rmtree(f"BINDATA/{dd['patient']}")
    else:
        logging.error("Request failed with status code: %d", response.status_code)

def subscribe(client):
    """Subscribe to MQTT messages and process clustering tasks."""
    def on_message(client, userdata, msg):
        decoded_message = str(msg.payload.decode("utf-8", errors="replace"))
        dd = json.loads(decoded_message)
        input_folder = f"pvcs/{dd['patient']}"
        output_folder = f"BINDATA/{dd['patient']}"
        try:
            image_clustering(dd, input_folder, output_folder, dd['patient'])
        except Exception as e:
            logging.error("Error: %s", e)
            dd.update({"template": {"status": "completed"}})
            client.publish("oom/ecg/templateUpdateRpm", json.dumps(dd), qos=2)
    client.on_message = on_message

def run():
    """Run the MQTT client and process incoming messages."""
    global client
    client = connect_mqtt()
    subscribe(client)
    while True:
        try:
            client.loop_forever()
        except Exception as e:
            logging.error("MQTT Loop Error: %s", e)

if __name__ == "__main__":
    run()

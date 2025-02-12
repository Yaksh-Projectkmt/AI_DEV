import os
import json
import glob
import shutil
import requests
import cv2
import time
import logging
import random
import ssl
import numpy as np
from pathlib import Path
from paho.mqtt import client as mqtt_client
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from shutil import move, rmtree
from warnings import filterwarnings

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# MQTT Configuration
broker = 'oomcardiodev.projectkmt.com'
port = 8883
client_id = f'python-mqtt-{random.randint(1, 20)}'
username = 'kmt'
password = 'Kmt123'
topic_x = "oom/ecg/templateCreateRpm"
client = None

# Initialize the global model
model = VGG16(weights="imagenet", include_top=False)

def connect_mqtt() -> mqtt_client:
    """Connect to the MQTT broker and subscribe to the topic."""
    global client

    def on_connect(client, userdata, flags, rc, protocol):
        if rc == 0:
            logging.info("Connected to MQTT")
            client.subscribe(topic_x, qos=2)
        else:
            logging.error(f"Failed to connect, return code {rc}")

    client = mqtt_client.Client(client_id, protocol=mqtt_client.MQTTv5)
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    client.tls_set_context(context)
    client.enable_shared_subscription = True
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def preprocess_image(image_path):
    """Preprocesses an image for VGG16 model input."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def image_embedding(image_path):
    """Extracts image embeddings using a pre-trained model."""
    img = preprocess_image(image_path)
    return model.predict(img, verbose=0).flatten()

def calculate_similarity_matrix(embeddings):
    """Calculates a pairwise similarity matrix using cosine distance."""
    return 1 - cdist(embeddings, embeddings, metric='cosine')

def hierarchical_clustering(similarity_matrix, threshold=0.26):
    """Clusters images based on similarity using Agglomerative Clustering."""
    distance_matrix = 1 - similarity_matrix  # Convert similarity to distance
    try:
      clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, affinity='precomputed', linkage='average')
    except:
      clustering = AgglomerativeClustering(n_clusters=None,distance_threshold=threshold,metric='precomputed',linkage='average')
    return clustering.fit_predict(distance_matrix)

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

def image_clustering(dd, input_folder, output_folder, patient):
    """Cluster images into groups based on similarity, filtering by timestamp."""
    start = time.time()
    os.makedirs(output_folder, exist_ok=True)

    # Extract date range from the JSON object
    from_date = int(time.mktime(time.strptime(dd['fromDate'], '%Y-%m-%dT%H:%M:%S.%fZ')))*1000
    to_date = int(time.mktime(time.strptime(dd['toDate'], '%Y-%m-%dT%H:%M:%S.%fZ')))*1000

    if not os.listdir(input_folder):
        dd.update({"template": {"status": "completed"}})
        logging.info("Nothing found in the input folder.")
        return

    # Filter images based on timestamp
    filtered_image_paths = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            try:
                # Extract timestamp from filename
                timestamp = int(filename.split("_")[-1].split(".jpg")[0])
                #print(from_date,timestamp,to_date,from_date <= timestamp <= to_date)
                if from_date <= timestamp <= to_date:
                    filtered_image_paths.append(os.path.join(input_folder, filename))
            except ValueError:
                logging.warning(f"Filename {filename} does not have a valid timestamp format. Skipping.")

    if not filtered_image_paths:
        dd.update({"template": {"status": "completed"}})
        logging.info("No images within the date range.")
        return 1

    # Compute embeddings for filtered images
    image_embeddings = [image_embedding(image_path) for image_path in filtered_image_paths]

    # Cluster based on similarity
    similarity_matrix = calculate_similarity_matrix(np.array(image_embeddings))
    cluster_labels = hierarchical_clustering(similarity_matrix)

    # Organize images into clusters
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(filtered_image_paths[i])

    # Save clustered images to output folder
    for cluster_label, cluster_images in clusters.items():
        cluster_folder = os.path.join(output_folder, f"T-{cluster_label}")
        os.makedirs(cluster_folder, exist_ok=True)
        for image_path in cluster_images:
            move(image_path, os.path.join(cluster_folder, os.path.basename(image_path)))

    [f.unlink() for f in Path(f"pvcs/{patient}").glob("*") if f.is_file()]
    logging.info("Images clustered dynamically. Results saved in: %s", output_folder)
    end = time.time()
    logging.info("Processing time: %.2f seconds", end - start)

    request_web(None, output_folder, patient, dd)

##def image_clustering(dd, input_folder, output_folder, patient):
##    """Cluster images into groups based on similarity."""
##    start = time.time()
##
##    if not os.listdir(input_folder):
##        dd.update({"template": {"status": "completed"}})
##        logging.info("Nothing found in the input folder.")
##        return
##
##    image_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith(".jpg")]
##    image_embeddings = [image_embedding(image_path) for image_path in image_paths]
##
##    similarity_matrix = calculate_similarity_matrix(np.array(image_embeddings))
##    cluster_labels = hierarchical_clustering(similarity_matrix)
##
##    clusters = {}
##    for i, label in enumerate(cluster_labels):
##        if label not in clusters:
##            clusters[label] = []
##        clusters[label].append(image_paths[i])
##
##    for cluster_label, cluster_images in clusters.items():
##        cluster_folder = os.path.join(output_folder, f"T-{cluster_label}")
##        os.makedirs(cluster_folder, exist_ok=True)
##        for image_path in cluster_images:
##            move(image_path, os.path.join(cluster_folder, os.path.basename(image_path)))
##    [f.unlink() for f in Path("pvcs/"+patient).glob("*") if f.is_file()]
##    logging.info("Images clustered dynamically. Results saved in: %s", output_folder)
##    end = time.time()
##    logging.info("Processing time: %.2f seconds", end - start)
##
##    request_web(None, output_folder, patient, dd)

def subscribe(client):
    """Subscribe to MQTT messages and process clustering tasks."""
    def on_message(client, userdata, msg):
        decoded_message = str(msg.payload.decode("utf-8", errors="replace"))
        dd = json.loads(decoded_message)
        input_folder = f"pvcs/{dd['patient']}"

        try:
            if not os.listdir(input_folder):
                dd.update({"template": {"status": "completed"}})
                client.publish("oom/ecg/templateUpdateRpm", json.dumps(dd), qos=2)
                logging.info("No images in input folder. Waiting for new data...")
                time.sleep(5)
            else:
                output_folder = f"BINDATA/{dd['patient']}"
                ret_unexpected = image_clustering(dd, input_folder, output_folder, dd['patient'])
                try:
                    if ret_unexpected == 1:
                        dd.update({"template": {"status": "completed"}})
                        client.publish("oom/ecg/templateUpdateRpm", json.dumps(dd), qos=2)
                except:
                    pass
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

# -*- coding: utf-8 -*-
import random
from paho.mqtt import client as mqtt_client
import traceback
import ast
import redis
import time
import re
from scipy.signal import medfilt, welch,argrelextrema
from scipy.fft import rfft
from scipy.signal import (find_peaks, firwin,medfilt)
from sewar.full_ref import mse, rmse, ergas,  rase, sam
from scipy.integrate import trapz
from dateutil import parser
from scipy.interpolate import interp1d
##import matplotlib
##matplotlib.use('module://thread')
import matplotlib
##matplotlib.use('Cairo')
matplotlib.use('agg')
from matplotlib import colormaps
import matplotlib.pyplot as plt
import csv
import math
import pandas as pd
from tensorflow.keras import backend as K
import ssl
import numpy as np
import random
from scipy.signal import medfilt
from scipy.stats import pearsonr
from scipy import sparse, signal
from scipy.sparse.linalg import spsolve
from statistics import pvariance,mode
import threading
import pymongo
import statistics
import matplotlib.pyplot as plt
import scipy.fftpack
import warnings
warnings.filterwarnings('ignore')
import scipy.signal as signal
import os
import pandas
from math import sqrt
import numpy as np
import datetime
import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers=['1.1.1.1']
import math
from scipy.ndimage import label
import json
import neurokit2 as nk
import seaborn as sns

# signal processing
import scipy.signal
import scipy.ndimage
from scipy import signal
from scipy.ndimage import label
import cv2
import shutil
import os
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler

# from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import glob
from statistics import pvariance
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
from BaselineRemoval import BaselineRemoval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.stats import pearsonr
from scipy import  signal
from statistics import pvariance
import tensorflow as tf
import concurrent.futures
import biosppy
import pywt
import pywt as pw
from scipy import signal
from pywt import wavedec
from scipy.stats import pearsonr
from scipy import sparse, signal
from scipy.sparse.linalg import spsolve
from statistics import pvariance
import pybeads as be
import scipy.stats as stats
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import tools as st
import utils
import uuid

#MI dependancy
from biosppy.signals import ecg as hami
from collections import Counter
from scipy.signal import butter, filtfilt
os.environ['CUDA_VISIBLE_DEVICES'] = ''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # Limit memory to 4GB
    except RuntimeError as e:
        print(e)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#thread worker
executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)

#Database Connection
##myclient = pymongo.MongoClient("mongodb://hopsadmin:fN6ZPg02713u8X9l@139.59.75.40:27017/ecgs1?authSource=admin&retryWrites=true&w=majority")
myclient = pymongo.MongoClient("mongodb://localhost:27017")
mydb = myclient["test"]
    
#MQTT Topics

##topic_x = "oom/mqtt1"
##topic_y = "oom/1"
##topic_x = "oom/ecg/rawData"
topic_y = "oom/ecg/processedDataOffline"
##


#MQTT Credentials
####
##broker = 'oomcardiotest.projectkmt.com'
##port = 8883
##client_id = f'{random.randint(1000000000, 2000000000)}'
##username = 'kmt'
##password = 'dVBbS3NxMtmzD438'

broker = 'oomcardiodev.projectkmt.com'
port = 8883
client_id = f'python-mqtt-{random.randint(1000000000, 2000000000)}'
username = 'kmt'
password = 'Kmt123'


#class RedisQueue:
#    def __init__(self, name, redis_conn):
#        self.name = name
#        self.redis = redis_conn
#
#    def push(self, item):
#        self.redis.lpush(self.name, json.dumps(item))
#
#    def peek(self):
#        item = self.redis.lindex(self.name, -1)  # Peek at the last item (FIFO)
#        return json.loads(item) if item is not None else None
#
#    def pop(self):
#        item = self.redis.rpop(self.name)
#        return json.loads(item) if item is not None else None
#
#    def size(self):
#        return self.redis.llen(self.name)
#
#    def is_empty(self):
#        return self.size() == 0
class RedisQueue:
    def __init__(self, name, redis_conn):
        self.name = name
        self.redis = redis_conn
        self.patient_counter_key = f"{name}_patient_counter"

    def push(self, item):
        # Push data to the list
        self.redis.lpush(self.name, json.dumps(item))

        # Increment patient counter
        patient_id = item.get("patient")
        self.redis.hincrby(self.patient_counter_key, patient_id, 1)

    def peek(self):
        item = self.redis.lindex(self.name, -1)
        return json.loads(item) if item is not None else None

    def pop(self):
        item = self.redis.rpop(self.name)
        if item is not None:
            try:
              # Decrement patient counter when popping an item
              patient_id = json.loads(item).get("patient")
              self.redis.hincrby(self.patient_counter_key, patient_id, -1)
            except Exception as e:
              print(e)
        return json.loads(item) if item is not None else None

    def size(self):
        return self.redis.llen(self.name)

    def delete(self, item):
        # Remove the rightmost occurrence of the specified item from the list
        self.redis.lrem(self.name, 1, json.dumps(item))

        # Decrement patient counter
        patient_id = item.get("patient")
        self.redis.hincrby(self.patient_counter_key, patient_id, -1)

    def is_empty(self):
        return self.size() == 0

    def peek_all(self):
        all_items = self.redis.lrange(self.name, 0, -1)
        return [json.loads(item) for item in all_items]

    def get_patient_data_count(self, patient_id):
        # Get the count of data entries for a specific patient
        count_bytes = self.redis.hget(self.patient_counter_key, patient_id)
    
        # Decode bytes to string and convert to integer
        count_str = count_bytes.decode('utf-8') if count_bytes else '0'
        count = int(count_str)
        
        return count

    def get_all_patient_keys(self):
        # Get all patient keys from the Redis hash
        patient_keys_bytes = self.redis.hkeys(self.patient_counter_key)
        
        # Decode bytes to string
        patient_keys = [key.decode('utf-8') for key in patient_keys_bytes]
        
        return patient_keys
    
    def hkeydelete(self,patient_id):
        self.redis.hdel(self.patient_counter_key, patient_id)



redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)
##print(redis_conn)
# Create a queue instance
queue = RedisQueue('devOfflinechannel', redis_conn)

#AI Detectron to Detect PVC Curve


results_lock = threading.RLock()

def predict_tflite_model(model:tuple, input_data:tuple):
    with results_lock:
        interpreter, input_details, output_details = model
        for i in range(len(input_data)):
            interpreter.set_tensor(input_details[i]['index'], input_data[i])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
    
    return output

def load_tflite_model(model_path):
    with results_lock:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

random_folder_name = str(uuid.uuid4())
folder_path = os.path.join("rawdataoffline/", random_folder_name)
os.makedirs(folder_path)


# Load the TFLite model
interpreterss = tf.lite.Interpreter(model_path='PVC_Trans_mob_41_test_tiny_iter1.tflite')
interpreterss.allocate_tensors()

# Get the input and output details
input_detailss = interpreterss.get_input_details()
output_detailss = interpreterss.get_output_details()

interpreter_noise = tf.lite.Interpreter(model_path='NOISE_16_GPT.tflite')
interpreter_noise.allocate_tensors()

input_details_noise = interpreter_noise.get_input_details()
output_details_noise = interpreter_noise.get_output_details()


with tf.device('/CPU:0'):
    # tf_cwt_model = load_tflite_model("JR_model_20_09_0_two_different_inputs.tflite")
    afib_load_model = load_tflite_model("afib_flutter_4_3.tflite")
    vfib_vfl_model = load_tflite_model("vfib_trans_mob_1.tflite")
    pac_load_model = load_tflite_model("PAC_TRANS_GRU_mob_24.tflite")
    block_load_model = load_tflite_model("Block_convex_2.tflite")
    let_inf_moedel = load_tflite_model("ST_21_10.tflite")

ref_files = "200_standard.csv"
ref_file  = pd.read_csv(ref_files)["ECG"][:2000]

def prediction_model_PAC(input_arr, target_shape=[224, 224], class_name=True):
    classes = ['Abnormal', 'Junctional', 'Normal', 'PAC']
    input_arr = tf.cast(input_arr, dtype=tf.float32)
    input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
    input_arr = (tf.expand_dims(input_arr, axis=0),)
    model_pred = predict_tflite_model(pac_load_model, input_arr)[0]
    idx = np.argmax(model_pred)
    if class_name:
        idx = np.argmax(model_pred)
        return model_pred, classes[idx]
    else:
        return model_pred
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def baseline_construction_200(ecg_signal, kernel_size=101):
    s_corrected = signal.detrend(ecg_signal)
    baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
    return baseline_corrected

def lowpass_4(baseline_signal, cutoff=0.3):
    b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, baseline_signal)
    return low_passed


def find_r_peak_mi(ecg_signal, fs):
    lowcut = 0.5
    highcut = 50.0
    filtered_signal = butter_bandpass_filter(ecg_signal, lowcut, highcut, fs, order=6)
    # Step 3: R-Peak Detection
    out = hami.hamilton_segmenter(filtered_signal, sampling_rate=fs)
    rpeaks = hami.correct_rpeaks(filtered_signal, out[0], sampling_rate=fs, tol=0.1)
    r_peaks = rpeaks[0].tolist()
    return r_peaks

# def mi_detection(baseline_signal,ecg_signal, r_peaks, fs):
#     try:
#         _, waves_peak = nk.ecg_delineate(baseline_signal, r_peaks, sampling_rate=fs, method="peak")

#         Speaks = np.where(np.isnan(waves_peak['ECG_S_Peaks']), 0, waves_peak['ECG_S_Peaks']).astype('int64').tolist()
#         Tpeaks = np.where(np.isnan(waves_peak['ECG_T_Peaks']), 0, waves_peak['ECG_T_Peaks']).astype('int64').tolist()
        
#     except:
#         label ="Abnormal"
# ##        print("<<<<<<<<<<<<<<<<<<<<<<<<< something wrong >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#         return label

#     mi_counter = 0  # Counter for MI detections
#     for r_peak, t_peak in zip(r_peaks, Tpeaks):
#         mid_value = (baseline_signal[r_peak] + baseline_signal[Speaks[Tpeaks.index(t_peak)]]) / 2  # Calculate mid value
        
#         if baseline_signal[t_peak] > mid_value: # Compare T peak to mid value
#             mi_counter += 1
    
#     mi_percentage = (mi_counter / len(r_peaks)) * 100
    
#     anterior = []
#     posterior = []
#     if mi_percentage >= 52:
#         for i in range(len(r_peaks)-1):
#             area = trapz(baseline_signal[r_peaks[i]:Tpeaks[i]+1])
#             if round(area, 2) < 12:
#                 anterior.append(round(area,2))
#             elif  12 <= round(area, 2) <= 30:
#                 posterior.append(round(area, 2))

    
#     anterior_per = (len(anterior)/len(r_peaks))*100
#     posterior_per = (len(posterior)/len(r_peaks))*100
#     if posterior_per >= 70:
#         label = "Posterior STEMI"
#     elif anterior_per >= 70:
#         label = "Anterior STEMI"
#     else:
#         label = "Abnormal"
#     return label


def prediction_model(image_path, target_shape=[224, 224], class_name=True):
    with results_lock:
        classes = ['LBBB', 'Noise', 'Normal', 'PVC', 'RBBB']
        image = tf.io.read_file(image_path)
        input_arr = tf.image.decode_jpeg(image, channels=3)
        input_arr = tf.image.resize(input_arr, size=target_shape, method=tf.image.ResizeMethod.BILINEAR)
        input_arr = tf.expand_dims(input_arr, axis=0)

        # Set the input tensor
        interpreterss.set_tensor(input_detailss[0]['index'], input_arr)
        
        # Perform inference
        interpreterss.invoke()
        # Get the output tensor
        output_data = interpreterss.get_tensor(output_detailss[0]['index'])

    if class_name:
        idx = np.argmax(output_data[0])
        return output_data[0], classes[idx]
    else:
        return output_data[0]


# def resampled_ecg_data(ecg_signal, original_freq, desire_freq):
#     original_time = np.arange(len(ecg_signal)) / original_freq
#     new_time = np.linspace(original_time[0], original_time[-1], int(len(ecg_signal) * (desire_freq/original_freq)))
#     interp_func = interp1d(original_time, ecg_signal, kind='linear')
#     scaled_ecg_data = interp_func(new_time)
#     return scaled_ecg_data

# def image_array_news_vfib(signal):
#     scales = np.arange(1, 50, 1)
#     coef, freqs = pywt.cwt(signal, scales, 'mexh')
#     abs_coef = np.abs(coef)
#     y_scale = abs_coef.shape[0] / 224
#     x_scale = abs_coef.shape[1] / 224
#     x_indices = np.arange(224) * x_scale
#     y_indices = np.arange(224) * y_scale
#     x, y = np.meshgrid(x_indices, y_indices, indexing='ij')
#     x = x.astype(int)
#     y = y.astype(int)
#     rescaled_coef = abs_coef[y, x]
#     min_val = np.min(rescaled_coef)
#     max_val = np.max(rescaled_coef)
#     normalized_coef = (rescaled_coef - min_val) / (max_val - min_val)
#     cmap_indices = (normalized_coef * 256).astype(np.uint8)
#     cmap = colormaps.get_cmap('viridis')
#     rgb_values = cmap(cmap_indices)
#     image = rgb_values.reshape((224, 224, 4))[:, :, :3]
#     denormalized_image = (image * 254) + 1
#     rotated_image = np.rot90(denormalized_image, k=1, axes=(1, 0))
#     return rotated_image.astype(np.uint8)

# def vfib_model_pred_tfite(raw_signal, model, fs):
# ##    if fs == 200 and (np.max(raw_signal) > 4.1 or np.min(raw_signal) < 0):
#     raw_signal = MinMaxScaler(feature_range=(0,4)).fit_transform(raw_signal.reshape(-1,1)).squeeze()
#     seconds = 2.5
#     steps_data = int(fs*seconds)
#     total_data = raw_signal.shape[0]
#     start = 0
#     normal, vfib_vflutter, asys, noise = [], [], [], []
#     percentage = {'NORMAL':0, 'VFIB-VFLUTTER':0, 'ASYS':0, 'NOISE':0}
#     model_prediction = []
#     while start < total_data:
#         end = start+steps_data
#         if end - start == steps_data and end < total_data:
#             _raw_s_ = raw_signal[start:end]
#             if _raw_s_.any() :
#                 raw = image_array_news_vfib(_raw_s_)
#             else:
#                 raw = np.array([])
#         else:
#             _raw_s_ = raw_signal[start:end]
#             if _raw_s_.any():
#                 _raw_s_ = raw_signal[-steps_data:total_data]
#                 raw = image_array_news_vfib(_raw_s_)
#                 end = total_data - 1
#             else:
#                 raw = np.array([])
#         if raw.any():
#             raw = raw.astype(np.float32)/255
#             rs_raw = resampled_ecg_data(_raw_s_, fs, 500/seconds)
#             if rs_raw.shape[0] != 500:
#                 rs_raw = signal.resample(rs_raw, 500)
# ##            image_data = (tf.expand_dims(raw, axis=0), tf.constant(rs_raw.reshape(1, -1, 1).astype(np.float32)))
#             # image_data = (tf.cast(image_data[0],dtype=tf.float32), )
#             image_data = (tf.expand_dims(raw, axis=0),)
#             model_pred = predict_tflite_model(model, image_data)[0]
#             label = np.argmax(model_pred)
#             model_prediction.append(f'{(start, end)}={model_pred}')
#             if label == 0: normal.append(((start, end), model_pred)); percentage['NORMAL'] += (end-start)/total_data
#             elif label == 1: vfib_vflutter.append(((start, end), model_pred)); percentage['VFIB-VFLUTTER'] += (end-start)/total_data
#             elif label == 2: asys.append(((start, end), model_pred)); percentage['ASYS'] += (end-start)/total_data
#             else: noise.append(((start, end), model_pred)); percentage['NOISE'] += (end-start)/total_data
#         start = start+steps_data
    
#     return normal, vfib_vflutter, asys, noise, model_prediction, percentage

# ##@time_wrapper
# def vfib_model_check_new(ecg_signal, model, fs):
#     normal, vfib_vflutter, asys, noise, model_prediction, percentage = vfib_model_pred_tfite(ecg_signal, model, fs)
    
#     final_label_index = np.argmax([percentage['NORMAL'], percentage['VFIB-VFLUTTER'],
#                              percentage['ASYS'], percentage['NOISE']])
    
#     if final_label_index == 0 and percentage['NORMAL'] > .50:
#         final_label = 'Normal'
#         percentage = percentage['NORMAL']

#     elif final_label_index == 0 and (percentage['VFIB-VFLUTTER'] < 0.3 and percentage['ASYS'] < 0.3 and percentage['NOISE'] < 0.3):
#         final_label = 'Normal'
#         percentage = percentage['NORMAL']

#     else:
#         final_label_index = np.argmax([percentage['VFIB-VFLUTTER'],
#                                 percentage['ASYS'], percentage['NOISE']])
#         if final_label_index == 0:
#             final_label = 'VFIB/Vflutter'
#             percentage = percentage['VFIB-VFLUTTER']
#         elif final_label_index == 1:
#             final_label = 'ASYS'
#             percentage = percentage['ASYS']
#         else:
#             final_label = 'Noise'
#             percentage = percentage['NOISE']
        
#     return final_label, percentage, (normal, vfib_vflutter, asys, noise, model_prediction)

def prediction_model_vfib_vfl(input_arr, vfib_vfl_model):
    classes = ['VFIB', 'asystole', 'noise', 'normal']
    input_arr = tf.io.decode_jpeg(tf.io.read_file(input_arr), channels=3)
    input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
    input_arr = (tf.expand_dims(input_arr, axis=0),)
    model_pred = predict_tflite_model(vfib_vfl_model, input_arr )[0]
    idx = np.argmax(model_pred)
    return model_pred, classes[idx]

def extract_number(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else float('inf')

def check_vfib_vfl_model(ecg_signal, vfib_vfl_model):
    baseline_signal = baseline_construction_200(ecg_signal)
    low_ecg_signal= lowpass(baseline_signal, cutoff=0.2)
    label = 'Abnormal'
    for img_path in glob.glob('vflutter_img/*.jpg'):
        os.remove(img_path)

    total_data = len(low_ecg_signal)
    num_phases = 2 if total_data > 1000 else 1  
    while total_data % num_phases != 0:
        num_phases += 1  

    step_size = total_data // num_phases 

    fi = 0
    while fi < total_data:
        temp_img = low_ecg_signal[fi: fi + step_size]
        fi += step_size
        plt.figure()
        plt.plot(temp_img)
        plt.axis("off")
        img_path = f"vflutter_img/p_{fi}.jpg"
        plt.savefig(img_path)
        plt.close()
        aq = cv2.imread(img_path)
        aq = cv2.resize(aq, (640, 480))
        cv2.imwrite(img_path, aq)

    combine_result = []
    label = 'Abnormal'

    files = sorted(glob.glob("vflutter_img/*.jpg"), key=extract_number)
    for vfib_file in files:
        with tf.device("CPU"):
            predictions, ids = prediction_model_vfib_vfl(vfib_file, vfib_vfl_model)
        #print(predictions, ids)
        label = "Abnormal" #"Normal"
        if str(ids) == "VFIB" and float(predictions[0]) > 0.75:
            label = "VFIB/Vflutter"
            combine_result.append(label)
        if str(ids) == "asystole" and float(predictions[1]) > 0.75:
            label = "ASYS"
            combine_result.append(label)
       
        if str(ids) == "noise" and float(predictions[2]) > 0.75:
            label = "Noise"
            combine_result.append(label)
       
        if str(ids) == "normal" and float(predictions[3]) > 0.75:
            label = "Normal"
            combine_result.append(label)

    temp_label = list(set(combine_result)) 
    if len(temp_label) > 1:
        label='Abnormal'
        if 'ASYS' in temp_label:
            label = 'ASYS'
        elif 'Noise' in temp_label:
            label = 'Noise'
    else:
        label = temp_label[0]
    return label

def image_array_new(signal):
    scales = np.arange(1, 25, 1)
    coef, freqs = pywt.cwt(signal, scales, 'gaus6')
    abs_coef = np.abs(coef)
    y_scale = abs_coef.shape[0] / 224
    x_scale = abs_coef.shape[1] / 224
    x_indices = np.arange(224) * x_scale
    y_indices = np.arange(224) * y_scale
    x, y = np.meshgrid(x_indices, y_indices, indexing='ij')
    x = x.astype(int)
    y = y.astype(int)
    rescaled_coef = abs_coef[y, x]
    min_val = np.min(rescaled_coef)
    max_val = np.max(rescaled_coef)
    normalized_coef = (rescaled_coef - min_val) / (max_val - min_val)
    cmap_indices = (normalized_coef * 256).astype(np.uint8)
    cmap = colormaps.get_cmap('viridis')
    rgb_values = cmap(cmap_indices)
    image = rgb_values.reshape((224, 224, 4))[:, :, :3]
    denormalized_image = (image * 254) + 1
    rotated_image = np.rot90(denormalized_image, k=1, axes=(1, 0))
    return rotated_image.astype(np.uint8)

def diff(x,y):
    pac=[]
    count = 0
    for i in y:
        count=0
        for j in x:
            temp = abs(i-j)
            if temp<=22:
                count=1
        if count==1:
            pac.append(1)
        else:
            pac.append(0)
                
    return pac

def diffoff(x,y):
    pac=[]
    count = 0
    for i in y:
        count=0
        for j in x:
            if i==math.nan:
                pass
            else:
                temp = abs(i-j)
            if temp<=30:#30
                count=1
        if count==1:
            pac.append(1)
        else:
            pac.append(0)
                
    return pac

def band_pass_filter(signal):
    result = None
    sig = signal.copy()
    for index in range(len(signal)):
      sig[index] = signal[index]

      if (index >= 1):
        sig[index] += 2*sig[index-1]

      if (index >= 2):
        sig[index] -= sig[index-2]

      if (index >= 6):
        sig[index] -= 2*signal[index-6]

      if (index >= 12):
        sig[index] += signal[index-12] 
	
    result = sig.copy()

    for index in range(len(signal)):
      result[index] = -1*sig[index]

      if (index >= 1):
        result[index] -= result[index-1]

      if (index >= 16):
        result[index] += 32*sig[index-16]

      if (index >= 32):
        result[index] += sig[index-32]
    max_val = max(max(result),-min(result))
    result = result/max_val

    return result


def get_median_filter_width(sampling_rate, duration):
    res = int( sampling_rate*duration )
    res += ((res%2) - 1) # needs to be an odd number
    return res

BASIC_SRATE= 500
# baseline fitting by filtering
# === Define Filtering Params for Baseline fitting Leads======================
ms_flt_array = [0.2,0.6]    #<-- length of baseline fitting filters (in seconds)
mfa = np.zeros(len(ms_flt_array), dtype='int')
for i in range(0, len(ms_flt_array)):
    mfa[i] = get_median_filter_width(BASIC_SRATE, ms_flt_array[i])

def filter_signal(X):
    global mfa
    X0 = X  #read orignal signal
    for mi in range(0,len(mfa)):
        X0 = medfilt (X0,mfa[mi]) # apply median filter one by one on top of each other
    X0 = np.subtract(X,X0)  # finally subtract from orignal signal
    return X0

def denoise_signal(X, dwt_transform, dlevels, cutoff_low, cutoff_high):
    coeffs = wavedec(X, dwt_transform, level=dlevels)   # wavelet transform 'bior4.4'
    # scale 0 to cutoff_low 
    for ca in range(0,cutoff_low):
        coeffs[ca]=np.multiply(coeffs[ca],[0.0])
    # scale cutoff_high to end
    for ca in range(cutoff_high, len(coeffs)):
        coeffs[ca]=np.multiply(coeffs[ca],[0.0])
    Y = pywt.waverec(coeffs, dwt_transform) # inverse wavelet transform
    return Y 

# def detect_beats(
#         baseline_signal,  # The raw ECG signal
#         fs,  # Sampling fs in HZ
#         # Window size in seconds to use for
#         ransac_window_size=3.0, #5.0
#         # Low frequency of the band pass filter
#         lowfreq=5.0,
#         # High frequency of the band pass filter
#         highfreq=10.0,
# ):
#     ransac_window_size = int(ransac_window_size * fs)

#     lowpass = scipy.signal.butter(1, highfreq / (fs / 2.0), 'low')
#     highpass = scipy.signal.butter(1, lowfreq / (fs / 2.0), 'high')
#     # TODO: Could use an actual bandpass filter
#     ecg_low = scipy.signal.filtfilt(*lowpass, x=baseline_signal)
#     ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)

#     # Square (=signal power) of the first difference of the signal
#     decg = np.diff(ecg_band)
#     decg_power = decg ** 2

#     # Robust threshold and normalizator estimation
#     thresholds = []
#     max_powers = []
#     for i in range(int(len(decg_power) / ransac_window_size)):
#         sample = slice(i * ransac_window_size, (i + 1) * ransac_window_size)
#         d = decg_power[sample]
#         thresholds.append(0.5 * np.std(d))
#         max_powers.append(np.max(d))

#     threshold = np.median(thresholds)
#     max_power = np.median(max_powers)
#     decg_power[decg_power < threshold] = 0

#     decg_power /= max_power
#     decg_power[decg_power > 1.0] = 1.0
#     square_decg_power = decg_power ** 4
#     # square_decg_power = decg_power**4

#     shannon_energy = -square_decg_power * np.log(square_decg_power)
#     shannon_energy[~np.isfinite(shannon_energy)] = 0.0

#     mean_window_len = int(fs * 0.125 + 1)
#     lp_energy = np.convolve(shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode='same')
#     # lp_energy = scipy.signal.filtfilt(*lowpass2, x=shannon_energy)

#     lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, fs / 14.0) # 20.0
#     # lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, fs/8.0)
#     lp_energy_diff = np.diff(lp_energy)

#     zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
#     zero_crossings = np.flatnonzero(zero_crossings)
#     zero_crossings -= 1
#     return zero_crossings
    
def detect_beats(ecg, rate, ransac_window_size=3.35, lowfreq=5.0, highfreq=15.0):
    ransac_window_size = int(ransac_window_size * rate)
    lowpass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
    highpass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
    ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
    ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)
    decg = np.diff(ecg_band)
    decg_power = decg ** 2
    thresholds, max_powers = [], []
    for i in range(int(len(decg_power) / ransac_window_size)):
        sample = slice(i * ransac_window_size, (i + 1) * ransac_window_size)
        d = decg_power[sample]
        thresholds.append(0.5 * np.std(d))
        max_powers.append(np.max(d))
    threshold = np.median(thresholds)
    max_power = np.median(max_powers)
    decg_power[decg_power < threshold] = 0
    decg_power /= max_power
    decg_power[decg_power > 1.0] = 1.0
    square_decg_power = decg_power ** 4
    shannon_energy = -square_decg_power * np.log(square_decg_power)
    shannon_energy[~np.isfinite(shannon_energy)] = 0.0
    mean_window_len = int(rate * 0.125 + 1)
    lp_energy = np.convolve(shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode='same')
    lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate / 14.0)
    lp_energy_diff = np.diff(lp_energy)
    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    zero_crossings = np.flatnonzero(zero_crossings)
    zero_crossings -= 1

    rpeaks = []
    for idx in zero_crossings:
        search_window = slice(max(0, idx - int(rate * 0.2)), min(len(ecg), idx + int(rate * 0.1)))
        local_signal = ecg[search_window]
        max_amplitude = np.max(local_signal)
        min_amplitude = np.min(local_signal)

        if abs(max_amplitude) > abs(min_amplitude):  
            rpeak = np.argmax(local_signal) + search_window.start
        elif abs(max_amplitude+0.11) < abs(min_amplitude):  
            rpeak = np.argmin(local_signal) + search_window.start
        else:  
            if max_amplitude >= 0:
                rpeak = np.argmax(local_signal) + search_window.start
            else:
                rpeak = np.argmin(local_signal) + search_window.start
 
        rpeaks.append(rpeak)
    return np.array(rpeaks)



def detect_beats1(
		ecg,	# The raw ECG signal
		rate,	# Sampling rate in HZ
		# Window size in seconds to use for 
		ransac_window_size=2.0,
		# Low frequency of the band pass filter
		lowfreq=5.0,
		# High frequency of the band pass filter
		highfreq=8.0,
		):
	ransac_window_size = int(ransac_window_size*rate)

	lowpass = scipy.signal.butter(1, highfreq/(rate/2.0), 'low')
	highpass = scipy.signal.butter(1, lowfreq/(rate/2.0), 'high')
	# TODO: Could use an actual bandpass filter
	ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
	ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)
	
	# Square (=signal power) of the first difference of the signal
	decg = np.diff(ecg_band)
	decg_power = decg**2
	
	# Robust threshold and normalizator estimation
	thresholds = []
	max_powers = []
	for i in range(int(len(decg_power)/ransac_window_size)):
		sample = slice(i*ransac_window_size, (i+1)*ransac_window_size)
		d = decg_power[sample]
		thresholds.append(0.5*np.std(d))
		max_powers.append(np.max(d))

	threshold = np.median(thresholds)
	max_power = np.median(max_powers)
	decg_power[decg_power < threshold] = 0

	decg_power /= max_power
	decg_power[decg_power > 1.0] = 1.0
	square_decg_power = decg_power**4
	#square_decg_power = decg_power**4

	shannon_energy = -square_decg_power*np.log(square_decg_power)
	shannon_energy[~np.isfinite(shannon_energy)] = 0.0

	mean_window_len = int(rate*0.125+1)
	lp_energy = np.convolve(shannon_energy, [1.0/mean_window_len]*mean_window_len, mode='same')
	#lp_energy = scipy.signal.filtfilt(*lowpass2, x=shannon_energy)
	
	lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate/20.0)
	#lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate/8.0)
	lp_energy_diff = np.diff(lp_energy)

	zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
	zero_crossings = np.flatnonzero(zero_crossings)
	zero_crossings -= 1
	return zero_crossings

def SACompare(list1, val):
    l=[]
    for x in list1:
        if x>=val:
            l.append(1)
        else:
            l.append(0)
    if 1 in l:
        return True
    else:
        return False

def SACompareShort(list1, val1,val2):
    l=[]
    for x in list1:
        if x>=val1 and x<=val2:
            l.append(1)
        else:
            l.append(0)
    if 1 in l:
        return True
    else:
        return False



def connect_mqtt() -> mqtt_client:



    def on_connect(client, userdata, flags, rc,protocol):
        global topic_x
        if rc == 0:
            print("Connected to MQTT")
##            client.publish("oom/ecg/aiServer",True,2,True)
##            client.subscribe(topic_x,qos=2)
            
            
            
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id,protocol=mqtt_client.MQTTv5)
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    client.tls_set_context(context)
    client.enable_shared_subscription = True
    client.username_pw_set(username, password)
##    client.will_set("oom/ecg/aiServer", False,2,False)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client



# Baseline shifting and wandering 

def moving_Average(file):
    window_size = 10    
    numbers_series = pd.Series(file)    
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()    
    moving_averages_list = moving_averages.tolist()
    final = moving_averages_list[window_size - 1:]
    final_list = np.array(final)
    return final_list

def baseline_reconstruction(data, kernel_Size):
    s_corrected = signal.detrend(data)
    baseline_corrected = s_corrected - medfilt(s_corrected,kernel_Size)
    return baseline_corrected


def baseline_beads(file):

    fc = 0.0075
    d = 1
    r = 6
    amp = 0.8
    lam0 = 0.5 * amp
    lam1 = 5 * amp
    lam2 = 4 * amp
    Nit = 15
    pen = 'L1_v2'
    signal_est, bg_est, cost = be.beads(file, d, fc, r, Nit, lam0, lam1, lam2, pen, conv=None)

    x = file-signal_est-bg_est

    return x

def baseline_detection(data):  
    df = signal.detrend(data)
    signal_thread = moving_Average(df)
    baseline = baseline_reconstruction(df, kernel_Size = 81)
    thread_corrected = moving_Average(baseline)

    diff =  thread_corrected - signal_thread

    diff_max = max(diff)
    diff_min = min(diff)
    diff_pvar = pvariance(diff)

    if (diff_pvar>0.0029 and (diff_min<=(-0.79) or 0.68<diff_max<0.72)) or ((-0.63)<diff_min<(-0.3) and 0.0093<diff_pvar<0.05):

        baseline_51 = baseline_reconstruction(data, kernel_Size=51)
        thread = moving_Average(baseline_51)
        baseline_decision = thread - signal_thread
        baseline_decision_max = max(baseline_decision) 
        baseline_decision_min = min(baseline_decision)
        baseline_decision_pvar = pvariance(baseline_decision)

        
        if ((1.3<baseline_decision_max<1.95 or baseline_decision_max>2) and (-1)>baseline_decision_min):
            label = "High Baseline"
        elif (1.05<baseline_decision_max<1.2 and (-0.79)<baseline_decision_min<(-0.67) and 0.039<baseline_decision_pvar<0.09):
            label = "Pybeads"
            base_beads = baseline_beads(df)
        else:
            label = " Normal Baseline"


    elif (0.174<diff_max<0.41 and -0.294>diff_min>-0.95 and 0.0052<diff_pvar<0.0089):
        label = "Pybeads"
        base_beads = baseline_beads(df)
    else:
        label = "Normal"
    
    
    if label == "High Baseline":
        return baseline_51
    
    elif label == "Pybeads":
        return base_beads
    
    else:
        return baseline


def normalize(signal, feature_range= (0,1)):
  scaler = MinMaxScaler(feature_range=feature_range)
  if signal.ndim==2: 
    a = scaler.fit_transform(signal)
    return a.reshape(len(a),)
  a = scaler.fit_transform(signal.reshape(-1,1))
  return a.reshape(len(a),)

def fft_clear_signal(signal, dt=1, low_thr=0., high_thr=0.5, power=True):
  n = len(signal)
  fhat = np.fft.fft(signal, n)
  PSD = fhat * np.conj(fhat)/n
  freq = (1/(dt*n))*np.arange(n)
  L = np.arange(1, np.floor(n/2), dtype='int')
  if power:
    indices = PSD > high_thr
  else:
    indices = np.where(((low_thr <= freq) & (freq <= high_thr)), True, False)
  
  fhat = indices * fhat
  ffilt = np.fft.ifft(fhat)
  return ffilt.real

def lowpass_11(file):
  b, a = signal.butter(3, 0.3, btype='lowpass', analog=False)
  low_passed = signal.filtfilt(b, a, file)
  return low_passed

def lowpass_1(file):
  b, a = signal.butter(3, 0.2, btype='lowpass', analog=False)
  low_passed = signal.filtfilt(b, a, file)
  return low_passed

def lowpass_2(file):
  b, a = signal.butter(3, 0.2, btype='lowpass', analog=False)
  low_passed = signal.filtfilt(b, a, file)
  return low_passed

def noise_final(ecg_signal):
    frequency = 200
    threshold_pos = 0.535
    threshold_neg = -0.7 
    gap = 20
    max_peak_diff = frequency * 0.28

    apply_scalar = MinMaxScaler(feature_range=(0, 2)).fit_transform(ecg_signal.reshape(-1, 1)).squeeze() #2
    if frequency == 200:
        bes = baseline_reconstruction(apply_scalar, kernel_Size=101)
        max_peak_diff= 50

    low = lowpass_2(bes)

    above_positive_threshold = argrelextrema(low, np.greater, order=1)[0]
    below_negative_threshold = argrelextrema(low, np.less, order=1)[0]

    # Filter out the indices of peaks that pass through the thresholds
    peaks_above_positive_threshold = [idx for idx in above_positive_threshold if low[idx] > threshold_pos]
    peaks_below_negative_threshold = [idx for idx in below_negative_threshold if low[idx] < threshold_neg]

    # Check if any elements fall within 15 indices and consider only the maximum point from nearby detected peaks
    if len(peaks_above_positive_threshold) > 1:
        peaks_pos = []
        last_idx = peaks_above_positive_threshold[0]
        for idx in peaks_above_positive_threshold[1:]:
            if idx - last_idx > gap:
                peaks_pos.append(last_idx)
                last_idx = idx
            elif low[idx] > low[last_idx]:
                last_idx = idx
        peaks_pos.append(last_idx)
    else:
        peaks_pos = peaks_above_positive_threshold

    if len(peaks_below_negative_threshold) > 1:
        peaks_neg = []
        last_idx = peaks_below_negative_threshold[0]
        for idx in peaks_below_negative_threshold[1:]:
            if idx - last_idx > gap:
                peaks_neg.append(last_idx)
                last_idx = idx
            elif low[idx] < low[last_idx]:
                last_idx = idx
        peaks_neg.append(last_idx)
    else:
        peaks_neg = peaks_below_negative_threshold

    # Get the values of the peak points
    peak_values_pos = low[peaks_pos]
    peak_values_neg = low[peaks_neg]

    min_noise_count = 4
    pos_diff = np.diff(peaks_pos)
    neg_diff = np.diff(peaks_neg)
    count_pos = len([num for num in pos_diff if num < max_peak_diff])
    count_neg = len([num for num in neg_diff if num < max_peak_diff])

    if len(peaks_pos) < 5 and len(peaks_neg) < 5:
        min_noise_count = 1

    r_index = detect_beats(bes, frequency)
    diff_r_peak = np.diff(r_index)
        
    filtered_list = [num for num in diff_r_peak if num < max_peak_diff]

    if (len(peaks_pos) == 0 and len(peaks_neg) == 0):
        result = "high_noise"
    elif (len(peaks_pos) != 0 or len(peaks_neg)  != 0):
        list_temp = min(len(peaks_pos), len(peaks_neg)) / max(len(peaks_pos), len(peaks_neg))
        if count_pos >= 7 or count_neg >= 7:
            result = "high_noise"
        elif len(filtered_list) > 10:
            result = 'high_noise'
        elif 0.85 > list_temp > 0.13 or list_temp == 0:
            if count_pos >= min_noise_count or count_neg >= min_noise_count:
                result = "high_noise"
            else:
                result = "Normal"
        else:
            result = "Normal"
    else:
        result = "Normal"

    return result


def prediction_model_noise(input_arr):
    with results_lock:
        classes = ['Noise', 'Normal']
        input_arr = tf.cast(input_arr, dtype=tf.float32)
        input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
        input_arr = tf.expand_dims(input_arr, axis=0)
        interpreter_noise.set_tensor(input_details_noise[0]['index'], input_arr)
        interpreter_noise.invoke()
        output_data = interpreter_noise.get_tensor(output_details_noise[0]['index'])
        idx = np.argmax(output_data[0])
        return output_data[0], classes[idx]

def plot_to_imagearray_noise(ecg_signal):
    fig, ax = plt.subplots(num=1, clear=True)
    ax.plot(ecg_signal, color='black')
    ax.axis(False)
##    random_letters = ''.join(random.choices(string.ascii_letters, k=12))
##    plt.savefig("images\\"+random_letters+".jpg")
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data1 = data[:, :, ::-1]
    plt.close(fig)
    return data1

def check_noise_model(ecg_signal, frequency):
    steps_data = int(frequency * 2.5)
    total_data = ecg_signal.shape[0]
    start = 0
    baseline, non_ecg = [], []
    percentage = {'Normal': 0, 'Noise': 0, 'total_slice': 0}
    while start < total_data:
        end = start + steps_data
        if end - start == steps_data and end < total_data:
            image_data = plot_to_imagearray_noise(ecg_signal[start:end])
        else:
            image_data = plot_to_imagearray_noise(ecg_signal[-steps_data:total_data])
            end = total_data - 1
        output_data, class_name = prediction_model_noise(image_data)
        if class_name == 'Normal':
            baseline.append(((start, end), output_data))
            # percent['Normal'] += (end - start) / total_data
            percentage['Normal'] += 1
        else:
            non_ecg.append(((start, end), output_data))
            # percent['Noise'] += (end - start) / total_data
            percentage['Noise'] += 1
        start += steps_data
    noise_label = 'Normal'
    if percentage['total_slice'] != 0:
        if percentage['Noise'] == percentage['total_slice'] and percentage['total_slice'] > 0.5:
            noise_label = 'high_noise'
        elif percentage['Noise']/percentage['total_slice']  >= 0.6:
            noise_label = 'high_noise'
    return noise_label

def noise_engine(flag,ecgdata):
    global ref_file
    noise_label = 'Normal'
    if flag == "200":
        ecg_signal = np.array(ecgdata["ECG"])
        noise_label = check_noise_model(ecg_signal, 200)

    return noise_label

##def noise_engine(flag,ecgdata):
##    global ref_file
##    if flag == "200":
##        final_label = "Normal"
####        ecg_signal = pd.read_csv("b.csv")["ECG"]
##        ecg_signal = ecgdata["ECG"]
##        val = 0
##        initial_file = ecg_signal
##        initial_600_data  = ecg_signal.iloc[0:600]
##        final_600_data = ecg_signal.iloc[-601:-1]
##        # check for non-ecg
##        initial_files = lowpass_2(ecg_signal)
##        s_norm = normalize(initial_files)
##        gradient = np.gradient(initial_file.values)
##        gradient_pVAR = (pvariance(gradient)/pvariance(initial_file.values))
##        gradient_std = np.std(gradient)
##        std = np.std(initial_file)
##        kurtosis = stats.kurtosis(initial_file)
##        # print(gradient_std, gradient_pVAR, std)
##        clean_signal = fft_clear_signal(s_norm, low_thr=0, high_thr=0.15, power=False)
##        cl_norm = normalize(clean_signal)
##
##        f, psd = welch(initial_file, fs=200, nperseg=1024)
##        spectral_centroid = np.sum(f * psd) / np.sum(psd)
##        spectral_bandwidth = np.sqrt(np.sum(psd * (f - spectral_centroid) ** 2) / np.sum(psd))
##        spectral_skewness = np.sum(psd * (f - spectral_centroid) ** 3) / np.sum(psd) / spectral_bandwidth ** 3
##        spectral_kurtosis = (np.sum(psd**4) / len(psd)) / (psd.var()**2) - 2*(stats.kurtosis(psd) - 3)
##        signal_diff = np.diff(initial_file)
##        zero_crossings = len(np.where(np.diff(np.sign(initial_file)))[0])
####        final_label = "Normal"
##        print(gradient_std, gradient_pVAR, std)

        
    # conditions for 200 frequency
        
##        if len(initial_file) == 2000:
##            if (gradient_pVAR<0.068 and std>1.1 )or std <0.001:
##                final_label = "Non_ECG"
####                print("OK1")
##            elif (gradient_std<0.07 or gradient_std>0.33) and (std <0.008 or std>0.7):
##                    final_label = "Non_ECG"
####                    print("OK2")
##            elif gradient_pVAR> 0.75:
##                final_label = "Non_ECG"
####                print("OK3")
##            elif(gradient_pVAR>0.58 and gradient_pVAR<0.7)and gradient_std>0.35 and std>0.44:
##                final_label = "Non_ECG"
####                print("OK4")
##            elif gradient_pVAR<0.125 and (std>0.74 and std<=1.1):
##                final_label = "high_noise"
##            elif kurtosis < 0.03 and spectral_skewness < 4 and spectral_centroid > 4 and std>0.7:
##                final_label ="high_noise"
##            elif gradient_std > 0.05 and spectral_kurtosis<(-270) and gradient_pVAR>0.5 : 
##                final_label="Non_ECG"
####                print("OK5")
##            else:
##                final_label = "Normal"
##        else:
##            if (gradient_pVAR<0.068 and std>1.1 )or std <0.001:
##                final_label = "Non_ECG"        
##            elif (gradient_std<0.07 or gradient_std>0.37) and (std <0.008 or std>0.71):
##                final_label = "Non_ECG"        
##            elif gradient_pVAR> 0.8:
##                final_label = "Non_ECG"        
##            elif (gradient_pVAR>0.58 and gradient_pVAR<0.7) and gradient_std>0.34 and std>0.44:
##                final_label = "Non_ECG"        
##            elif gradient_pVAR<0.11 and (std>0.79 and std<=1.1):
##                final_label = "high_noise"
##            elif kurtosis < 0.03 and spectral_skewness < 4 and spectral_centroid > 4 and std>0.7:
##                final_label ="high_noise"
##            else:
##                final_label = "Normal"

        
##        final_label = "Normal"
##        if final_label == "Normal":
##            std_initial_600 = np.std(initial_600_data)
##            std_final_600 = np.std(final_600_data)
##            if std_final_600 < 0.008 or std_initial_600<0.008:
##                final_label = "high_noise"
##                val = 1
##            elif len(initial_file)>100:
##                max_val = max(initial_file)
##                min_val = min(initial_file)
##                start = 0
##                end =500
##                while True:
##                    data_500 = initial_file.iloc[start:end]
##                    extreme_max_count = 0
##                    extreme_min_count = 0
##                    for l in data_500:
##                        
##                        if max_val - 0.009 <= l :
##                            extreme_max_count+=1
##
##                        elif min_val +0.009>= l:
##                            extreme_min_count +=1
##                        else: 
##                            pass
##                    if extreme_max_count>200 or extreme_min_count>200:
##                        final_label = "high_noise"
##                        break
##                    else:
##                        final_label = "Normal"
##                    start+=500
##                    end+=500
##                    if end>len(initial_file):
##                        break
##
##
##
##        if final_label == "Normal":
##
##            min_voltage = np.min(initial_file)
##            max_voltage = np.max(initial_file)
##            vol_range = max_voltage - min_voltage
##            kurtosis = stats.kurtosis(initial_file)
##
##
##        # Calculate time-domain properties
##
##            signal_diff = np.diff(initial_file)
##            zero_crossings = len(np.where(np.diff(np.sign(initial_file)))[0])
##
##            if vol_range > 0.4 and  vol_range< 1 and zero_crossings ==0 and kurtosis<1.5 and (std>0.09 and std<0.1):
##                final_label = "Non_ECG"
##            elif min_voltage> 0 and zero_crossings ==0:
##                final_label = "Normal"
##            elif min_voltage ==0 and zero_crossings <=8:
##                final_label = "Normal"
##            elif min_voltage ==0 and zero_crossings> 70:
##                final_label = "Non_ECG"
##            else:
##                final_label = "Normal"
##                
####        if val == 0 or val == 1:
####            final_label = noise_final(initial_files)
##
##        
##        if final_label == "Normal":
##            baseline = baseline_detection(initial_file)
##            out = pvariance(ref_file)/pvariance(initial_file)
##            final_label = "high_noise" if out < 0.1 else "Normal"
##            base1 = signal.detrend(ecg_signal)
##            medfilt1 = signal.medfilt(base1,171)
##            outputss = np.where(abs(medfilt1)>0.60, 1, 0)
##            final_label = 'high_noise_recovery' if outputss.any() else 'Normal'
##            if final_label == "high_noise_recovery":
##                return "high_noise_recovery",baseline
##            else:
##                return baseline
####        print(final_label)
##        if final_label == "Non_ECG":
##            return None
##        elif final_label == "high_noise":
##            return  "high_noise"
####        else:
####            return baseline





def unique(list1):
 
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    return unique_list

def QRS_detection(signal,sample_rate,max_bpm):

    ## Stationary Wavelet Transform
    coeffs = pw.swt(signal, wavelet = "haar", level=2, start_level=0, axis=-1)
    d2 = coeffs[1][1] ##2nd level detail coefficients
    
    
    ## Threhold the detail coefficients
    avg = np.mean(d2)
    std = np.std(d2)
    sig_thres = [abs(i) if abs(i)>2.0*std else 0 for i in d2-avg]
    
    ## Find the maximum modulus in each window
    window = int((60.0/max_bpm)*sample_rate)
    sig_len = len(signal)
    n_windows = int(sig_len/window)
    modulus,qrs = [],[]
    
    ##Loop through windows and find max modulus
    for i in range(n_windows):
        start = i*window
        end = min([(i+1)*window,sig_len])
        mx = max(sig_thres[start:end])
        if mx>0:
            modulus.append( (start + np.argmax(sig_thres[start:end]),mx))
    
    
    ## Merge if within max bpm
    merge_width = int((0.2)*sample_rate)
    i=0
    while i < len(modulus)-1:
        ann = modulus[i][0]
        if modulus[i+1][0]-modulus[i][0] < merge_width:
            if modulus[i+1][1]>modulus[i][1]: # Take larger modulus
                ann = modulus[i+1][0]
            i+=1
                
        qrs.append(ann)
        i+=1 
    ## Pin point exact qrs peak
    window_check = int(sample_rate/6)
    #signal_normed = np.absolute((signal-np.mean(signal))/(max(signal)-min(signal)))
    r_peaks = [0]*len(qrs)
    
    for i,loc in enumerate(qrs):
        start = max(0,loc-window_check)
        end = min(sig_len,loc+window_check)
        wdw = np.absolute(signal[start:end] - np.mean(signal[start:end]))
        pk = np.argmax(wdw)        
        r_peaks[i] = start+pk
        
    return r_peaks


def get_percentage_diff(previous, current):
    try:
        percentage = abs(previous - current)/max(previous, current) * 100
    except ZeroDivisionError:
        percentage = float('inf')
    return percentage

def Average(lst):
    return sum(lst) / len(lst)




def indexes(y, thres=0.8, min_dist=80, thres_abs=False):
    """Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.
    thres_abs: boolean
        If True, the thres value will be interpreted as an absolute value, instead of
        a normalized threshold.

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected.
        When using with Pandas DataFrames, iloc should be used to access the values at the returned positions.
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not thres_abs:
        thres = thres * (np.max(y) - np.min(y)) + np.min(y)

    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros, = np.where(dy == 0)

    # check if the signal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])

    if len(zeros):
        # compute first order difference of zero indexes
        zeros_diff = np.diff(zeros)
        # check when zeros are not chained together
        zeros_diff_not_one, = np.add(np.where(zeros_diff != 1), 1)
        # make an array of the chained zero indexes
        zero_plateaus = np.split(zeros, zeros_diff_not_one)

        # fix if leftmost value in dy is zero
        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)

        # fix if rightmost value of dy is zero
        if len(zero_plateaus) and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)

        # for each chain of zero indexes
        for plateau in zero_plateaus:
            median = np.median(plateau)
            # set leftmost values to leftmost non zero values
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            # set rightmost and middle values to rightmost non zero values
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    # find the peaks by using the first order difference
    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0)
        & (np.hstack([0.0, dy]) > 0.0)
        & (np.greater(y, thres))
    )[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks



def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 1))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def lowpass(file):
    b, a = signal.butter(3, 0.2, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, file)
    return low_passed




def baseline_construction_250(ecg_signal, kernel_size=131):
    als_baseline = baseline_als(ecg_signal, 16 ** 5, 0.01)
    s_als = ecg_signal - als_baseline
    s_corrected = signal.detrend(s_als)
    corrected_baseline = s_corrected - signal.medfilt(s_corrected, kernel_size)
    return corrected_baseline


def find_s_indexs(ecg, R_index, d):
    d = int(d) + 1
    s = []
    for i in R_index:
        if i == len(ecg):
            continue
        elif i + d <= len(ecg):
            s_array = ecg[i:i + d]
        else:
            s_array = ecg[i:]
        if ecg[i] > 0:
            s_index = i + np.where(s_array == min(s_array))[0][0]
        else:
            s_index = i + np.where(s_array == max(s_array))[0][0]
        s.append(s_index)
    return s

# def find_s_index(r,file):
#     s = []
#     data= []
#     for d in range (0, len(r)):
#         s_p = file[r[d]:r[d]+30]
#         for k in range (0,len(s_p)):
#             data.append(s_p[k])
#         min_d = min(data)
#         min_id = data.index(min_d)
#         s.append(r[d]+min_id)
#         data.clear()
#     return s


def find_q_indexs(ecg, R_index, d):
    d = int(d) + 1
    q = []
    for i in R_index:
        if i == 0:
            continue
        elif 0 <= i - d:
            q_array = ecg[i - d:i]
        else:
            q_array = ecg[:i]
        if ecg[i] > 0:
            q_index = i - (len(q_array) - np.where(q_array == min(q_array))[0][0])
        else:
            q_index = i - (len(q_array) - np.where(q_array == max(q_array))[0][0])
        q.append(q_index)
    return q


def find_j_indexs(file, s_index):
    j = []
    data = []
    for z in range (0,len(s_index)):
        j_index = file[s_index[z]:s_index[z]+10]
        for k in range (0,len(j_index)):
            data.append(j_index[k])
        max_d = max(data)
        max_id = data.index(max_d)
        j.append(s_index[z]+max_id)
        data.clear()
    return j

def find_p_t(signal, r_index, q_index, j_index):
    def check_array(a, array):
        global C
        C -= 1
        mid_val = a / 4
        arr = np.where(array >= mid_val, array, 0)

        if C < 0:
            try:
                thredsold = np.max(arr) / 4
            except:
                thredsold = mid_val
            if arr.any():
                a = indexes(arr, thredsold, 21) + s0
                return a
            else:
                return np.array([0])
        if arr.any():
            try:
                thredsold = np.max(arr) / 2
            except:
                thredsold = mid_val
            a = indexes(arr, thredsold, 21) + s0
            if len(a) >= 5:
                return a
            else:
                return check_array(mid_val, array)
        else:
            return check_array(mid_val, array)

    pt = []
    p_t = []
    for i in range(1, len(r_index)):
        q0, r0, s0 = q_index[i - 1], r_index[i - 1], j_index[i - 1]
        q1, r1, s1 = q_index[i], r_index[i], j_index[i]
        arr = signal[s0:q1]
        global C
        C = 16
        diff = signal[r0] - signal[s0]
        a = check_array(diff, arr)
        if a.any():
            p_t.append(list(a))
            pt.extend(list(a))
        else:
            p_t.append([0])
            pt.extend([0])
    return pt, p_t



def AFIB(p_t):
    PT = []
    for i in range (len(p_t)):
        PT.append(len(p_t[i]))

    Total_beats = len(PT)
    Normal_Count = PT.count(2)
    afib_count = Total_beats - Normal_Count
    Ratio = (afib_count/Total_beats)*100
##    print("Ratio:",Ratio)
    if Ratio>40:
        return 1
    else:
        return 0




def hr_count(ecg_signal, r_index, fs=200):
    cal_sec = round(ecg_signal.shape[0]/fs)
    if cal_sec != 0:
        hr = round(r_index.shape[0]*60/cal_sec)
        return hr
    return 0




def fir_lowpass_filter(data, cutoff, numtaps=21):
    """A finite impulse response (FIR) lowpass filter to a given data using a
    specified cutoff frequency and number of filter taps.

    Args:
        data (array): The input data to be filtered
        cutoff (float): The cutoff frequency of the lowpass filter, specified in the same units as the
    sampling frequency of the input data. It determines the frequency below which the filter allows
    signals to pass through and above which it attenuates them
        numtaps (int, optional): the number of coefficients (taps) in the FIR filter. Defaults to 21.

    Returns:
        array: The filtered signal 'y' after applying a lowpass filter with a specified cutoff frequency
    and number of filter taps to the input signal 'data'.
    """
    
    b = firwin(numtaps, cutoff)
    y = signal.convolve(data, b, mode="same")
    return y

def find_j_index(signal, s_index, fs=200):
    """The index of the maximum value in a given range of a file and returns a list of
    those indices.

    Args:
        signal (array): ECG signal values
        s_index (list/array): _description_
        fs (int, optional): sampling rate of the ECG signal, defaults to 200 (optional)

    Returns:
        list: Indices (j) where the maximum value is found in a specific range of the input
    ecg_signal (signal) defined by the start indices (s_index).
    """
    j = []
    increment = int(fs*0.05)
    for z in range (0,len(s_index)):
        data = []
        j_index = signal[s_index[z]:s_index[z]+increment]
        for k in range (0,len(j_index)):
            data.append(j_index[k])
        max_d = max(data)
        max_id = data.index(max_d)
        j.append(s_index[z]+max_id)
    return j

def find_s_index(ecg, R_index, d):
    """The S-wave in an ECG signal given the R-wave index and a specified
    distance.

    Args:
        ecg (1D numpy array): The ECG signal
        R_index (list/array): The R peak indices in an ECG signal
        d (_type_): The maximum distance between the R peak and the S peak in the ECG signal

    Returns:
        _type_: The function `find_s_index` returns a list of indices representing the location of the
    S-wave in the ECG signal, given the R-wave indices, the ECG signal, and a parameter `d` that
    determines the search window for the S-wave.
    """
    d = int(d)+1
    s = []
    for i in R_index:
        if i == len(ecg):
            s.append(i)
            continue
        elif i+d<=len(ecg):
            s_array = ecg[i:i+d]
        else:
            s_array = ecg[i:]
        if ecg[i] > 0:
            s_index = i+np.where(s_array == min(s_array))[0][0]
        else:
            s_index = i+np.where(s_array == max(s_array))[0][0]
        s.append(s_index)
    return np.sort(s)

def find_q_index(ecg, R_index, d):
    """The Q wave index in an ECG signal given the R wave index and a specified
    distance.

    Args:
        ecg (array): ECG signal values
        R_index (array/list): R peak indices in the ECG signal
        d (int): The maximum distance (in samples) between the R peak and the Q wave onset that we want to find.

    Returns:
        list: Q-wave indices for each R-wave index in the ECG signal.
    """
    d = int(d) + 1
    q = []
    for i in R_index:
        if i == 0:
            q.append(i)
            continue
        elif 0 <= i - d:
            q_array = ecg[i - d:i]
        else:
            q_array = ecg[:i]
        if ecg[i] > 0:
            q_index = i - (len(q_array) - np.where(q_array == min(q_array))[0][0])
        else:
            q_index = i - (len(q_array) - np.where(q_array == max(q_array))[0][0])
        q.append(q_index)
    return np.sort(q)

def find_new_q_index(ecg, R_index, d):
    q = []
    for i in R_index:
        q_ = []
        if i == 0:
            q.append(i)
            continue
        if ecg[i] > 0:
            c = i
            while c > 0 and ecg[c-1] < ecg[c]:
                c -= 1                  
            if ecg[i] * 0.01 > ecg[c] or ecg[c] < 0 or c == 0:
                if abs(i-c) <= d:
                    q.append(c)
                    continue
                else:
                    q_.append(c)
            while c > 0:
                while c > 0 and ecg[c-1] > ecg[c]:
                    c -= 1
                # q_.append(c)
                while c > 0 and ecg[c-1] < ecg[c]:
                    c -= 1
                if q_ and q_[-1] == c:
                    break
                q_.append(c)
                if ecg[i] * 0.01 > ecg[c] or ecg[c] < 0 or c == 0:
                    break
        else:
            c = i
            while c > 0 and ecg[c-1] > ecg[c]:
                c -= 1
            if ecg[i] * 0.01 < ecg[c] or ecg[c] > 0 or c == 0:
                if abs(i-c) <= d:
                    q.append(c)
                    continue
                else:
                    q_.append(c)
            while c > 0:
                while c > 0 and ecg[c-1] < ecg[c]:
                    c -= 1
                # q_.append(c)
                while c > 0 and ecg[c-1] > ecg[c]:
                    c -= 1
                if q_ and q_[-1] == c:
                    break
                q_.append(c)
                if ecg[i] * 0.01 < ecg[c] or ecg[c] > 0 or c == 0:
                    break
        if q_:
            a = 0
            for _q in q_[::-1]:
                if abs(i-_q) <= d:
                    a = 1
                    q.append(_q)
                    break
            if a == 0:
                q.append(q_[0])
    return np.sort(q)

def find_new_s_index(ecg, R_index, d):
    s = []
    end_index = len(ecg)
    for i in R_index:
        s_ = []
        if i == len(ecg):
            s.append(i)
            continue
        if ecg[i] > 0:
            c = i
            while c+1 < end_index and ecg[c+1] < ecg[c]:
                c += 1
            if ecg[i] * 0.01 > ecg[c] or ecg[c] < 0 or c == end_index-1:
                if abs(i-c) <= d:
                    s.append(c)
                    continue
                else:
                    s_.append(c)
            while c+1 < end_index:
                while c+1 < end_index and ecg[c+1] > ecg[c]:
                    c += 1
                while c+1 < end_index and ecg[c+1] < ecg[c]:
                    c += 1
                if s_ and s_[-1] == c:
                    break
                s_.append(c)
                if ecg[i] * 0.01 > ecg[c] or ecg[c] < 0 or c == end_index-1:
                    break
        else:
            c = i
            while c+1 < end_index and ecg[c+1] > ecg[c]:
                c += 1
            if ecg[i] * 0.01 < ecg[c] or ecg[c] > 0 or c == end_index-1:
                if abs(i-c) <= d:
                    s.append(c)
                    continue
                else:
                    s_.append(c)
            while c < end_index:
                while c+1 < end_index and ecg[c+1] > ecg[c]:
                    c += 1
                while c+1 < end_index and ecg[c+1] < ecg[c]:
                    c += 1
                if s_ and s_[-1] == c:
                    break
                s_.append(c)
                if ecg[i] * 0.01 < ecg[c] or ecg[c] > 0 or c == end_index-1:
                    break
        if s_:
            a = 0
            for _s in s_[::-1]:
                if abs(i-_s) <= d:
                    a = 1
                    s.append(_s)
                    break
            if a == 0:
                s.append(s_[0])
    return np.sort(s)

def find_r_peaks(ecg_signal,fs=200):
    """Finds R-peaks in an ECG signal using the Hamilton segmenter algorithm.

    Args:
        ecg_signal (array): The ECG signal of numpy array
        fs (int, optional): sampling rate of the ECG signal, defaults to 200 (optional)

    Returns:
        list: the R-peak indices of the ECG signal using the Hamilton QRS complex detector algorithm.
    """
##    out = hamilton_segmenter(signal=ecg_signal, sampling_rate=fs)
##    r_index_ = out["rpeaks"]
    r_index_ = detect_beats(ecg_signal, fs)
    return r_index_


def baseline_als(file, lam, p, niter=10):
    L = len(file)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*file)
        w = p * (file > z) + (1-p) * (file < z)
    return z

def lowpass(ecg_signal, cutoff=0.3):
    """A lowpass filter to a given file using the Butterworth filter.

    Args:
        signal (array): ECG Signal
        cutoff (float): 0.3 for PVC & 0.2 AFIB
    
    Returns:
        array: the low-pass filtered signal of the input file.
    """
    b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, ecg_signal)
    return low_passed

def pt_detection_1(ecg_signal, r_index, q_index, s_index, width, lp_thres=0.2):
    """Detects peaks in a given signal within a specified range and returns the peak indices.

    Args:
        ecg_signal (array): ECG signal
        r_index (list/array): Indices representing the R-peaks in an ECG signal
        q_index (_type_): Indices representing the Q waves in an ECG signal
        s_index (_type_): Indices representing the S waves in an ECG signal
        width (_type_): In the find_peaks function to specify the minimum width of
    peaks to be detected. It is a positive integer value

    Returns:
        tuple: two lists: pt and p_t.
    """
    max_signal = max(ecg_signal)/100
    pt = []
    p_t = []
    for i in range (0, len(r_index)-1):
        aoi = ecg_signal[s_index[i]:q_index[i+1]]
        max_signal = max(ecg_signal)/100
        low = fir_lowpass_filter(aoi,lp_thres,30)
        if ecg_signal[r_index[i]]<0:
            max_signal=0.05
        else:
            max_signal=max_signal
        if aoi.any():
            peaks,_ = find_peaks(low,height=max_signal,width=width)
            peaks1=peaks+(s_index[i])
        else:
            peaks1 = [0]
        p_t.append(list(peaks1))
        pt.extend(list(peaks1))
        for i in range (len(p_t)):
            if not p_t[i]:
                p_t[i] = [0]
    return pt, p_t

def pt_detection_2(ecg_signal, r_index, q_index, s_index, width, lp_thres=0.2):
    """Detects peaks in a given signal within a specified range and returns the peak indices.

    Args:
        ecg_signal (array): ECG signal
        r_index (list/array): Indices representing the R-peaks in an ECG signal
        q_index (_type_): Indices representing the Q waves in an ECG signal
        s_index (_type_): Indices representing the S waves in an ECG signal
        width (_type_): In the find_peaks function to specify the minimum width of
    peaks to be detected. It is a positive integer value

    Returns:
        tuple: two lists: pt and p_t.
    """
    pt = []
    p_t = []
    for i in range (0, len(r_index)-1):
        aoi = ecg_signal[s_index[i]:q_index[i+1]]
        if aoi.any():
            low = fir_lowpass_filter(aoi,lp_thres,30)
            if ecg_signal[r_index[i]]<0:
                max_signal=0.05
            else:
                max_signal= max(low)*0.2
            if aoi.any():
                peaks,_ = find_peaks(low,height=max_signal,width=width)
                peaks1=peaks+(s_index[i])
            else:
                peaks1 = [0]
            p_t.append(list(peaks1))
            pt.extend(list(peaks1))
            for i in range (len(p_t)):
                if not p_t[i]:
                    p_t[i] = [0]
        else:
            p_t.append([0])
    return pt, p_t

def pt_detection_3(ecg_signal, r_index, q_index, s_index, width, lp_thres=0.2):
    """Detects peaks in a given signal within a specified range and returns the peak indices.

    Args:
        ecg_signal (array): ECG signal
        r_index (list/array): Indices representing the R-peaks in an ECG signal
        q_index (_type_): Indices representing the Q waves in an ECG signal
        s_index (_type_): Indices representing the S waves in an ECG signal
        width (_type_): In the find_peaks function to specify the minimum width of
    peaks to be detected. It is a positive integer value

    Returns:
        tuple: two lists: pt and p_t.
    """
    pt = []
    p_t = []
    for i in range (0, len(r_index)-1):
        aoi = ecg_signal[s_index[i]:q_index[i+1]]
        low = fir_lowpass_filter(aoi,lp_thres,30)
        if aoi.any():
            peaks,_ = find_peaks(low,prominence=0.05,width=width)
            peaks1=peaks+(s_index[i])
        else:
            peaks1 = [0]
        p_t.append(list(peaks1))
        pt.extend(list(peaks1))
        for i in range (len(p_t)):
            if not p_t[i]:
             p_t[i] = [0]

    return pt, p_t

def pt_detection_4(ecg_signal, r_index, q_index, s_index):
    """Detects peaks in a given signal within a specified range and returns the peak indices.

    Args:
        b_signal (array): ECG signal
        r_index (list/array): Indices representing the R-peaks in an ECG signal
        q_index (_type_): Indices representing the Q waves in an ECG signal
        s_index (_type_): Indices representing the S waves in an ECG signal
        width (_type_): In the find_peaks function to specify the minimum width of
    peaks to be detected. It is a positive integer value

    Returns:
        tuple: two lists: pt and p_t.
    """
    def all_peaks_7(arr):
        """The indices of all peaks in the array, where a peak is
        defined as a point that is higher than its neighboring points.

        Args:
            arr (array): An input array of numbers

        Returns:
            array: The function `all_peaks_7` returns a sorted numpy array of indices where peaks occur in
        the input array `arr`.
        """
        sign_arr = np.sign(np.diff(arr))
        pos = np.where(np.diff(sign_arr) == -2)[0] + 1
        neg = np.where(np.diff(sign_arr) == 2)[0] + 1
        all_peaks = np.sort(np.concatenate((pos, neg)))
        al = all_peaks.tolist()
        diff = {}
        P, Pa, Pb = [], [], []
        if len(al) > 2:
            for p in pos:
                index = al.index(p)
                if index == 0:
                    m, n, o = arr[0], arr[al[index]], arr[al[index+1]]
                elif index == len(al)-1:
                    m, n, o = arr[al[index-1]], arr[al[index]], arr[-1]
                else:
                    m, n, o = arr[al[index-1]], arr[al[index]], arr[al[index+1]]
                diff[p] = [abs(n-m), abs(n-o)]
            th = np.mean([np.mean([v, m]) for v, m in diff.values()])*.66
            for p, (a, b) in diff.items():
                if a >= th and b >= th:
                    P.append(p)
                    continue
                if a >= th and not Pa:
                    Pa.append(p)
                elif a >= th and arr[p] > arr[Pa[-1]] and np.where(pos==Pa[-1])[0]+1 == np.where(pos==p)[0]:
                    Pa[-1] = p
                elif a >= th:
                    Pa.append(p)
                if b >= th and not Pb:
                    Pb.append(p)
                elif b >= th and arr[p] < arr[Pb[-1]] and np.where(pos==Pb[-1])[0]+1 == np.where(pos==p)[0]:
                    Pb[-1] = p
                elif b >= th:
                    Pb.append(p)
            if len(pos)>1:
                for i in range(1, len(pos)):
                    m, n = pos[i-1], pos[i]
                    if m in Pa and n in Pb:
                        P.append(m) if arr[m] > arr[n] else P.append(n)
            # if Pa and Pa[-1] == pos[-1]:
            #     P.append(Pa[-1])
            # if Pb and Pb[0] == pos[0]:
            #     P.append(Pb[0])
        else:
            P = pos
        return np.sort(P)
    pt, p_t = [], []
    for i in range(1, len(r_index)):
        q0, r0, s0 = q_index[i - 1], r_index[i - 1], s_index[i - 1]
        q1, r1, s1 = q_index[i], r_index[i], s_index[i]
        arr = ecg_signal[s0+7:q1-7]
        peaks = list(all_peaks_7(arr) + s0 + 7) 
        if peaks:
            pt.extend(peaks)
            p_t.append(peaks)
        else:
            p_t.append([0])
    return pt, p_t

def afib_percentage(p_t, fs=200):
    percentage = []
    # threshold = int(fs*0.28)
    if fs == 200:
        threshold = 40
    else:
        threshold = 70
    for k in range(0, len(p_t)):
        d = p_t[k]
        cnt = 0
        difference = np.diff(p_t[k]).tolist()
        # print(difference, "=============difference")
        # if difference==[]:
        #      percentage.append(0)
        if 1 < len(p_t[k]) < 5:
            for jj in range(0, len(difference)):
                if difference[jj] < threshold:
                    cnt += 1
                else:
                    pass
            if cnt > (0.5 * len(difference)):
                percentage.append(1)
            else:
                percentage.append(0)
        elif len(p_t[k]) > 4:
            percentage.append(1)
        elif len(p_t[k]) == 1:
            if p_t[k] == [0]:
                percentage.append(0)
            else:
                percentage.append(1)
        else:
            percentage.append(0)
    # print(percentage, "============percentage")
    per = (percentage.count(1) / len(percentage)) * 100
    # print(per, "=======per")
    return per



    """Calculates the percentage of atrial fibrillation (AFib) in a given patient's
    electrocardiogram (ECG) data based on the duration of irregular heartbeats.

    Args:
        p_t (list of lists): A list of lists, where each inner list represents the time intervals between consecutive
    heartbeats for a patient. The time intervals are in milliseconds

    Returns:
        _type_: The average percentage of AFib episodes based on the input parameter `p_t`, which is a list
    of lists containing the time intervals between consecutive R-peaks in an ECG signal.
    """
##    percentage = []
##    threshold = int(fs*0.35)
##    if fs != 200:
##        a = 1
##    else:
##        a = 2
##    for k in range (0,len(p_t)):
##        d = p_t[k]
##        cnt = 0
##        difference = np.diff(p_t[k]).tolist()
##        # if difference==[]:
##        #      percentage.append(0)
##        if a<len(p_t[k])<5:
##            for jj in range (0,len(difference)):
##                if difference[jj] < threshold:
##                    cnt +=1
##                else:
##                    pass
##            if cnt>(0.5*len(difference)):
##                percentage.append(1)
##            else:
##                percentage.append(0)
##        elif len(p_t[k])>4:
##            percentage.append(1)
##        elif len(p_t[k]) == 1:
##            if p_t[k] == [0]:
##                percentage.append(0)
##            else:
##                percentage.append(1)
##        else:
##            percentage.append(0)
##
##    per = (percentage.count(1)/len(percentage))*100
##
##    return per

def AFIB_cross_check(ecg_signal, p_t, r_label, fs=200):
    total_len = len(p_t)
    # Get the length of each inner list
    lengths = []
    output_list = []
    label = "Abnormal"
    for inner_list in p_t:
        if inner_list == [0]:
            lengths.append(0)
        else:
            lengths.append(len(inner_list))

            if len(inner_list) > 1:
                window_size = 20
                area_list = []

                for peaks in inner_list:
                    window_start = max(0, peaks - window_size // 2)
                    window_end = min(len(ecg_signal), peaks + window_size // 2)
                    area = trapz(ecg_signal[window_start:window_end])
##                    print(area)
                    if 25 <= area <= 50:
                        area_list.append(round(area,2))

                if len(area_list)/len(inner_list)>=0.4:
                    output_list.append(1)
                else:
                    output_list.append(0)
            else:
                output_list.append(0)
    # print(output_list)
    # Print the lengths
    # print(lengths, "length--------------")
    zeros = lengths.count(0)
    ones = lengths.count(1)
    twos = lengths.count(2)
    threes_to_nines = sum([lengths.count(i) for i in range(3, 10)])
    af_per = 0.
    if (round((zeros / total_len) * 100) >= 11 and round((ones / total_len) * 100) >= 30 and round(
            ((zeros + ones) / total_len) * 100) >= 30) or (
            round((threes_to_nines / total_len) * 100) >= 4 and afib_percentage(p_t,
                                                                                fs=fs) >= 56) and r_label != 'Regular':
        if len(output_list) != 0:
            if output_list.count(1) / len(output_list) > 0.40:  # 0.65

                label = "Afib"
                af_per = 1.
                
    return label, af_per, zeros, ones, twos, threes_to_nines


##    """
##    Classify the ECG signal based on certain criteria.
##
##    Args:
##        p_t (list): It is a list of lists where each inner list represents a segment of ECG signal's P & T waves.
##
##    Returns:
##        str: a label indicating whether the signal is "Afib", "Abnormal", or "Normal".
##    """
##    total_len = len(p_t)
##    # Get the length of each inner list
##    lengths = []
##    for inner_list in p_t:
##        if inner_list == [0]:
##            lengths.append(0)
##        else:
##            lengths.append(len(inner_list))
##    zeros = lengths.count(0)
##    ones = lengths.count(1)
##    twos = lengths.count(2)
##    threes_to_nines = sum([lengths.count(i) for i in range(3, 10)])
##    
##    # if round(((twos+ones)/total_len)*100)>=70 and round((ones/total_len)*100)>=50:
##    #     label = "Abnormal"
##    if (round((zeros/total_len)*100)>=17 and 
##        round((ones/total_len)*100)>=40 and 
##        round(((zeros+ones)/total_len)*100)>=40) or (round((threes_to_nines/total_len)*100) >= 6 and 
##                                                     afib_percentage(p_t,fs=fs)>=56):
##        label = "Afib"
##    elif round((twos/total_len)*100) >= 50:
##        label = "Abnormal"
##    elif round((zeros/total_len)*100) >= 70 or round((ones/total_len)*100) >= 70:
##        label = "Abnormal"
##    else:
##        label = "Abnormal"
##    return label, zeros, ones, twos, threes_to_nines

def find_pt(ecg_signal, r_index, q_index, s_index, width=(5,50), lp_thres = 0.2):
    _, p_t1 = pt_detection_1(ecg_signal, r_index, q_index, s_index, width, lp_thres)
    _, p_t2 = pt_detection_2(ecg_signal, r_index, q_index, s_index, width, lp_thres)
    _, p_t3 = pt_detection_3(ecg_signal, r_index, q_index, s_index, width, lp_thres)
    _, p_t4 = pt_detection_4(ecg_signal, r_index, q_index, s_index) 
    pt = []
    p_t = []
    for i in range(len(p_t1)):
        _ = []
        for _pt in set(p_t1[i]+p_t2[i]+p_t3[i]+p_t4[i]):
            count = 0
            if any(val in p_t1[i] for val in range (_pt-2,_pt+3)):
                count += 1
            if any(val in p_t2[i] for val in range (_pt-2,_pt+3)):
                count += 1
            if any(val in p_t3[i] for val in range (_pt-2,_pt+3)):
                count += 1
            if any(val in p_t4[i] for val in range (_pt-2,_pt+3)):
                count += 1
            if count >= 3:
                _.append(_pt)
            _.sort()
        if _:
            p_t.append(_)
        else:
            p_t.append([0])
    result = []
    for sublist in p_t:
        # print(sublist)
        temp = [sublist[0]]
        for i in range(1, len(sublist)):
            if abs(sublist[i] - sublist[i-1]) > 5:
                temp.append(sublist[i])
            else:
                temp[-1] = sublist[i]  
        if temp:
            result.append(temp)
            pt.extend(temp)
        else:
            result.append([0])
    p_t = result
    return p_t, pt

def segricate_p_t_pr_inerval(r_index, p_t, fs=200, thres=0.5):
    """
    threshold = 0.37 for JR and 0.5 for other diseases
    """
    # print(thres, fs)
    diff_arr = ((np.diff(r_index)*thres)/fs).tolist()
    t_peaks_list, p_peaks_list, pr_interval, extra_peaks_list = [], [], [], []
    # threshold = (-0.0012 * len(r_index)) + 0.25
    # print(f"threshold == {threshold}")
    # print(p_t)
    for i in range(len(p_t)):
        p_dis = (r_index[i+1]-p_t[i][-1])/fs
        t_dis = (r_index[i+1]-p_t[i][0])/fs
        threshold = diff_arr[i]
        # print(f"Distance == {p_dis}")
        if t_dis > threshold and (p_t[i][0]>r_index[i]): t_peaks_list.append(p_t[i][0])
        if p_dis <= threshold: 
            p_peaks_list.append(p_t[i][-1])
            pr_interval.append(p_dis*fs)
        if len(p_t[i])>0:
            if p_t[i][0] in t_peaks_list:
                if p_t[i][-1] in p_peaks_list:
                    extra_peaks_list.extend(p_t[i][1:-1])
                else:
                    extra_peaks_list.extend(p_t[i][1:])
            elif p_t[i][-1] in p_peaks_list:
                extra_peaks_list.extend(p_t[i][:-1])
            else:
                extra_peaks_list.extend(p_t[i])

    p_label, pr_label = "", ""
    if thres >= 0.5 and p_peaks_list and len(p_peaks_list)>2:
        pp_intervals = np.diff(p_peaks_list)
        pp_std = np.std(pp_intervals)
        pp_mean = np.mean(pp_intervals)
        threshold = 0.12 * pp_mean
        if pp_std <= threshold:
            p_label = "Constanat"
        else:
            p_label = "Not Constant"
        
        count=0
        for i in pr_interval:
            if round(np.mean(pr_interval)*0.75) <= i <= round(np.mean(pr_interval)*1.25):
                count +=1
        if len(pr_interval) != 0: 
            per = count/len(pr_interval)
            pr_label = 'Not Constant' if per<=0.7 else 'Constant'
    data = {'T_Index':t_peaks_list, 
            'P_Index':p_peaks_list, 
            'PR_Interval':pr_interval, 
            'P_Label':p_label, 
            'PR_label':pr_label,
            'Extra_Peaks':extra_peaks_list}
    return data

def pqrst_detection(ecg_signal, fs=200, thres=0.5, lp_thres=0.2, rr_thres=0.12, width=(5, 50)):
    
    r_index = find_r_peaks(ecg_signal,fs=fs)
    rr_intervals = np.diff(r_index)
    rr_std = np.std(rr_intervals)
    rr_mean = np.mean(rr_intervals)
    threshold = rr_thres * rr_mean
    if rr_std <= threshold:
        r_label = "Regular"
    else:
        r_label = "Irregular"
    hr_ = hr_count(ecg_signal, r_index, fs=fs)
    if thres < 0.5:
        ecg_signal = lowpass(ecg_signal, cutoff = 0.2)
    sd, qd = int(fs * 0.115), int(fs * 0.08)
    s_index = find_s_index(ecg_signal, r_index, sd)
    # q_index = find_q_index(ecg_signal, r_index, qd)
    # s_index = find_new_s_index(ecg_signal,r_index,sd)
    q_index = find_new_q_index(ecg_signal,r_index,qd)
    j_index = find_j_index(ecg_signal, s_index, fs=fs)
    p_t, pt = find_pt(ecg_signal, r_index, q_index, s_index, lp_thres=lp_thres, width=width)
    data_ = segricate_p_t_pr_inerval(r_index, p_t, fs=fs, thres=thres)
    data = {'R_Label':r_label, 
            'R_index':r_index, 
            'Q_Index':q_index, 
            'S_Index':s_index, 
            'J_Index':j_index, 
            'P_T List':p_t, 
            'PT PLot':pt, 
            'HR_Count':hr_, 
            'T_Index':data_['T_Index'], 
            'P_Index':data_['P_Index'],
            'Ex_Index':data_['Extra_Peaks'], 
            'PR_Interval':data_['PR_Interval'], 
            'P_Label':data_['P_Label'], 
            'PR_label':data_['PR_label']}
    return data

def afib_detection(ecg_signal, fs=200):
    
    baseline_signal = baseline_construction_200(ecg_signal,kernel_size=101)
        
##    low_pass_signal = lowpass(ecg_signal, cutoff=0.2)
    
    pqrst_data = pqrst_detection(baseline_signal, fs=fs)
    r_label = pqrst_data['R_Label']
    p_t = pqrst_data['P_T List']
    label = 'Normal'
    if r_label == "Irregular":
        label,af_per, zeros,ones,twos,three = AFIB_cross_check(ecg_signal, p_t, r_label,fs=fs)
    
    return label


def funcs(sorted_data):
        A = pd.DataFrame(sorted_data)[['dateTime', 'data']]
        A1 = A["data"].to_list()
        l = []
        data_list = []
        date_time = []
        for i in range(len(A1)):
            d = A["dateTime"][i]
            # print(d)
            start = 0
            stop = 4
            kk= 0
            while True:
                l.append(A1[i][start:stop])
                high = l[0][2]+l[0][3]
                low = l[0][0]+ l[0][1]

                highdec = int(str(high), 16)
                lowdec = int(str(low),16)
                val = (int(highdec)*256)+(int(lowdec))
                voltage = (4.6/4095)*val
                data_list.append(voltage)
                
                date_time.append(d)
                start+=4
                stop+=4
                l.clear()
                kk+=1
                if stop> len(A1[i]):
                    break
        date_time = np.array(date_time)
##        plt.plot(data_list)
##        plt.show()
        df = pd.DataFrame({"DateTime":date_time, "ECG":data_list})
        return df 


##def wide_qrs_find(q_index, r_index, s_index, hr_count, fs=200):
##    """The distance between the QRS complex's Q wave and S wave.
##
##    Args:
##        q_index (list): Q waves indices in an ECG signal
##        r_index (list): R-peak indices in an ECG signal
##        s_index (list): S waves indices in an ECG signal
##        fs (int, optional): sampling rate of the ECG signal, defaults to 200 (optional)
##
##    Returns:
##        array: the R-peak indices that correspond to wide QRS complexes in an ECG signal.
##    The function takes in three arrays as inputs: q_index, r_index, and s_index, which represent the
##    indices of the Q-wave onset, R-peak, and S-wave offset, respectively. The function loops through the
##    R-peak indices and calculates the difference between
##    """
##    max_indexs = 0
##    if hr_count<60:
##        ms = 0.135
##        max_indexs = int(fs * ms)
##    elif 60 <= hr_count <100:
##        ms = 0.135
##        max_indexs = int(fs * ms)
##    pvc = []
##    difference = []
##    for k in range(len(r_index)):
##        diff = s_index[k] - q_index[k]
##        # print(diff)
##        difference.append(diff)
##        if max_indexs != 0:
##            if diff>=max_indexs:
##                pvc.append(r_index[k])
##    mode_ = mode(difference)
##    # print("Mode:",mode_,"\n","Difference:",[i/200 for i in difference])
##    if hr_count>=100:
##        for k in range(len(r_index)-1):
##            diff = s_index[k] - q_index[k]
##            if diff>33:#or diff > mode_+2:
##                pvc.append(r_index[k])
##            else:
##                pass
##    # print("PVC:",pvc)
##    q_s_difference = [round(i/fs, 2) for i in difference]
##    return np.array(pvc), q_s_difference



def find_s_newnew_index(ecg, R_index, d):
    end_index = len(ecg) - 1
    range_per = 0.03
    small_range_per = 0.01
    s = []
    for r in R_index:
        r_range = abs(ecg[r] * range_per)
        r_range__ = abs(ecg[r] * small_range_per)
        s_, sss = [], []
        if r == len(ecg):
            s.append(r)
            continue
        if ecg[r] > 0:
            c = r
            while c+1 <= end_index and ecg[c+1] < ecg[c] and abs(r-c) <= d:
                c += 1
                if (-(r_range) <= ecg[c] <= r_range):
                    sss.append(c)
            if (-(r_range) <= ecg[c] <= r_range)  or c == end_index or abs(r-c) > d:
                s_.append(c)
            # s_.append(c)
            while c+1 <= end_index and abs(r-c) <= d:
                while c+1 <= end_index and ecg[c+1] > ecg[c] and abs(r-c) <= d:
                    c += 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        sss.append(c)
                while c+1 <= end_index and ecg[c+1] < ecg[c] and abs(r-c) <= d:
                    c += 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        sss.append(c)
                if s_ and s_[-1] == c:
                    break
                s_.append(c)
                if abs(r-c) <= d and (-(r_range) <= ecg[c] <= r_range) or c == end_index:
                    break
            
        else:
            c = r
            while c+1 <= end_index and ecg[c+1] > ecg[c] and abs(r-c) <= d:
                c += 1
                if (-(r_range) <= ecg[c] <= r_range):
                    sss.append(c)
            if (-(r_range) <= ecg[c] <= r_range) or c == end_index or abs(r-c) > d:
                s_.append(c)
            # s_.append(c)
            while c <= end_index and abs(r-c) <= d:
                while c+1 <= end_index and ecg[c+1] > ecg[c] and abs(r-c) <= d:
                    c += 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        sss.append(c)
                while c+1 <= end_index and ecg[c+1] < ecg[c] and abs(r-c) <= d:
                    c += 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        sss.append(c)
                if s_ and s_[-1] == c:
                    break
                s_.append(c)
                if abs(r-c) <= d and (-(r_range) <= ecg[c] <= r_range)  or c == end_index:
                    break
        if s_ or sss:
            a = 0
            for _s in s_[::-1]:
                if (-(r_range__) <= ecg[_s] <= r_range__):
                    a = 1
                    s.append(_s)
                    break
            if a == 0:
                for _s in sss[::-1]:
                    if (-(r_range__) <= ecg[_s] <= r_range__):
                        a = 1
                        s.append(_s)
                        break
            if a == 0:
                if ecg[r] > 0:
                    for _s in s_[::-1]:
                        if ecg[_s] <= r_range:
                            a = 1
                            s.append(_s)
                            break
                    if a == 0:
                        for _s in sss[::-1]:
                            if ecg[_s] <= r_range__:
                                a = 1
                                s.append(_s)
                                break
                else:
                    for _s in s_[::-1]:
                        if -r_range <= ecg[_s] :
                            a = 1
                            s.append(_s)
                            break
                    if a == 0:
                        for _s in sss[::-1]:
                            if -r_range__ <= ecg[_s] :
                                a = 1
                                s.append(_s)
                                break
            if a == 0: 
                if r+d<=len(ecg):
                    s_array = ecg[r:r+d]
                else:
                    s_array = ecg[r:]
                if ecg[r] > 0:
                    s_index = r+np.where(s_array == min(s_array))[0][0]
                else:
                    s_index = r+np.where(s_array == max(s_array))[0][0]
                s.append(s_index)
    return np.sort(s)

def find_q_newnew_index(ecg, R_index, d):
    q = []
    range_per = 0.03
    small_range_per = 0.01
    for r in R_index:
        r_range = abs(ecg[r] * range_per)
        r_range__ = abs(ecg[r] * small_range_per)
        q_, qqq = [], []
        if r == 0:
            q.append(r)
            continue
        if ecg[r] > 0:
            c = r
            while c > 0 and ecg[c-1] < ecg[c] and abs(r-c) <= d:
                c -= 1
                if (-(r_range) <= ecg[c] <= r_range):
                        qqq.append(c)
            if (-(r_range) <= ecg[c] <= r_range) or c == 0 or abs(r-c) > d:
                q_.append(c)
            # q_.append(c)
            while c > 0 and abs(r-c) <= d:
                while c > 0 and ecg[c-1] > ecg[c] and abs(r-c) <= d:
                    c -= 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        qqq.append(c)
                while c > 0 and ecg[c-1] < ecg[c] and abs(r-c) <= d:
                    c -= 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        qqq.append(c)
                if q_ and q_[-1] == c:
                    break
                q_.append(c)
                if abs(r-c) <= d and (-(r_range) <= ecg[c] <= r_range) or c == 0:
                    break
        else:
            c = r
            while c > 0 and ecg[c-1] > ecg[c] and abs(r-c) <= d:
                c -= 1
                if (-(r_range) <= ecg[c] <= r_range):
                    qqq.append(c)
            if (-(r_range) <= ecg[c] <= r_range) or c == 0 or abs(r-c) > d:
                q_.append(c)
            # q_.append(c)
            while c > 0 and abs(r-c) <= d:
                while c > 0 and ecg[c-1] < ecg[c] and abs(r-c) <= d:
                    c -= 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        qqq.append(c)
                while c > 0 and ecg[c-1] > ecg[c] and abs(r-c) <= d:
                    c -= 1
                    if (-(r_range) <= ecg[c] <= r_range):
                        qqq.append(c)
                if q_ and q_[-1] == c:
                    break
                q_.append(c)
                if abs(r-c) <= d and (-(r_range) <= ecg[c] <= r_range) or c == 0:
                    break
        if q_ or qqq:
            a = 0
            for _q in q_[::-1]:
                if (-(r_range__) <= ecg[_q] <= r_range__):
                    a = 1
                    q.append(_q)
                    break
            if a == 0:
                for _q in qqq[::-1]:
                    if (-(r_range__) <= ecg[_q] <= r_range__):
                        a = 1
                        q.append(_q)
                        break
            if a == 0:
                if ecg[r] > 0:
                    for _q in q_[::-1]:
                        if ecg[_q] <= r_range:
                            a = 1
                            q.append(_q)
                            break
                    if a == 0:
                        for _q in qqq[::-1]:
                            if ecg[_q] <= r_range__:
                                a = 1
                                q.append(_q)
                                break
                else:
                    for _q in q_[::-1]:
                        if -r_range <= ecg[_q] :
                            a = 1
                            q.append(_q)
                            break
                    if a == 0:
                        for _q in qqq[::-1]:
                            if -r_range__ <= ecg[_q] :
                                a = 1
                                q.append(_q)
                                break
            if a == 0:
                if 0 <= r - d:
                    q_array = ecg[r - d:r]
                else:
                    q_array = ecg[:r]
                if ecg[r] > 0:
                    q_index = r - (len(q_array) - np.where(q_array == min(q_array))[0][0])
                else:
                    q_index = r - (len(q_array) - np.where(q_array == max(q_array))[0][0])
                q.append(q_index)
    return np.sort(q)

def wide_qrs(q_index, r_index, s_index, fs=200):
    label = 'Abnormal'
    wideQRS = []
    thresold = round(fs * 0.12)
    for k in range(len(r_index)):
        diff = s_index[k] - q_index[k]
        # print(diff)
        if diff > thresold:
            wideQRS.append(r_index[k])
    if len(wideQRS)/ len(r_index) >= 0.50:
        label = 'Wide_QRS'
    return label, wideQRS

def mobitz_I_list(pr_interval):
    sign = np.sign(np.diff(pr_interval))
    indexs = np.where(sign == -1)[0] + 1
    l = []
    for i, val in enumerate(indexs):
        if l:
            l.append(pr_interval[indexs[i-1]:val])
        else:
            l.append(pr_interval[:val])
    if l:
        l.append(pr_interval[indexs[-1]:])
    else:
        l.append(pr_interval)
    length_l = [len(i) for i in l]
    max_lists_length = [i for i in length_l if i > np.mean(length_l)]
    if max_lists_length:
        output = [i for i in l if len(i) == max(max_lists_length)]
        return max(max_lists_length), output
    return 0,[]

def block_check(r_label, pr_label, p_label, pr_interval, fs=200):
    pr_cont, pr_list = mobitz_I_list(pr_interval)
    if len(pr_interval) > 0:
        if pr_label == 'Not Constant' and r_label == 'Regular' and p_label == 'Constanat':
            label = 'III Degree'
        elif round((pr_cont/len(pr_interval))*100)>=85 and r_label == 'Irregular' and pr_label == 'Not Constant':
            label = "Mobitz I"
        elif r_label == 'Irregular':
            label = 'Abnormal'
        else:
            label = "Abnormal"
    else:
        label = "Abnormal"
    data = {'Mobiz_I_list': pr_list,
            'Block_Label': label}
    return data



def detect_block(ecg_signal, fs=200):
    if fs != 200:
        ecg_signal = MinMaxScaler(feature_range=(0,4)).fit_transform(ecg_signal.reshape(-1,1)).squeeze()
    if fs == 200:
        baseline_signal = baseline_construction_200(ecg_signal,kernel_size=101)
    pqrst_data = pqrst_detection(baseline_signal, fs=fs, lp_thres=0.1)
    r_label = pqrst_data['R_Label']
    pr_label = pqrst_data['PR_label']
    p_label = pqrst_data['P_Label']
    pr_interval = pqrst_data['PR_Interval']
    hr_counts = pqrst_data['HR_Count']
    r_index = pqrst_data['R_index']
    q_index = pqrst_data['Q_Index']
    s_index = pqrst_data['S_Index']
    j_index = pqrst_data['J_Index']
    p_t = pqrst_data['P_T List']
    pt = pqrst_data['PT PLot']
    t_index = pqrst_data['T_Index'] 
    p_index = pqrst_data['P_Index']
    ex_index = pqrst_data['Ex_Index']
    if 30<=hr_counts<=80:
        block_label = block_check(r_label, pr_label, p_label, pr_interval, fs=fs)['Block_Label']
    else:
        block_label = "Abnormal"
    lowpass_signal = lowpass(baseline_signal)
    if block_label == "Abnormal": block_label = 'Normal' if r_label == 'Regular' else 'Abnormal'
    data = {'Input_Signal':ecg_signal, 
            'Baseline_Signal':baseline_signal, 
            'Lowpass_signal':lowpass_signal, 
            'Block_Label':block_label.upper().replace("_","-"), 
            'RR_Label':r_label, 
            'R_Index':r_index, 
            'Q_Index':q_index, 
            'S_Index':s_index, 
            'J_Index':j_index,
            'T_Index' : t_index,
            'P_Index' : p_index,
            'Ex_Index' : ex_index, 
            'P_T':pt, 
            'HR_Count':hr_counts}
    return data

def wide_qrss(q_index, r_index, s_index, fs=200):
    label = 'Abnormal'
    wideQRS = []
    diff_arr = []
    thresold = round(fs * 0.12)
    for k in range(len(r_index)):
        diff = s_index[k] - q_index[k]
        # print(diff)
        diff_arr.append(diff)
        if diff > thresold:
            wideQRS.append(r_index[k])
    if len(wideQRS)/ len(r_index) >= 0.50:
        label = 'Wide_QRS'
        
    data = {"wideqrs_label": label,
            "wideqrs_index": wideQRS,
            "diff_array": diff_arr}
    return data


def wide_qrs_detection(ecg_signal, fs=200):
    if fs != 200:
        ecg_signal = MinMaxScaler(feature_range=(0,4)).fit_transform(ecg_signal.reshape(-1,1)).squeeze()
    if fs == 200:
        baseline_signal = baseline_construction_200(ecg_signal,kernel_size=101)
    pqrst_data = pqrst_detection(baseline_signal, fs=fs)
    r_label = pqrst_data['R_Label']
    pr_label = pqrst_data['PR_label']
    p_label = pqrst_data['P_Label']
    pr_interval = pqrst_data['PR_Interval']
    hr_counts = pqrst_data['HR_Count']
    r_index = pqrst_data['R_index']
    q_index = pqrst_data['Q_Index']
    s_index = pqrst_data['S_Index']
    j_index = pqrst_data['J_Index']
    p_t = pqrst_data['P_T List']
    pt = pqrst_data['PT PLot']
    t_index = pqrst_data['T_Index'] 
    p_index = pqrst_data['P_Index']
    ex_index = pqrst_data['Ex_Index']
    
    wideqrs_data = wide_qrss(q_index, r_index, s_index, fs=200)
    
    return wideqrs_data

def hamilton_segmenter(signal=None, sampling_rate=200.0):
    """
    The hamilton_segmenter function is a QRS complex detection algorithm that uses filtering, smoothing,
    and detection rules to identify R-peaks in an ECG signal.
    
    :param signal: The input ECG signal to be analyzed
    :param sampling_rate: The sampling rate of the input signal in Hz (samples per second)
    :return: a tuple containing an array of R-peaks detected in the input signal. The name of the array
    is "rpeaks".
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    sampling_rate = float(sampling_rate)
    length = len(signal)
    dur = length / sampling_rate

    # algorithm parameters
    v1s = int(1.0 * sampling_rate)
    v100ms = int(0.1 * sampling_rate)
    TH_elapsed = np.ceil(0.36 * sampling_rate)
    sm_size = int(0.08 * sampling_rate)
    init_ecg = 4  # seconds for initialization
    if dur < init_ecg:
        init_ecg = int(dur)

    # filtering
    filtered, _, _ = st.filter_signal(
        signal=signal,
        ftype="butter",
        band="lowpass",
        order=4,
        frequency=20.0,
        sampling_rate=sampling_rate,
    )
    filtered, _, _ = st.filter_signal(
        signal=filtered,
        ftype="butter",
        band="highpass",
        order=4,
        frequency=3.0,
        sampling_rate=sampling_rate,
    )

    # diff
    dx = np.abs(np.diff(filtered, 1) * sampling_rate)

    # smoothing
    dx, _ = st.smoother(signal=dx, kernel="hamming", size=sm_size, mirror=True)

    # buffers
    qrspeakbuffer = np.zeros(init_ecg)
    noisepeakbuffer = np.zeros(init_ecg)
    peak_idx_test = np.zeros(init_ecg)
    noise_idx = np.zeros(init_ecg)
    rrinterval = sampling_rate * np.ones(init_ecg)

    a, b = 0, v1s
    all_peaks, _ = st.find_extrema(signal=dx, mode="max")
    for i in range(init_ecg):
        peaks, values = st.find_extrema(signal=dx[a:b], mode="max")
        try:
            ind = np.argmax(values)
        except ValueError:
            pass
        else:
            # peak amplitude
            qrspeakbuffer[i] = values[ind]
            # peak location
            peak_idx_test[i] = peaks[ind] + a

        a += v1s
        b += v1s

    # thresholds
    ANP = np.median(noisepeakbuffer)
    AQRSP = np.median(qrspeakbuffer)
    TH = 0.475
    DT = ANP + TH * (AQRSP - ANP)
    DT_vec = []
    indexqrs = 0
    indexnoise = 0
    indexrr = 0
    npeaks = 0
    offset = 0

    beats = []

    # detection rules
    # 1 - ignore all peaks that precede or follow larger peaks by less than 200ms
    lim = int(np.ceil(0.15 * sampling_rate))
    diff_nr = int(np.ceil(0.045 * sampling_rate))
    bpsi, bpe = offset, 0

    for f in all_peaks:
        DT_vec += [DT]
        # 1 - Checking if f-peak is larger than any peak following or preceding it by less than 200 ms
        peak_cond = np.array(
            (all_peaks > f - lim) * (all_peaks < f + lim) * (all_peaks != f)
        )
        peaks_within = all_peaks[peak_cond]
        if peaks_within.any() and (max(dx[peaks_within]) > dx[f]):
            continue

        # 4 - If the peak is larger than the detection threshold call it a QRS complex, otherwise call it noise
        if dx[f] > DT:
            # 2 - look for both positive and negative slopes in raw signal
            if f < diff_nr:
                diff_now = np.diff(signal[0 : f + diff_nr])
            elif f + diff_nr >= len(signal):
                diff_now = np.diff(signal[f - diff_nr : len(dx)])
            else:
                diff_now = np.diff(signal[f - diff_nr : f + diff_nr])
            diff_signer = diff_now[diff_now > 0]
            if len(diff_signer) == 0 or len(diff_signer) == len(diff_now):
                continue
            # RR INTERVALS
            if npeaks > 0:
                # 3 - in here we check point 3 of the Hamilton paper
                # that is, we check whether our current peak is a valid R-peak.
                prev_rpeak = beats[npeaks - 1]

                elapsed = f - prev_rpeak
                # if the previous peak was within 360 ms interval
                if elapsed < TH_elapsed:
                    # check current and previous slopes
                    if prev_rpeak < diff_nr:
                        diff_prev = np.diff(signal[0 : prev_rpeak + diff_nr])
                    elif prev_rpeak + diff_nr >= len(signal):
                        diff_prev = np.diff(signal[prev_rpeak - diff_nr : len(dx)])
                    else:
                        diff_prev = np.diff(
                            signal[prev_rpeak - diff_nr : prev_rpeak + diff_nr]
                        )

                    slope_now = max(diff_now)
                    slope_prev = max(diff_prev)

                    if slope_now < 0.5 * slope_prev:
                        # if current slope is smaller than half the previous one, then it is a T-wave
                        continue
                if dx[f] < 3.0 * np.median(qrspeakbuffer):  # avoid retarded noise peaks
                    beats += [int(f) + bpsi]
                else:
                    continue

                if bpe == 0:
                    rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                    indexrr += 1
                    if indexrr == init_ecg:
                        indexrr = 0
                else:
                    if beats[npeaks] > beats[bpe - 1] + v100ms:
                        rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                        indexrr += 1
                        if indexrr == init_ecg:
                            indexrr = 0

            elif dx[f] < 3.0 * np.median(qrspeakbuffer):
                beats += [int(f) + bpsi]
            else:
                continue

            npeaks += 1
            qrspeakbuffer[indexqrs] = dx[f]
            peak_idx_test[indexqrs] = f
            indexqrs += 1
            if indexqrs == init_ecg:
                indexqrs = 0
        if dx[f] <= DT:
            # 4 - not valid
            # 5 - If no QRS has been detected within 1.5 R-to-R intervals,
            # there was a peak that was larger than half the detection threshold,
            # and the peak followed the preceding detection by at least 360 ms,
            # classify that peak as a QRS complex
            tf = f + bpsi
            # RR interval median
            RRM = np.median(rrinterval)  # initial values are good?

            if len(beats) >= 2:
                elapsed = tf - beats[npeaks - 1]

                if elapsed >= 1.5 * RRM and elapsed > TH_elapsed:
                    if dx[f] > 0.5 * DT:
                        beats += [int(f) + offset]
                        # RR INTERVALS
                        if npeaks > 0:
                            rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                            indexrr += 1
                            if indexrr == init_ecg:
                                indexrr = 0
                        npeaks += 1
                        qrspeakbuffer[indexqrs] = dx[f]
                        peak_idx_test[indexqrs] = f
                        indexqrs += 1
                        if indexqrs == init_ecg:
                            indexqrs = 0
                else:
                    noisepeakbuffer[indexnoise] = dx[f]
                    noise_idx[indexnoise] = f
                    indexnoise += 1
                    if indexnoise == init_ecg:
                        indexnoise = 0
            else:
                noisepeakbuffer[indexnoise] = dx[f]
                noise_idx[indexnoise] = f
                indexnoise += 1
                if indexnoise == init_ecg:
                    indexnoise = 0

        # Update Detection Threshold
        ANP = np.median(noisepeakbuffer)
        AQRSP = np.median(qrspeakbuffer)
        DT = ANP + 0.475 * (AQRSP - ANP)

    beats = np.array(beats)

    r_beats = []
    thres_ch = 1
    adjacency = 0.01 * sampling_rate
    for i in beats:
        error = [False, False]
        if i - lim < 0:
            window = signal[0 : i + lim]
            add = 0
        elif i + lim >= length:
            window = signal[i - lim : length]
            add = i - lim
        else:
            window = signal[i - lim : i + lim]
            add = i - lim
        # meanval = np.mean(window)
        w_peaks, _ = st.find_extrema(signal=window, mode="max")
        w_negpeaks, _ = st.find_extrema(signal=window, mode="min")
        zerdiffs = np.where(np.diff(window) == 0)[0]
        w_peaks = np.concatenate((w_peaks, zerdiffs))
        w_negpeaks = np.concatenate((w_negpeaks, zerdiffs))

        pospeaks = sorted(zip(window[w_peaks], w_peaks), reverse=True)
        negpeaks = sorted(zip(window[w_negpeaks], w_negpeaks))
        
        try:
            twopeaks = [pospeaks[0]]
        except IndexError:
            twopeaks = []
        try:
            twonegpeaks = [negpeaks[0]]
        except IndexError:
            twonegpeaks = []

        # getting positive peaks
        for i in range(len(pospeaks) - 1):
            if abs(pospeaks[0][1] - pospeaks[i + 1][1]) > adjacency:
                twopeaks.append(pospeaks[i + 1])
                break
        try:
            posdiv = abs(twopeaks[0][0] - twopeaks[1][0])
        except IndexError:
            error[0] = True

        # getting negative peaks
        for i in range(len(negpeaks) - 1):
            if abs(negpeaks[0][1] - negpeaks[i + 1][1]) > adjacency:
                twonegpeaks.append(negpeaks[i + 1])
                break
        try:
            negdiv = abs(twonegpeaks[0][0] - twonegpeaks[1][0])
        except IndexError:
            error[1] = True

        # choosing type of R-peak
        n_errors = sum(error)
        try:
            if not n_errors:
                if posdiv > thres_ch * negdiv:
                    # pos noerr
                    r_beats.append(twopeaks[0][1] + add)
                else:
                    # neg noerr
                    r_beats.append(twonegpeaks[0][1] + add)
            elif n_errors == 2:
                if abs(twopeaks[0][1]) > abs(twonegpeaks[0][1]):
                    # pos allerr
                    r_beats.append(twopeaks[0][1] + add)
                else:
                    # neg allerr
                    r_beats.append(twonegpeaks[0][1] + add)
            elif error[0]:
                # pos poserr
                r_beats.append(twopeaks[0][1] + add)
            else:
                # neg negerr
                r_beats.append(twonegpeaks[0][1] + add)
        except IndexError:
            continue

    rpeaks = sorted(list(set(r_beats)))
    rpeaks = np.array(rpeaks, dtype="int")

    return utils.ReturnTuple((rpeaks,), ("rpeaks",))


try:                        
    mycol = mydb["Test6"]
except:
    pass



def pacemaker_detect(ecg_signal, fs=200):
    pqrst_data = pqrst_detection(ecg_signal, fs=fs, width=(3, 50))
    r_index = pqrst_data['R_index']
    q_index = pqrst_data['Q_Index']
    s_index = pqrst_data['S_Index']
    p_index = pqrst_data['P_Index']

    v_pacemaker = []
    a_pacemaker = []

    qd = int(fs * 0.08)
    for q in q_index:
        _q = q - qd
        aoi1 = ecg_signal[_q:q]
        if aoi1.any():
            peaks1 = np.where(np.min(aoi1) == aoi1)[0][0]
            peaks1 += _q
            if -0.6 <= ecg_signal[peaks1] <= -0.1 and ecg_signal[q] > ecg_signal[peaks1] and abs(ecg_signal[q] - ecg_signal[peaks1]) >= 0.15:
                if np.min(np.abs(r_index - peaks1)) > 14:
                    v_pacemaker.append(peaks1)

    for i in range(len(r_index) - 1):
        aoi = ecg_signal[s_index[i]:q_index[i + 1]]
        if aoi.any():
            check, _ = find_peaks(aoi, prominence=(0.2, 0.3), distance=100, width=(1, 6))
            peaks1 = check + s_index[i]
        else:
            peaks1 = np.array([])

        if peaks1.any():
            if np.min(np.abs(r_index - peaks1)) > 15:
                a_pacemaker.extend(list(peaks1))

    # Remove a_pacemaker if it falls within 20 data points of a v_pacemaker or Atrial_&_Ventricular_pacemaker
    for v_peak in v_pacemaker:
        for k in range(len(a_pacemaker) - 1, -1, -1):
            if abs(a_pacemaker[k] - v_peak) <= 20:
                a_pacemaker.pop(k)

    atrial_per = venti_per = 0

    if len(r_index) != 0:
        atrial_per = round((len(a_pacemaker) / len(r_index)) * 100)
        venti_per = round((len(v_pacemaker) / len(r_index)) * 100)

    if atrial_per > 30 and venti_per > 30:
        pacemaker = np.concatenate((v_pacemaker, a_pacemaker)).astype('int64').tolist()
        pacemaker_per = round((len(a_pacemaker) / len(r_index)) * 100)
        label = "Atrial_&_Ventricular_pacemaker"
    elif atrial_per >= 50 and venti_per >= 50:
        if venti_per > atrial_per:
            label = "Ventricular_Pacemaker"
            pacemaker = v_pacemaker
        else:
            label = "Atrial_Pacemaker"
            pacemaker = a_pacemaker
    elif atrial_per >= 50:
        label = "Atrial_Pacemaker"
        pacemaker = a_pacemaker
    elif venti_per >= 50:
        label = "Ventricular_Pacemaker"
        pacemaker = v_pacemaker
    else:
        label = "False"
        pacemaker = np.array([])

    return label, pacemaker, r_index, q_index, s_index, p_index




def find_r_peakss(ecg_signal,fs=200):
    """Finds R-peaks in an ECG signal using the Hamilton segmenter algorithm.
    Args:
        ecg_signal (array): The ECG signal of numpy array
        fs (int, optional): sampling rate of the ECG signal, defaults to 200 (optional)
    Returns:
        list: the R-peak indices of the ECG signal using the Hamilton QRS complex detector algorithm.
    """
    r_ = []
    out = hamilton_segmenter(signal=ecg_signal, sampling_rate=fs)
    r_index_ = out["rpeaks"]
    heart_rate = hr_count(ecg_signal, r_index_, fs=fs)
    diff_indexs = abs(round((fs * 0.4492537) + (heart_rate * -1.05518351) + 40.40601032654332))
   
    for r in r_index_:
        if r - diff_indexs >= 0 and len(ecg_signal) >= r+diff_indexs:
            data = ecg_signal[r-diff_indexs:r+diff_indexs]
            abs_data = np.abs(data)
            r_.append(np.where(abs_data == max(abs_data))[0][0] + r-diff_indexs)
        else:
            r_.append(r)
    return np.unique(r_) if r_ else r_index_


def PACcounter(PAC_R_Peaks, hr_counts):
    at_counter = 0
    couplet_counter = 0
    triplet_counter = 0
    bigeminy_counter = 0
    trigeminy_counter = 0
    quadrigeminy_counter = 0
    at = 0
    i = 0
    while i < len(PAC_R_Peaks):
        count = 0
        ones_count = 0
        while i < len(PAC_R_Peaks) and PAC_R_Peaks[i] == 1:
            count += 1
            ones_count += 1
            i += 1

        if count >= 4:
            at_counter += 1
            at += ones_count
            count = 0
            ones_count = 0
        if count == 3:
            triplet_counter += 1
        elif count == 2:
            couplet_counter += 1

        i += 1

    j = 0
    while j < len(PAC_R_Peaks) - 1:
        if PAC_R_Peaks[j] == 1:
            k = j + 1
            spaces = 0
            while k < len(PAC_R_Peaks) and PAC_R_Peaks[k] == 0:
                spaces += 1
                k += 1

            if k < len(PAC_R_Peaks) and PAC_R_Peaks[k] == 1:
                if spaces == 1:
                    bigeminy_counter += 1
                elif spaces == 2:
                    trigeminy_counter += 1
                elif spaces == 3:
                    quadrigeminy_counter += 1
            j = k

        else:
            j += 1
    
    total_one = (1*at) + (couplet_counter*2)+ (triplet_counter*3)+ (bigeminy_counter*2)+ (trigeminy_counter*2)+ (quadrigeminy_counter*2)
    total = at_counter + couplet_counter+ triplet_counter+ bigeminy_counter+ trigeminy_counter+ quadrigeminy_counter
    ones = PAC_R_Peaks.count(1)
    if total == 0:
        Isolated = ones
    else:
        Common = total-1
        Isolated = ones - (total_one - Common)
    if hr_counts > 100:
        triplet_counter = couplet_counter = quadrigeminy_counter = trigeminy_counter = bigeminy_counter = Isolated = 0
    if at_counter>=1 and hr_counts > 190:
        at_counter=1
    else:
        at_counter=0

    data = {"PAC-ISO_counter":Isolated,
            "PAC-Bigem_counter":bigeminy_counter,
            "PAC-Trigem_counter":trigeminy_counter,
            "PAC-Quadrigem_counter":quadrigeminy_counter,
            "PAC-Couplet_counter":couplet_counter,
            "PAC-Triplet_counter":triplet_counter,
            "PAC-AT_counter":at_counter}
    return data


def calculate_average(lst):
    if len(lst) == 0:
        return 0  # Handle empty list case

    total_sum = sum(lst)
    average = total_sum / len(lst)
    return average



def wide_qrs_find(q_index, r_index, s_index, hr_count, fs=200):
    """The distance between the QRS complex's Q wave and S wave.

    Args:
        q_index (list): Q waves indices in an ECG signal
        r_index (list): R-peak indices in an ECG signal
        s_index (list): S waves indices in an ECG signal
        fs (int, optional): sampling rate of the ECG signal, defaults to 200 (optional)

    Returns:
        array: the R-peak indices that correspond to wide QRS complexes in an ECG signal.
    The function takes in three arrays as inputs: q_index, r_index, and s_index, which represent the
    indices of the Q-wave onset, R-peak, and S-wave offset, respectively. The function loops through the
    R-peak indices and calculates the difference between
    """
    max_indexs = 0
    # if hr_count<60:
    # print(hr_count,"========hr")
    if hr_count <= 88:
        ms = 0.10
    else:
        ms = 0.12
    max_indexs = int(fs * ms)
    pvc = []
    difference = []
    pvc_index = []
    wide_qs_diff = []
    for k in range(len(r_index)):
        diff = s_index[k] - q_index[k]
        difference.append(diff)
        if max_indexs != 0:
            if diff>=max_indexs:
                pvc.append(r_index[k])

    # print("len_r_index", len(r_index), "wide_r_index", len(pvc))
    if hr_count <= 88:
        wide_r_index_per = len(pvc)/ len(r_index)
        # print(wide_r_index_per,"=========wide_r_index_per")
        if wide_r_index_per < 0.8:
            pvc_index = np.array(pvc)
        else:
            ms = 0.12
            max_indexs = int(fs * ms)
            for k in range(len(r_index)):
                diff = s_index[k] - q_index[k]
                wide_qs_diff.append(diff)
                if max_indexs != 0:
                    if diff>=max_indexs:
                        pvc_index.append(r_index[k])
            difference = wide_qs_diff
    else:
        pvc_index = np.array(pvc) 
    q_s_difference = [i/200 for i in difference]
    return np.array(pvc_index), q_s_difference


def wide_qrs_find_PAC(q_index, r_index, s_index, hr_count, fs=200):
    """The distance between the QRS complex's Q wave and S wave.

    Args:
        q_index (list): Q waves indices in an ECG signal
        r_index (list): R-peak indices in an ECG signal
        s_index (list): S waves indices in an ECG signal
        fs (int, optional): sampling rate of the ECG signal, defaults to 200 (optional)

    Returns:
        array: the R-peak indices that correspond to wide QRS complexes in an ECG signal.
    The function takes in three arrays as inputs: q_index, r_index, and s_index, which represent the
    indices of the Q-wave onset, R-peak, and S-wave offset, respectively. The function loops through the
    R-peak indices and calculates the difference between
    """
    max_indexs = 0
    # if hr_count<60:
    # print(hr_count,"========hr")
    if hr_count <= 88:
        ms = 0.19
    else:
        ms = 0.17
    max_indexs = int(fs * ms)
    pvc = []
    difference = []
    pvc_index = []
    wide_qs_diff = []
    for k in range(len(r_index)):
        diff = s_index[k] - q_index[k]
        difference.append(diff)
        if max_indexs != 0:
            if diff>=max_indexs:
                pvc.append(r_index[k])

    # print("len_r_index", len(r_index), "wide_r_index", len(pvc))
    if hr_count <= 88:
        wide_r_index_per = len(pvc)/ len(r_index)
        # print(wide_r_index_per,"=========wide_r_index_per")
        if wide_r_index_per < 0.8:
            pvc_index = np.array(pvc)
        else:
            ms = 0.12
            max_indexs = int(fs * ms)
            for k in range(len(r_index)):
                diff = s_index[k] - q_index[k]
                wide_qs_diff.append(diff)
                if max_indexs != 0:
                    if diff>=max_indexs:
                        pvc_index.append(r_index[k])
            difference = wide_qs_diff
    else:
        pvc_index = np.array(pvc) 
    q_s_difference = [i/200 for i in difference]
    return np.array(pvc_index), q_s_difference

def pac_model_GPU(ecg_signal, r_index, r_label, hr_counts, wide_qrs, model, fs):
    pac_r = []
    sd, qd = int(fs * 0.115), int(fs * 0.08)
    s_index = find_s_newnew_index(ecg_signal, r_index, sd)
    q_index = find_q_newnew_index(ecg_signal, r_index, qd)
    diff_arr = np.diff(r_index)
    avg = np.mean(diff_arr)
    n = len(r_index)
    lowpass_signal = lowpass(ecg_signal, (fs * -0.00016964) + 0.20821429)
    i = 0
    if hr_counts <= 80:
        pac_thre = 0.10
    else:
        pac_thre = 0.12
    print("wideqrs:",wide_qrs)
##    plot_pac = []
    if r_label == "Irregular":
        for q, _s, s, q_ in zip(q_index[:-1], s_index[:-1], s_index[1:], q_index[1:]):
            if diff_arr[i] < avg-n and r_index[i+1] not in wide_qrs and r_index[i] not in wide_qrs:
            # if diff_arr[i] < avg: 
                data_with_r, data_without_r = lowpass_signal[q:s], lowpass_signal[_s:q_]
               
                if data_with_r.any() and data_without_r.any():
                    image_data1, image_data2 = image_array_new(data_with_r), image_array_new(data_without_r)
                    # image_data = (np.expand_dims(image_data1, axis=0), np.expand_dims(image_data2, axis=0))
                    image_data = (tf.expand_dims(image_data1, axis=0), tf.expand_dims(image_data2, axis=0))
                    image_data = (tf.cast(image_data[0], dtype=tf.float32), tf.cast(image_data[1], dtype=tf.float32))
##                    image_data = (image_data1.reshape((1,) + image_data1.shape).astype('float32'), image_data2.reshape((1,) + image_data2.shape).astype('float32'))
                    model_pred = predict_tflite_model(model, image_data)[0]
                    label_pvc = np.argmax(model_pred)
                    if label_pvc == 0:
##                        diff = s - q_
##                        print(diff/fs,"========diff_irr")
##                        if (diff/fs) > pac_thre:
##                            pac_r.append(0)
##                        else:
##                            plot_pac.append((_s, q_))
                            pac_r.append(1)
                    else:
                        pac_r.append(0)
                    del image_data1, image_data2, image_data, data_with_r, data_without_r
                else:
                    pac_r.append(0)
            else:
                pac_r.append(0)
            i += 1
    else:
        for q, _s, s, q_ in zip(q_index[:-1], s_index[:-1], s_index[1:], q_index[1:]):
            data_with_r, data_without_r = lowpass_signal[q:s], lowpass_signal[_s:q_]
            if data_with_r.any() and data_without_r.any():
                image_data1, image_data2 = image_array_new(data_with_r), image_array_new(data_without_r)
                # image_data = (np.expand_dims(image_data1, axis=0), np.expand_dims(image_data2, axis=0))
                image_data = (tf.expand_dims(image_data1, axis=0), tf.expand_dims(image_data2, axis=0))
                image_data = (tf.cast(image_data[0], dtype=tf.float32), tf.cast(image_data[1], dtype=tf.float32))
##                image_data = (image_data1.reshape((1,) + image_data1.shape).astype('float32'), image_data2.reshape((1,) + image_data2.shape).astype('float32'))
                model_pred = predict_tflite_model(model, image_data)[0]
                label_pvc = np.argmax(model_pred)
                if label_pvc == 0:
##                    diff = s - q_
##                    print(diff,"============regu_diff")
##                    if (diff/fs) > pac_thre:
##                        pac_r.append(0)
##                    else:
                        pac_r.append(1)
                else:
                    pac_r.append(0)
                del image_data1, image_data2, image_data, data_with_r, data_without_r
            else:
                pac_r.append(0)
##    pac_data = PACcounter(pac_r, hr_counts)
##    pac_data['PAC_Union'] = pac_r
##    print("PAC:",pac_r)
    return pac_r

def pac_model_check(baseline_signal, model, fs=200):

    pqrst_data = pqrst_detection(baseline_signal, fs=fs)
    r_label = pqrst_data['R_Label'] 
    r_index = pqrst_data['R_index']
    q_index = pqrst_data['Q_Index']
    s_index = pqrst_data['S_Index']
    j_index = pqrst_data['J_Index']
    p_t = pqrst_data['P_T List']
    pt = pqrst_data['PT PLot']
    hr_counts = pqrst_data['HR_Count']
    t_index = pqrst_data['T_Index'] 
    p_index = pqrst_data['P_Index']
    ex_index = pqrst_data['Ex_Index'] 
    pr_interval = pqrst_data['PR_Interval'] 
    p_label = pqrst_data['P_Label']
    pr_label = pqrst_data['PR_label']
    
    wide_qrs, difference = wide_qrs_find_PAC(q_index, r_index, s_index, hr_counts, fs)
    
##    if r_label == "Irregular" and hr_counts < 125:
##        pac_model_data = pac_model(baseline_signal, r_index, r_label, wide_qrs, model, fs)
##    elif r_label == "Regular" and hr_counts >= 130:
##        pac_model_data = pac_model(baseline_signal, r_index, r_label, wide_qrs, model, fs)
##    else:
##        pac_model_data  = []

    if r_label == "Irregular" and hr_counts < 190:
        pac_model_data = pac_model_GPU(baseline_signal, r_index, r_label, hr_counts, wide_qrs, model, fs)
    elif r_label == "Regular" and hr_counts >= 120:
        pac_model_data = pac_model_GPU(baseline_signal, r_index, r_label, hr_counts, wide_qrs, model, fs)
    else:
        pac_label_model = []

    return pac_model_data


def pac_new_model_check(ecg_signal, model, fs=200):
    if fs != 200:
        ecg_signal = MinMaxScaler(feature_range=(0,4)).fit_transform(ecg_signal.reshape(-1,1)).squeeze()
    if fs == 200:
        baseline_signal = baseline_construction_200(ecg_signal, 101)
    elif fs == 250:
        baseline_signal = baseline_reconstruction_250(ecg_signal, 131)
    elif fs == 360:
        baseline_signal = baseline_construction_200(ecg_signal, 151)
    else:
        baseline_signal = baseline_construction_200(ecg_signal, 101)
    
    return pac_model_check(baseline_signal, model, fs=200)

class FilterSignal:
    def __init__(self, ecg_signal, fs = 200):
        self.ecg_signal = ecg_signal
        self.fs = fs

    def baseline_construction_200(self, kernel_size=131):
        s_corrected = signal.detrend(self.ecg_signal)
        baseline_corrected = s_corrected - signal.medfilt(s_corrected, kernel_size)
        return baseline_corrected

    def baseline_als(self, file, lam, p, niter=10):
        L = len(file)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*file)
            w = p * (file > z) + (1-p) * (file < z)
        return z

    def baseline_construction_250(self, kernel_size=131):
        als_baseline = self.baseline_als(self.ecg_signal, 16**5, 0.01) 
        s_als = self.ecg_signal - als_baseline
        s_corrected = signal.detrend(s_als)
        corrected_baseline = s_corrected - medfilt(s_corrected, kernel_size)
        return corrected_baseline

    def lowpass(self, cutoff=0.3):
        b, a = signal.butter(3, cutoff, btype='lowpass', analog=False)
        low_passed = signal.filtfilt(b, a, self.baseline_signal)
        return low_passed
    
    def get_data(self):
        
        self.ecg_signal = MinMaxScaler(feature_range=(0,4)).fit_transform(self.ecg_signal.reshape(-1,1)).squeeze()
                    
        if self.fs == 200:
            self.baseline_signal = self.baseline_construction_200(kernel_size = 101)
            lowpass_signal = self.lowpass(cutoff = 0.3)
        elif self.fs == 250:
            self.baseline_signal = self.baseline_construction_250(kernel_size = 131)
            lowpass_signal = self.lowpass(cutoff = 0.25)
        elif self.fs == 360:
            self.baseline_signal = self.baseline_construction_200(kernel_size = 151)
            lowpass_signal = self.lowpass(cutoff = 0.2)
        elif self.fs == 1000:
            self.baseline_signal = self.baseline_construction_200(kernel_size = 399)
            lowpass_signal = self.lowpass(cutoff = 0.05)
        elif self.fs == 128:
            self.baseline_signal = self.baseline_construction_200(kernel_size = 101)
            lowpass_signal = self.lowpass(cutoff = 0.5)
            
        return self.baseline_signal, lowpass_signal

class pqrst_detections:
    def __init__(self, ecg_signal, fs=200, thres=0.5, lp_thres=0.2, rr_thres=0.12, width=(5, 50)):
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.thres = thres
        self.lp_thres = lp_thres
        self.rr_thres = rr_thres
        self.width = width
    
    def hamilton_segmenter(self):

        # check inputs
        if self.ecg_signal is None:
            raise TypeError("Please specify an input signal.")

        sampling_rate = float(self.fs)
        length = len(self.ecg_signal)
        dur = length / sampling_rate

        # algorithm parameters
        v1s = int(1.0 * sampling_rate)
        v100ms = int(0.1 * sampling_rate)
        TH_elapsed = np.ceil(0.36 * sampling_rate)
        sm_size = int(0.08 * sampling_rate)
        init_ecg = 4 # seconds for initialization
        if dur < init_ecg:
            init_ecg = int(dur)

        # filtering
        filtered, _, _ = st.filter_signal(
            signal=self.ecg_signal,
            ftype="butter",
            band="lowpass",
            order=4,
            frequency=20.0,
            sampling_rate=sampling_rate,
        )
        filtered, _, _ = st.filter_signal(
            signal=filtered,
            ftype="butter",
            band="highpass",
            order=4,
            frequency=3.0,
            sampling_rate=sampling_rate,
        )

        # diff
        dx = np.abs(np.diff(filtered, 1) * sampling_rate)

        # smoothing
        dx, _ = st.smoother(signal=dx, kernel="hamming", size=sm_size, mirror=True)

        # buffers
        qrspeakbuffer = np.zeros(init_ecg)
        noisepeakbuffer = np.zeros(init_ecg)
        peak_idx_test = np.zeros(init_ecg)
        noise_idx = np.zeros(init_ecg)
        rrinterval = sampling_rate * np.ones(init_ecg)

        a, b = 0, v1s
        all_peaks, _ = st.find_extrema(signal=dx, mode="max")
        for i in range(init_ecg):
            peaks, values = st.find_extrema(signal=dx[a:b], mode="max")
            try:
                ind = np.argmax(values)
            except ValueError:
                pass
            else:
                # peak amplitude
                qrspeakbuffer[i] = values[ind]
                # peak location
                peak_idx_test[i] = peaks[ind] + a

            a += v1s
            b += v1s

        # thresholds
        ANP = np.median(noisepeakbuffer)
        AQRSP = np.median(qrspeakbuffer)
        TH = 0.475
        DT = ANP + TH * (AQRSP - ANP)
        DT_vec = []
        indexqrs = 0
        indexnoise = 0
        indexrr = 0
        npeaks = 0
        offset = 0

        beats = []

        # detection rules
        # 1 - ignore all peaks that precede or follow larger peaks by less than 200ms
        lim = int(np.ceil(0.15 * sampling_rate))
        diff_nr = int(np.ceil(0.045 * sampling_rate))
        bpsi, bpe = offset, 0

        for f in all_peaks:
            DT_vec += [DT]
            # 1 - Checking if f-peak is larger than any peak following or preceding it by less than 200 ms
            peak_cond = np.array(
                (all_peaks > f - lim) * (all_peaks < f + lim) * (all_peaks != f)
            )
            peaks_within = all_peaks[peak_cond]
            if peaks_within.any() and (max(dx[peaks_within]) > dx[f]):
                continue

            # 4 - If the peak is larger than the detection threshold call it a QRS complex, otherwise call it noise
            if dx[f] > DT:
                # 2 - look for both positive and negative slopes in raw signal
                if f < diff_nr:
                    diff_now = np.diff(self.ecg_signal[0 : f + diff_nr])
                elif f + diff_nr >= len(self.ecg_signal):
                    diff_now = np.diff(self.ecg_signal[f - diff_nr : len(dx)])
                else:
                    diff_now = np.diff(self.ecg_signal[f - diff_nr : f + diff_nr])
                diff_signer = diff_now[diff_now > 0]
                if len(diff_signer) == 0 or len(diff_signer) == len(diff_now):
                    continue
                # RR INTERVALS
                if npeaks > 0:
                    # 3 - in here we check point 3 of the Hamilton paper
                    # that is, we check whether our current peak is a valid R-peak.
                    prev_rpeak = beats[npeaks - 1]

                    elapsed = f - prev_rpeak
                    # if the previous peak was within 360 ms interval
                    if elapsed < TH_elapsed:
                        # check current and previous slopes
                        if prev_rpeak < diff_nr:
                            diff_prev = np.diff(self.ecg_signal[0 : prev_rpeak + diff_nr])
                        elif prev_rpeak + diff_nr >= len(self.ecg_signal):
                            diff_prev = np.diff(self.ecg_signal[prev_rpeak - diff_nr : len(dx)])
                        else:
                            diff_prev = np.diff(
                                self.ecg_signal[prev_rpeak - diff_nr : prev_rpeak + diff_nr]
                            )

                        slope_now = max(diff_now)
                        slope_prev = max(diff_prev)

                        if slope_now < 0.5 * slope_prev:
                            # if current slope is smaller than half the previous one, then it is a T-wave
                            continue
                    if dx[f] < 3.0 * np.median(qrspeakbuffer):  # avoid retarded noise peaks
                        beats += [int(f) + bpsi]
                    else:
                        continue

                    if bpe == 0:
                        rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                        indexrr += 1
                        if indexrr == init_ecg:
                            indexrr = 0
                    else:
                        if beats[npeaks] > beats[bpe - 1] + v100ms:
                            rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                            indexrr += 1
                            if indexrr == init_ecg:
                                indexrr = 0

                elif dx[f] < 3.0 * np.median(qrspeakbuffer):
                    beats += [int(f) + bpsi]
                else:
                    continue

                npeaks += 1
                qrspeakbuffer[indexqrs] = dx[f]
                peak_idx_test[indexqrs] = f
                indexqrs += 1
                if indexqrs == init_ecg:
                    indexqrs = 0
            if dx[f] <= DT:
                # 4 - not valid
                # 5 - If no QRS has been detected within 1.5 R-to-R intervals,
                # there was a peak that was larger than half the detection threshold,
                # and the peak followed the preceding detection by at least 360 ms,
                # classify that peak as a QRS complex
                tf = f + bpsi
                # RR interval median
                RRM = np.median(rrinterval)  # initial values are good?

                if len(beats) >= 2:
                    elapsed = tf - beats[npeaks - 1]

                    if elapsed >= 1.5 * RRM and elapsed > TH_elapsed:
                        if dx[f] > 0.5 * DT:
                            beats += [int(f) + offset]
                            # RR INTERVALS
                            if npeaks > 0:
                                rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                                indexrr += 1
                                if indexrr == init_ecg:
                                    indexrr = 0
                            npeaks += 1
                            qrspeakbuffer[indexqrs] = dx[f]
                            peak_idx_test[indexqrs] = f
                            indexqrs += 1
                            if indexqrs == init_ecg:
                                indexqrs = 0
                    else:
                        noisepeakbuffer[indexnoise] = dx[f]
                        noise_idx[indexnoise] = f
                        indexnoise += 1
                        if indexnoise == init_ecg:
                            indexnoise = 0
                else:
                    noisepeakbuffer[indexnoise] = dx[f]
                    noise_idx[indexnoise] = f
                    indexnoise += 1
                    if indexnoise == init_ecg:
                        indexnoise = 0

            # Update Detection Threshold
            ANP = np.median(noisepeakbuffer)
            AQRSP = np.median(qrspeakbuffer)
            DT = ANP + 0.475 * (AQRSP - ANP)

        beats = np.array(beats)

        r_beats = []
        thres_ch = 1
        adjacency = 0.01 * sampling_rate
        for i in beats:
            error = [False, False]
            if i - lim < 0:
                window = self.ecg_signal[0 : i + lim]
                add = 0
            elif i + lim >= length:
                window = self.ecg_signal[i - lim : length]
                add = i - lim
            else:
                window = self.ecg_signal[i - lim : i + lim]
                add = i - lim
            # meanval = np.mean(window)
            w_peaks, _ = st.find_extrema(signal=window, mode="max")
            w_negpeaks, _ = st.find_extrema(signal=window, mode="min")
            zerdiffs = np.where(np.diff(window) == 0)[0]
            w_peaks = np.concatenate((w_peaks, zerdiffs))
            w_negpeaks = np.concatenate((w_negpeaks, zerdiffs))

            pospeaks = sorted(zip(window[w_peaks], w_peaks), reverse=True)
            negpeaks = sorted(zip(window[w_negpeaks], w_negpeaks))
        
            try:
                twopeaks = [pospeaks[0]]
            except IndexError:
                twopeaks = []
            try:
                twonegpeaks = [negpeaks[0]]
            except IndexError:
                twonegpeaks = []

            # getting positive peaks
            for i in range(len(pospeaks) - 1):
                if abs(pospeaks[0][1] - pospeaks[i + 1][1]) > adjacency:
                    twopeaks.append(pospeaks[i + 1])
                    break
            try:
                posdiv = abs(twopeaks[0][0] - twopeaks[1][0])
            except IndexError:
                error[0] = True

            # getting negative peaks
            for i in range(len(negpeaks) - 1):
                if abs(negpeaks[0][1] - negpeaks[i + 1][1]) > adjacency:
                    twonegpeaks.append(negpeaks[i + 1])
                    break
            try:
                negdiv = abs(twonegpeaks[0][0] - twonegpeaks[1][0])
            except IndexError:
                error[1] = True

            # choosing type of R-peak
            n_errors = sum(error)
            try:
                if not n_errors:
                    if posdiv > thres_ch * negdiv:
                        # pos noerr
                        r_beats.append(twopeaks[0][1] + add)
                    else:
                        # neg noerr
                        r_beats.append(twonegpeaks[0][1] + add)
                elif n_errors == 2:
                    if abs(twopeaks[0][1]) > abs(twonegpeaks[0][1]):
                        # pos allerr
                        r_beats.append(twopeaks[0][1] + add)
                    else:
                        # neg allerr
                        r_beats.append(twonegpeaks[0][1] + add)
                elif error[0]:
                    # pos poserr
                    r_beats.append(twopeaks[0][1] + add)
                else:
                    # neg negerr
                    r_beats.append(twonegpeaks[0][1] + add)
            except IndexError:
                continue

        rpeaks = sorted(list(set(r_beats)))
        rpeaks = np.array(rpeaks, dtype="int")

        return utils.ReturnTuple((rpeaks,), ("rpeaks",))

    def hr_count(self):
        cal_sec = round(self.ecg_signal.shape[0]/self.fs)
        if cal_sec != 0:
            hr = round(self.r_index.shape[0]*60/cal_sec)
            return hr
        return 0

    def fir_lowpass_filter(self, data, cutoff, numtaps=21):
        b = firwin(numtaps, cutoff)
        y = signal.convolve(data, b, mode="same")
        return y

    def find_j_index(self):
        j = []
        increment = int(self.fs*0.05)
        for z in range (0,len(self.s_index)):
            data = []
            j_index = self.ecg_signal[self.s_index[z]:self.s_index[z]+increment]
            for k in range (0,len(j_index)):
                data.append(j_index[k])
            max_d = max(data)
            max_id = data.index(max_d)
            j.append(self.s_index[z]+max_id)
        return j

    def find_s_index(self, d):
            d = int(d)+1
            s = []
            for i in self.r_index:
                if i == len(self.ecg_signal):
                    s.append(i)
                    continue
                elif i+d<=len(self.ecg_signal):
                    s_array = self.ecg_signal[i:i+d]
                else:
                    s_array = self.ecg_signal[i:]
                if self.ecg_signal[i] > 0:
                    s_index = i+np.where(s_array == min(s_array))[0][0]
                else:
                    s_index = i+np.where(s_array == max(s_array))[0][0]
                    if abs(s_index - i) < d/2:
                        s_index_ = i+np.where(s_array == min(s_array))[0][0]
                        if abs(s_index_ - i) > d/2:
                            s_index = s_index_
                s.append(s_index)
            return np.sort(s)

    def find_q_index(self, d):
        d = int(d) + 1
        q = []
        for i in self.r_index:
            if i == 0:
                q.append(i)
                continue
            elif 0 <= i - d:
                q_array = self.ecg_signal[i - d:i]
            else:
                q_array = self.ecg_signal[:i]
            if self.ecg_signal[i] > 0:
                q_index = i - (len(q_array) - np.where(q_array == min(q_array))[0][0])
            else:
                q_index = i - (len(q_array) - np.where(q_array == max(q_array))[0][0])
            q.append(q_index)
        return np.sort(q)

    def find_new_q_index(self, d):
        q = []
        for i in self.r_index:
            q_ = []
            if i == 0:
                q.append(i)
                continue
            if self.ecg_signal[i] > 0:
                c = i
                while c > 0 and self.ecg_signal[c-1] < self.ecg_signal[c]:
                    c -= 1                  
                if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == 0:
                    if abs(i-c) <= d:
                        q.append(c)
                        continue
                    else:
                        q_.append(c)
                while c > 0:
                    while c > 0 and self.ecg_signal[c-1] > self.ecg_signal[c]:
                        c -= 1
                    # q_.append(c)
                    while c > 0 and self.ecg_signal[c-1] < self.ecg_signal[c]:
                        c -= 1
                    if q_ and q_[-1] == c:
                        break
                    q_.append(c)
                    if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == 0:
                        break
            else:
                c = i
                while c > 0 and self.ecg_signal[c-1] > self.ecg_signal[c]:
                    c -= 1
                if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == 0:
                    if abs(i-c) <= d:
                        q.append(c)
                        continue
                    else:
                        q_.append(c)
                while c > 0:
                    while c > 0 and self.ecg_signal[c-1] < self.ecg_signal[c]:
                        c -= 1
                    # q_.append(c)
                    while c > 0 and self.ecg_signal[c-1] > self.ecg_signal[c]:
                        c -= 1
                    if q_ and q_[-1] == c:
                        break
                    q_.append(c)
                    if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == 0:
                        break
            if q_:
                a = 0
                for _q in q_[::-1]:
                    if abs(i-_q) <= d:
                        a = 1
                        q.append(_q)
                        break
                if a == 0:
                    q.append(q_[0])
        return np.sort(q)

    def find_new_s_index(self, d):
        s = []
        end_index = len(self.ecg_signal)
        for i in self.r_index:
            s_ = []
            if i == len(self.ecg_signal):
                s.append(i)
                continue
            if self.ecg_signal[i] > 0:
                c = i
                while c+1 < end_index and self.ecg_signal[c+1] < self.ecg_signal[c]:
                    c += 1
                if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == end_index-1:
                    if abs(i-c) <= d:
                        s.append(c)
                        continue
                    else:
                        s_.append(c)
                while c+1 < end_index:
                    while c+1 < end_index and self.ecg_signal[c+1] > self.ecg_signal[c]:
                        c += 1
                    while c+1 < end_index and self.ecg_signal[c+1] < self.ecg_signal[c]:
                        c += 1
                    if s_ and s_[-1] == c:
                        break
                    s_.append(c)
                    if self.ecg_signal[i] * 0.01 > self.ecg_signal[c] or self.ecg_signal[c] < 0 or c == end_index-1:
                        break
            else:
                c = i
                while c+1 < end_index and self.ecg_signal[c+1] > self.ecg_signal[c]:
                    c += 1
                if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == end_index-1:
                    if abs(i-c) <= d:
                        s.append(c)
                        continue
                    else:
                        s_.append(c)
                while c < end_index:
                    while c+1 < end_index and self.ecg_signal[c+1] > self.ecg_signal[c]:
                        c += 1
                    while c+1 < end_index and self.ecg_signal[c+1] < self.ecg_signal[c]:
                        c += 1
                    if s_ and s_[-1] == c:
                        break
                    s_.append(c)
                    if self.ecg_signal[i] * 0.01 < self.ecg_signal[c] or self.ecg_signal[c] > 0 or c == end_index-1:
                        break
            if s_:
                a = 0
                for _s in s_[::-1]:
                    if abs(i-_s) <= d:
                        a = 1
                        s.append(_s)
                        break
                if a == 0:
                    s.append(s_[0])
        return np.sort(s)

    def find_r_peaks(self):
        r_ = []
        out = self.hamilton_segmenter()
        self.r_index = out["rpeaks"]
        heart_rate = self.hr_count()
        diff_indexs = abs(round((self.fs * 0.4492537) + (heart_rate * -1.05518351) + 40.40601032654332))
    
        for r in self.r_index:
            if r - diff_indexs >= 0 and len(self.ecg_signal) >= r+diff_indexs:
                data = self.ecg_signal[r-diff_indexs:r+diff_indexs]
                abs_data = np.abs(data)
                r_.append(np.where(abs_data == max(abs_data))[0][0] + r-diff_indexs)
            else:
                r_.append(r)
            
        new_r = np.unique(r_) if r_ else self.r_index
        fs_diff = int((25*self.fs)/200)
        final_r = []
        if new_r.any(): final_r = [new_r[0]] + [new_r[j+1] for j, i in enumerate(np.diff(new_r)) if i >= fs_diff]
        return np.array(final_r)

    def pt_detection_1(self):
        max_signal = max(self.ecg_signal)/100
        pt = []
        p_t = []
        for i in range (0, len(self.r_index)-1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i+1]]
            max_signal = max(self.ecg_signal)/100
            low = self.fir_lowpass_filter(aoi,self.lp_thres,30)
            if self.ecg_signal[self.r_index[i]]<0:
                max_signal=0.05
            else:
                max_signal=max_signal
            if aoi.any():
                peaks,_ = find_peaks(low,height=max_signal,width=self.width)
                peaks1=peaks+(self.s_index[i])
            else:
                peaks1 = [0]
            p_t.append(list(peaks1))
            pt.extend(list(peaks1))
            for i in range (len(p_t)):
                if not p_t[i]:
                    p_t[i] = [0]
        return pt, p_t

    def pt_detection_2(self):
        pt = []
        p_t = []
        for i in range (0, len(self.r_index)-1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i+1]]
            if aoi.any():
                low = self.fir_lowpass_filter(aoi,self.lp_thres,30)
                if self.ecg_signal[self.r_index[i]]<0:
                    max_signal=0.05
                else:
                    max_signal= max(low)*0.2
                if aoi.any():
                    peaks,_ = find_peaks(low,height=max_signal,width=self.width)
                    peaks1=peaks+(self.s_index[i])
                else:
                    peaks1 = [0]
                p_t.append(list(peaks1))
                pt.extend(list(peaks1))
                for i in range (len(p_t)):
                    if not p_t[i]:
                        p_t[i] = [0]
            else:
                p_t.append([0])
        return pt, p_t

    def pt_detection_3(self):
        pt = []
        p_t = []
        for i in range (0, len(self.r_index)-1):
            aoi = self.ecg_signal[self.s_index[i]:self.q_index[i+1]]
            low = self.fir_lowpass_filter(aoi,self.lp_thres,30)
            if aoi.any():
                peaks,_ = find_peaks(low,prominence=0.05,width=self.width)
                peaks1=peaks+(self.s_index[i])
            else:
                peaks1 = [0]
            p_t.append(list(peaks1))
            pt.extend(list(peaks1))
            for i in range (len(p_t)):
                if not p_t[i]:
                    p_t[i] = [0]

        return pt, p_t

    def pt_detection_4(self):
        def all_peaks_7(arr):
            sign_arr = np.sign(np.diff(arr))
            pos = np.where(np.diff(sign_arr) == -2)[0] + 1
            neg = np.where(np.diff(sign_arr) == 2)[0] + 1
            all_peaks = np.sort(np.concatenate((pos, neg)))
            al = all_peaks.tolist()
            diff = {}
            P, Pa, Pb = [], [], []
            if len(al) > 2:
                for p in pos:
                    index = al.index(p)
                    if index == 0:
                        m, n, o = arr[0], arr[al[index]], arr[al[index+1]]
                    elif index == len(al)-1:
                        m, n, o = arr[al[index-1]], arr[al[index]], arr[-1]
                    else:
                        m, n, o = arr[al[index-1]], arr[al[index]], arr[al[index+1]]
                    diff[p] = [abs(n-m), abs(n-o)]
                th = np.mean([np.mean([v, m]) for v, m in diff.values()])*.66
                for p, (a, b) in diff.items():
                    if a >= th and b >= th:
                        P.append(p)
                        continue
                    if a >= th and not Pa:
                        Pa.append(p)
                    elif a >= th and arr[p] > arr[Pa[-1]] and np.where(pos==Pa[-1])[0]+1 == np.where(pos==p)[0]:
                        Pa[-1] = p
                    elif a >= th:
                        Pa.append(p)
                    if b >= th and not Pb:
                        Pb.append(p)
                    elif b >= th and arr[p] < arr[Pb[-1]] and np.where(pos==Pb[-1])[0]+1 == np.where(pos==p)[0]:
                        Pb[-1] = p
                    elif b >= th:
                        Pb.append(p)
                if len(pos)>1:
                    for i in range(1, len(pos)):
                        m, n = pos[i-1], pos[i]
                        if m in Pa and n in Pb:
                            P.append(m) if arr[m] > arr[n] else P.append(n)
            else:
                P = pos
            return np.sort(P)
        pt, p_t = [], []
        for i in range(1, len(self.r_index)):
            q0, r0, s0 = self.q_index[i - 1], self.r_index[i - 1], self.s_index[i - 1]
            q1, r1, s1 = self.q_index[i], self.r_index[i], self.s_index[i]
            arr = self.ecg_signal[s0+7:q1-7]
            peaks = list(all_peaks_7(arr) + s0 + 7) 
            if peaks:
                pt.extend(peaks)
                p_t.append(peaks)
            else:
                p_t.append([0])
        return pt, p_t

    def find_pt(self):
        _, p_t1 = self.pt_detection_1()
        _, p_t2 = self.pt_detection_2()
        _, p_t3 = self.pt_detection_3()
        _, p_t4 = self.pt_detection_4() 
        pt = []
        p_t = []
        for i in range(len(p_t1)):
            _ = []
            for _pt in set(p_t1[i]+p_t2[i]+p_t3[i]+p_t4[i]):
                count = 0
                if any(val in p_t1[i] for val in range (_pt-2,_pt+3)):
                    count += 1
                if any(val in p_t2[i] for val in range (_pt-2,_pt+3)):
                    count += 1
                if any(val in p_t3[i] for val in range (_pt-2,_pt+3)):
                    count += 1
                if any(val in p_t4[i] for val in range (_pt-2,_pt+3)):
                    count += 1
                if count >= 3:
                    _.append(_pt)
                _.sort()
            if _:
                p_t.append(_)
            else:
                p_t.append([0])
        result = []
        for sublist in p_t:
            # print(sublist)
            temp = [sublist[0]]
            for i in range(1, len(sublist)):
                if abs(sublist[i] - sublist[i-1]) > 5:
                    temp.append(sublist[i])
                else:
                    temp[-1] = sublist[i]  
            if temp:
                result.append(temp)
                pt.extend(temp)
            else:
                result.append([0])
        p_t = result
        return p_t, pt

    def segricate_p_t_pr_inerval(self):
        """
        threshold = 0.37 for JR and 0.5 for other diseases
        """
        diff_arr = ((np.diff(self.r_index)*self.thres)/self.fs).tolist()
        t_peaks_list, p_peaks_list, pr_interval, extra_peaks_list = [], [], [], []
       
        for i in range(len(self.p_t)):
            p_dis = (self.r_index[i+1]-self.p_t[i][-1])/self.fs
            t_dis = (self.r_index[i+1]-self.p_t[i][0])/self.fs
            threshold = diff_arr[i]
            if t_dis > threshold and (self.p_t[i][0]>self.r_index[i]): 
                t_peaks_list.append(self.p_t[i][0])
            else:
                t_peaks_list.append(0)
            if p_dis <= threshold: 
                p_peaks_list.append(self.p_t[i][-1])
                pr_interval.append(p_dis*self.fs)
            else:
                p_peaks_list.append(0)
            if len(self.p_t[i])>0:
                if self.p_t[i][0] in t_peaks_list:
                    if self.p_t[i][-1] in p_peaks_list:
                        extra_peaks_list.extend(self.p_t[i][1:-1])
                    else:
                        extra_peaks_list.extend(self.p_t[i][1:])
                elif self.p_t[i][-1] in p_peaks_list:
                    extra_peaks_list.extend(self.p_t[i][:-1])
                else:
                    extra_peaks_list.extend(self.p_t[i])

        p_label, pr_label = "", ""
        if self.thres >= 0.5 and p_peaks_list and len(p_peaks_list)>2:
            pp_intervals = np.diff(p_peaks_list)
            pp_std = np.std(pp_intervals)
            pp_mean = np.mean(pp_intervals)
            threshold = 0.12 * pp_mean
            if pp_std <= threshold:
                p_label = "Constanat"
            else:
                p_label = "Not Constant"
            
            count=0
            for i in pr_interval:
                if round(np.mean(pr_interval)*0.75) <= i <= round(np.mean(pr_interval)*1.25):
                    count +=1
            if len(pr_interval) != 0: 
                per = count/len(pr_interval)
                pr_label = 'Not Constant' if per<=0.7 else 'Constant'
        data = {'T_Index':t_peaks_list, 
                'P_Index':p_peaks_list, 
                'PR_Interval':pr_interval, 
                'P_Label':p_label, 
                'PR_label':pr_label,
                'Extra_Peaks':extra_peaks_list}
        return data

    def find_inverted_t_peak(self):
        t_index = []
        for i in range(0, len(self.s_index)-1):
            t = self.ecg_signal[self.s_index[i]: self.q_index[i+1]]
            if t.any():
                check, _ = find_peaks(-t,  height=(0.21, 1), distance=70)
                peaks = check + self.s_index[i]
            else:
                peaks = np.array([])
            if peaks.any():
                t_index.extend(list(peaks))
        return t_index

    def get_data(self):
        
        self.r_index = self.find_r_peaks()
        rr_intervals = np.diff(self.r_index)
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        threshold = self.rr_thres * rr_mean
        if rr_std <= threshold:
            self.r_label = "Regular"
        else:
            self.r_label = "Irregular"
        if self.rr_thres == 0.15:
            self.ecg_signal = lowpass(self.ecg_signal,0.2)
        self.hr_ = self.hr_count()
        sd, qd = int(self.fs * 0.115), int(self.fs * 0.08)
        self.s_index = self.find_s_index(sd)
        # q_index = find_q_index(ecg_signal, r_index, qd)
        # s_index = find_new_s_index(ecg_signal,r_index,sd)
        self.q_index = self.find_new_q_index(qd)
        self.j_index = self.find_j_index()
        self.p_t, self.pt = self.find_pt()
        self.data_ = self.segricate_p_t_pr_inerval()
        self.inv_t_index = self.find_inverted_t_peak()
        data = {'R_Label':self.r_label, 
                'R_index':self.r_index, 
                'Q_Index':self.q_index, 
                'S_Index':self.s_index, 
                'J_Index':self.j_index, 
                'P_T List':self.p_t, 
                'PT PLot':self.pt, 
                'HR_Count':self.hr_, 
                'T_Index':self.data_['T_Index'], 
                'P_Index':self.data_['P_Index'],
                'Ex_Index':self.data_['Extra_Peaks'], 
                'PR_Interval':self.data_['PR_Interval'], 
                'P_Label':self.data_['P_Label'], 
                'PR_label':self.data_['PR_label'],
                'inv_t_index': self.inv_t_index}
        return data



class block_detection:
    def __init__(self, ecg_signal, fs):
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.block_processing()

    def block_processing(self):
        self.baseline_signal, self.lowpass_signal = FilterSignal(self.ecg_signal, self.fs).get_data()
        pqrst_data = pqrst_detections(self.baseline_signal, fs=self.fs).get_data()
        self.r_index = pqrst_data["R_index"]
        self.q_index = pqrst_data["Q_Index"]
        self.s_index = pqrst_data["S_Index"]
        self.p_index = pqrst_data["P_Index"]
        self.hr_counts = pqrst_data["HR_Count"]
        self.p_t = pqrst_data["P_T List"]
        self.pr = pqrst_data["PR_Interval"]

    def third_degree_block_deetection(self):
        label= 'Abnormal'
        third_degree = []
        possible_3rd = possible_mob_3rd = False
        if self.hr_counts <= 100 and len(self.p_t) != 0: # 60 70
            constant_2 = all(map(lambda innerlist: len(innerlist) == 2, self.p_t))
            cons_2_1 = all(len(inner_list) in {1, 2} for inner_list in self.p_t)
            ampli_val = list(map(lambda inner_list: sum(self.baseline_signal[i] > 0.05 for i in inner_list) / len(inner_list),self.p_t))
            count_above_threshold = sum(1 for value in ampli_val if value > 0.7)
            percentage_above_threshold = count_above_threshold / len(ampli_val)
            count = 0
            if percentage_above_threshold >= 0.7:
                inc_dec_count = 0
                for i in range(0, len(self.pr)):
                    if self.pr[i] > self.pr[i -1]:
                        inc_dec_count += 1
                if round(inc_dec_count / (len(self.pr)), 2) >= 0.50 and constant_2 == False: # if posibale to change more then 0.5
                    possible_mob_3rd = True
                for inner_list in self.p_t:
                    if len(inner_list) in [3, 4] :
                        ampli_val = [self.baseline_signal[i] for i in inner_list] 
                        if ampli_val  and (sum(value > 0.05 for value in ampli_val) / len(ampli_val)) > 0.7: 
                            differences = np.diff(inner_list).tolist()
                            diff_list = [x for x in differences if x >= 70]
                            if len(diff_list) != 0:
                                third_degree.append(1)
                            else:
                                third_degree.append(0)    
                    elif len(inner_list) in [3,4] and possible_mob_3rd==True:
                        differences = np.diff(inner_list).tolist()
                        if all(diff > 70 for diff in differences):
                            third_degree.append(1)
                        else:
                            third_degree.append(0)
                    else:
                        third_degree.append(0)
        if len(third_degree) != 0:
            if third_degree.count(1) /len(third_degree) >= 0.4 or possible_mob_3rd: # 0.5 0.4   
                label = "3rd Degree block"
        return label    

    def second_degree_block_new(self):
        label= 'Abnormal'
        constant_3_peak = []
        possible_mob_1 = False
        possible_mob_2 = False
        mob_count = 0
        if self.hr_counts <= 100: # 80
            if len(self.p_t) != 0:
                constant_2 = all(map(lambda innerlist: len(innerlist) == 2, self.p_t))
                rhythm_flag = all(len(inner_list) in {1, 2, 3} for inner_list in self.p_t)
                ampli_val = list(map(lambda inner_list: sum(self.baseline_signal[i] > 0.05 for i in inner_list) / len(inner_list), self.p_t))
                count_above_threshold = sum(1 for value in ampli_val if value > 0.7)
                percentage_above_threshold = count_above_threshold / len(ampli_val)
                if percentage_above_threshold >= 0.7:
                    if rhythm_flag and constant_2 == False:
                        pr_interval = []
                        for i, r_element in enumerate(self.r_index[1:], start=1):
                            if i <= len(self.p_t):
                                inner_list = self.p_t[i - 1]  
                                last_element = inner_list[-1] 
                                result = r_element - last_element 
                                pr_interval.append(result)

                        counts = {}
                        count_2 = 0
                        for i in range(0, len(pr_interval)):
                            counts[i] = 1
                            if i in counts:
                                counts[i] += 1
                            if pr_interval[i] > pr_interval[i -1]:
                                count_2 += 1
                        most_frequent = max(counts.values())
                        if round(count_2 / (len(pr_interval)), 2) >= 0.50: 
                            possible_mob_1 = True
                        elif round(most_frequent / len(pr_interval), 2) >= 0.4: 
                            possible_mob_2 = True

                        for inner_list in self.p_t:
                            if len(inner_list) == 3 :
                                differences = np.diff(inner_list).tolist()
                                if differences[0] <= 0.5 * differences[1] or differences[1] <= 0.5 * differences[0]:
                                    if possible_mob_1 or possible_mob_2:
                                        mob_count += 1
                                    else:
                                        constant_3_peak.append(1)
                            else:
                                constant_3_peak.append(0)
                    else:
                        for inner_list in self.p_t:
                            if len(inner_list) == 3 :
                                differences = np.diff(inner_list).tolist()
                                if differences[0] <= 0.5 * differences[1] or differences[1] <= 0.5 * differences[0]:
                                    constant_3_peak.append(1)
                                else:
                                    constant_3_peak.append(0)
                            else:
                                constant_3_peak.append(0)
        if len(constant_3_peak) != 0 and constant_3_peak.count(1) != 0:

            if constant_3_peak.count(1) /len(constant_3_peak) >= 0.4: # 0.4 0.5
                label = "Mobitz II"
        elif possible_mob_1 and mob_count > 1: # 0 1 4
            label = "Mobitz I"
        elif possible_mob_2 and mob_count > 1: # 0  4
            label = "Mobitz II"
        return label

    # Block new trans model for added 
    def prediction_model_block(self, input_arr):
        classes = ['1st_deg', '2nd_deg', '3rd_deg', 'abnormal', 'normal']
        input_arr = tf.io.decode_jpeg(tf.io.read_file(input_arr), channels=3)
        input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
        input_arr = (tf.expand_dims(input_arr, axis=0),)
        model_pred = predict_tflite_model(block_load_model, input_arr )[0]
        # print(model_pred)
        idx = np.argmax(model_pred)
        return model_pred, classes[idx]

    def check_block_model(self, low_ecg_signal):
        label = 'Abnormal'
        for i in glob.glob('temp_block_img' + "/*.jpg"):
            os.remove(i)
        
        randome_number = random.randint(200000, 1000000)
        temp_img = low_ecg_signal
        plt.figure(layout="constrained")
        plt.plot(temp_img)
        plt.axis("off")
        plt.savefig(f"temp_block_img/p_{randome_number}.jpg")
        aq = cv2.imread(f"temp_block_img/p_{randome_number}.jpg")
        aq = cv2.resize(aq, (1080, 460))
        cv2.imwrite(f"temp_block_img/p_{randome_number}.jpg", aq)
        plt.close()
        ei_ti_label = []
        files = sorted(glob.glob("temp_block_img/*.jpg"), key=extract_number)
        for pvcfilename in files:
            predictions, ids = self.prediction_model_block(pvcfilename)
            # print(predictions, ids)
            label = "Abnormal" #"Normal"
            if str(ids) == "3rd_deg" and float(predictions[2]) > 0.78:
                label = "3rd degree"
            if str(ids) == "2nd_deg" and float(predictions[1]) > 0.78:
                label = "2nd degree"
            if str(ids) == "1st_deg" and float(predictions[0]) > 0.78:
                label = "1st degree"

            if 0.40 < float(predictions[1]) < 0.70:
                ei_ti_label.append('2nd degree')
            if 0.40 < float(predictions[0]) < 0.70:
                ei_ti_label.append('1st degree')
            if 0.40 < float(predictions[2]) < 0.70:
                ei_ti_label.append('3rd degree')
        return label, ei_ti_label, predictions

# Block new trans model, need to add 80/20 approach
def block_model_check(ecg_signal, frequency, abs_result):
    model_label = 'Abnormal'
    ei_ti_block = []
    baseline_signal = baseline_construction_200(ecg_signal)
    lowpass_signal = lowpass(baseline_signal)
    get_block = block_detection(ecg_signal, frequency)
     
    block_result, ei_ti_label, model_pre = get_block.check_block_model(lowpass_signal)
    if block_result == '1st degree' and abs_result != 'Abnormal':
        model_label = 'I DEGREE'
    if block_result == '2nd degree' and (abs_result == '' or abs_result == 'Mobitz II'):
        if abs_result=="Mobitz I":
            model_label = 'MOBITZ-I'
        if abs_result=="Mobitz II":
            model_label = 'MOBITZ-II'
    if block_result == '3rd degree' and abs_result!="Abnormal":
        model_label = 'III Degree'
    if ei_ti_label:
        if '1st degree' in ei_ti_label and abs_result!="Abnormal":
            model_label = 'I DEGREE'
            ei_ti_block.append({"Arrhythmia":"I DEGREE","percentage":model_pre[0]*100})
        if '2nd degree' in ei_ti_label and (abs_result == 'Mobitz I' or abs_result == 'Mobitz II'):
            if abs_result=="Mobitz I":
                model_label = 'MOBITZ-I'
                ei_ti_block.append({"Arrhythmia":"MOBITZ-I","percentage":model_pre[1]*100})
            if abs_result=="Mobitz II":
                model_label = 'MOBITZ-II'
                ei_ti_block.append({"Arrhythmia":"MOBITZ-II","percentage":model_pre[1]*100})
        if '3rd degree' in ei_ti_label and abs_result!="Abnormal":
            model_label = 'III Degree'
            ei_ti_block.append({"Arrhythmia":"III Degree","percentage":model_pre[2]*100})
    return model_label, ei_ti_block

def block_process(ecg_signal, frequency):
    abs_result = 'Abnormal'
    get_block = block_detection(ecg_signal, frequency)
    second_deg_check = get_block.second_degree_block_new()
    if second_deg_check != 'Abnormal':
        abs_result = second_deg_check
    if second_deg_check == 'Abnormal':
        third_deg_check = get_block.third_degree_block_deetection()
        abs_result = third_deg_check
    return abs_result

class RPeakDetection:
    def __init__(self, baseline_signal, fs = 200):
        self.baseline_signal = baseline_signal
        self.fs = fs
        

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def find_r_peak(self):
        lowcut = 0.5
        highcut = 50.0
        filtered_signal = self.butter_bandpass_filter(self.baseline_signal, lowcut, highcut, self.fs, order=6)
        # Step 3: R-Peak Detection
        out = hami.hamilton_segmenter(filtered_signal, sampling_rate=self.fs)
        rpeaks = hami.correct_rpeaks(filtered_signal, out[0], sampling_rate=self.fs, tol=0.1)
        r_peaks = rpeaks[0].tolist()
        return r_peaks

class afib_flutter_detection:
    def __init__(self, ecg_signal, r_index, q_index, s_index,p_index, p_t,pr_interval, load_model):
        self.ecg_signal = ecg_signal
        self.r_index = r_index
        self.q_index = q_index
        self.s_index = s_index
        self.p_index = p_index
        self.p_t = p_t
        self.pr_inter = pr_interval
        self.load_model = load_model

    def image_array_new(self, signal, scale=25):
        scales = np.arange(1, scale, 1)
        coef, freqs = pywt.cwt(signal, scales, 'gaus6')
        # coef, freqs = pywt.cwt(signal, scales, wavelet_name)
        abs_coef = np.abs(coef)
        y_scale = abs_coef.shape[0] / 224
        x_scale = abs_coef.shape[1] / 224
        x_indices = np.arange(224) * x_scale
        y_indices = np.arange(224) * y_scale
        x, y = np.meshgrid(x_indices, y_indices, indexing='ij')
        x = x.astype(int)
        y = y.astype(int)
        rescaled_coef = abs_coef[y, x]
        min_val = np.min(rescaled_coef)
        max_val = np.max(rescaled_coef)
        normalized_coef = (rescaled_coef - min_val) / (max_val - min_val)
        cmap_indices = (normalized_coef * 256).astype(np.uint8)
        cmap = colormaps.get_cmap('viridis')
        rgb_values = cmap(cmap_indices)
        image = rgb_values.reshape((224, 224, 4))[:, :, :3]
        denormalized_image = (image * 254) + 1
        rotated_image = np.rot90(denormalized_image, k=1, axes=(1, 0))
        return rotated_image.astype(np.uint8)

    def abs_afib_flutter_check(self):
        R_label = "Regular"
        check_afib_flutter = False
        rpeak_diff = np.diff(self.r_index)
        more_then_3_rhythm_per = len(list(filter(lambda x: len(x) > 3, self.p_t))) / len(self.r_index)
        inner_list_less_2 = len(list(filter(lambda x: len(x) < 2, self.p_t))) / len(self.r_index)

        zeros_count = self.p_index.count(0)
        list_per = zeros_count / len(self.p_index)
        pr_int = [round(num, 2) for num in self.pr_inter]

        constant_list = []
        if len(pr_int) >1:
            for i in range(len(pr_int) - 1):
                diff = abs(pr_int[i] - pr_int[i + 1])
                if diff == 0 or diff == 1:
                    constant_list.append(pr_int[i]) 
            
            if abs(pr_int[-1] - pr_int[-2]) == 0 or abs(pr_int[-1] - pr_int[-2]) == 1:
                constant_list.append(pr_int[-1])


        if more_then_3_rhythm_per >= 0.6:
            check_afib_flutter = True
        elif list_per >= 0.5:
            check_afib_flutter = True
        elif len(constant_list) != 0:
            if (len(constant_list) / len(pr_int) < 0.7):
                check_afib_flutter = True
        else:
            p_peak_diff = np.diff(self.p_index)
            percentage_diff = np.abs(np.diff(p_peak_diff) / p_peak_diff[:-1]) * 100

            mean_p = np.mean(percentage_diff)
            if mean_p != mean_p or mean_p == float('inf') or mean_p == float('-inf'):
                check_afib_flutter = True
            if (mean_p > 15 and more_then_3_rhythm_per >= 0.4) or (mean_p > 70 and inner_list_less_2 > 0.3):
                check_afib_flutter = True
            elif mean_p > 100 and inner_list_less_2 > 0.3:
                check_afib_flutter = True
            elif (mean_p > 20 and more_then_3_rhythm_per >= 0.1):
                check_afib_flutter = True

        return check_afib_flutter

    def predict_tflite_model(self, model:tuple, input_data:tuple):
        with results_lock:
            interpreter, input_details, output_details = model
            for i in range(len(input_data)):
                interpreter.set_tensor(input_details[i]['index'], input_data[i])
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
        
        return output
    
    def check_model(self, q_new, s_new, ecg_signal, last_s, last_q, newsublist):
        percent = {'ABNORMAL': 0, 'AFIB': 0, 'FLUTTER': 0, 'NOISE': 0, 'NORMAL': 0}
        total_data = len(self.s_index) - 1
        afib_data_index, flutter_data_index = [], []
        epoch_afib_index = []
        raw_date_time = newsublist["DateTime"]
        for q, s in zip(q_new, s_new):
            data = ecg_signal[q:s]
            if data.any():
                image_data = self.image_array_new(data)
                image_data = (tf.expand_dims(image_data.astype(np.float32), axis=0),)
                model_pred = self.predict_tflite_model(self.load_model, image_data)[0]
                # print(model_pred)

                model_idx = np.argmax(model_pred)
                if model_idx == 0:
                    if last_s and s > last_s[0]:
                        percent['ABNORMAL'] += last_s[1] / total_data
                    else:
                        percent['ABNORMAL'] += 4 / total_data
                elif model_idx == 1:
                    if last_s and s > last_s[0]:
                        percent['AFIB'] += last_s[1] / total_data
                        q_start_time = raw_date_time[last_q]
                        s_end_time = raw_date_time[s]
                        afib_data_index.append({"StartTime": int(q_start_time),
                                         "EndTime": int(s_end_time)})
                    else:
                        percent['AFIB'] += 4 / total_data
                        q_start_time = raw_date_time[q]
                        s_end_time = raw_date_time[s]
                        afib_data_index.append({"StartTime": int(q_start_time),
                                         "EndTime": int(s_end_time)})
                elif model_idx == 2:
                    if last_s and s > last_s[0]:
                        percent['FLUTTER'] += last_s[1] / total_data
                        q_start_time = raw_date_time[last_q]
                        s_end_time = raw_date_time[s]
                        afib_data_index.append({"StartTime": int(q_start_time),
                                         "EndTime": int(s_end_time)})
                    else:
                        percent['FLUTTER'] += 4 / total_data
                        q_start_time = raw_date_time[q]
                        s_end_time = raw_date_time[s]
                        afib_data_index.append({"StartTime": int(q_start_time),
                                         "EndTime": int(s_end_time)})
                elif model_idx == 3:
                    if last_s and s > last_s[0]:
                        percent['NOISE'] += last_s[1] / total_data
                    else:
                        percent['NOISE'] += 4 / total_data
                elif model_idx == 4:
                    if last_s and s > last_s[0]:
                        percent['NORMAL'] += last_s[1] / total_data
                    else:
                        percent['NORMAL'] += 4 / total_data

        return percent, afib_data_index, flutter_data_index


    def get_data(self, newsublist):
        total_data = len(self.s_index)-1
        last_s = None
        last_q = None
        check_2nd_lead = {}
        afib_data_index = []
        if len(self.q_index) > 4 and len(self.s_index) > 4:
            q_new  = self.q_index[:-4:4].tolist()
            s_new = self.s_index[4::4].tolist()
            if s_new[-1] != self.s_index[-1]:
                temp_s = list(self.s_index).index(s_new[-1])
                fin_s = total_data - temp_s
                last_q = self.q_index[temp_s]
                last_s = (s_new[-1], fin_s)
                q_new.append(self.q_index[-5])
                s_new.append(self.s_index[-1])
            check_2nd_lead,afib_data_index, flutter_data_index = self.check_model(q_new, s_new, self.ecg_signal, last_s,last_q, newsublist)         
        return check_2nd_lead,afib_data_index, flutter_data_index

##@time_wrapper

def afib_fultter_model_check(ecg_signal, load_model,  frequency, newsublist):
    baseline_signal, lowpass_signal = FilterSignal(ecg_signal, frequency).get_data()

    pqrst_data = pqrst_detections(baseline_signal, fs=frequency).get_data()
    r_label = pqrst_data['R_Label']
    r_index = pqrst_data['R_index']
    q_index = pqrst_data['Q_Index']
    s_index = pqrst_data['S_Index']
    p_t = pqrst_data['P_T List']
    pr_interval = pqrst_data['PR_Interval']
    p_index = pqrst_data['P_Index']
    afib_per = 0
    flutter_per = 0
    final_perc = 0 
    afib_flutter = afib_flutter_detection(lowpass_signal, r_index, q_index, s_index, p_index, p_t,pr_interval, load_model)
    abs_afib_flutter_check = afib_flutter.abs_afib_flutter_check()
    ei_ti = []
    label = "Abnormal"
    if abs_afib_flutter_check:
        check_2nd_lead,afib_data_index, flutter_data_index = afib_flutter_detection(lowpass_signal, r_index, q_index, s_index, p_index, p_t,pr_interval, load_model).get_data(newsublist)
        
        afib_per = int(check_2nd_lead['AFIB']*100)
        flutter_per = int(check_2nd_lead['FLUTTER']*100)
        
        if afib_per>=70:
            label = "Afib"
            final_perc = afib_per
        if afib_per>55 and afib_per<70:
            for afib_data in afib_data_index:
                ei_ti.append({"Arrhythmia":"AFIB","startTime":afib_data["StartTime"],"endTime":afib_data["EndTime"],"percentage":afib_per})
        if flutter_per>70:
            label = "Aflutter"
            final_perc = flutter_per
        if flutter_per>55 and flutter_per<70:
            for flutter_data in flutter_data_index:
                ei_ti.append({"Arrhythmia":"AFL","startTime":flutter_data["StartTime"],"endTime":flutter_data["EndTime"],"percentage":flutter_per})
#        else:
#            label = "Abnormal"
            
        return label,ei_ti
    else:
        label = "Abnormal"
        return label,ei_ti
    
def noise_check_again(ecg_signal):
        base1 = signal.detrend(ecg_signal)
        medfilt1 = signal.medfilt(base1,101)
        outputss = np.where(abs(medfilt1)>0.40, 1, 0)
        final_label = 'high_noise_recovery' if outputss.any() else 'Normal'
        return final_label



def sorting_key(filename):
    match = re.search(r'p_(\d+).jpg', filename)
    if match:
        return int(match.group(1))
    return float('inf')  # If there is no match, place it at the end

def position_management(patient,x,y,z):
            def ava(lst):
                return sum(lst) / len(lst)

            maindata = []
            avg = ava(x)
            maindata.append(max(x))
            dictpatient = {}
            try:
                if dictpatient[patient]=="":
                    dictpatient.update({patient:""})
            except:
                dictpatient.update({patient:""})

            xdiff = np.diff(x)
            ydiff = np.diff(y)
            zdiff = np.diff(z)

            xdiffavg = ava(xdiff)
            ydiffavg = ava(ydiff)
            zdiffavg = ava(zdiff)

            if len([*filter(lambda x: x >= 32767, x)]) > 0 and abs(max(x)-min(x))>30000:
                vara = "RUNNING"
                dictpatient.update({patient:6})
                
            elif len([*filter(lambda x: x >= 32767, x)]) > 0:
                vara = "RUNNING"
                dictpatient.update({patient:6})
                
            elif len([*filter(lambda x: x > 21000,x)])>=2 and len([*filter(lambda x: x <= 40000, x)])>0:
                vara = "RUNNING"
                dictpatient.update({patient:6})
                
            elif ((abs(max(x)-min(x))>2100 and abs(max(x)-min(x))<9900)) and ((abs(max(y)-min(y))>2100 and abs(max(y)-min(y))<9900)) and ((abs(max(z)-min(z))>2100 and abs(max(z)-min(z))<9900)):
                    vara = "WALKING"       
                    dictpatient.update({patient:5})
                    

            elif any(value >= 15000 or value <= -13000 for value in z):
                    
                if len([*filter(lambda y: y >= 5000, y)])>=2 and len([*filter(lambda y: y <= 16000, y)])>0:
                    vara = "LAYDOWN_RIGHT"
                    dictpatient.update({patient:4})
                    

                elif len([*filter(lambda y: y <=-7000, y)])>=2 and len([*filter(lambda y: y >=-15900, y)])>0:
                    vara = "LAYDOWN_LEFT"
                    dictpatient.update({patient:3})
                    
                elif ava(z)>9400 and len([*filter(lambda x: x <= 14000, x)]) > 0:
                    vara = "LAYDOWN_DOWNWARD"
                    dictpatient.update({patient:9})

                else:
                    mpuData = 1
                    vara = "LAYDOWN"
                    dictpatient.update({patient:1})
            elif any(value >= 13000 or value <= -13000 for value in x):
                if len([*filter(lambda y: y <=-7000, y)])>=2 and len([*filter(lambda y: y >=-15900, y)])>0:
                    vara = "LAYDOWN_LEFT"
                    dictpatient.update({patient:3})

                elif len([*filter(lambda y: y >= 5000, y)])>=2 and len([*filter(lambda y: y <= 16000, y)])>0:
                    vara = "LAYDOWN_RIGHT"
                    dictpatient.update({patient:4})
                    
                elif ava(z)>9400 and len([*filter(lambda x: x <= 14000, x)]) > 0:
                    vara = "LAYDOWN_DOWNWARD"
                    dictpatient.update({patient:9})
                    

                else:
                    vara = "STAND_SIT"
                    mpuData = 2
                    dictpatient.update({patient:2})
                   

            if ava(z)>9400 and len([*filter(lambda x: x <= 14000, x)]) > 0:
                vara = "LAYDOWN_DOWNWARD"
                dictpatient.update({patient:9})

                
            elif len([*filter(lambda y: y <=-7000, y)])>=2 and len([*filter(lambda y: y >=-15900, y)])>0:
                vara = "LAYDOWN_LEFT"
                dictpatient.update({patient:3})
                

            elif len([*filter(lambda y: y <= 0, y)])>=2 and len([*filter(lambda y: y <= 5000, y)])>1:
                if len([*filter(lambda x: x >= 0, x)])>2 and len([*filter(lambda y: y <= 0, y)])>2 and len([*filter(lambda z: z <= 0, z)])>2 and len([*filter(lambda x: x <= 9000, x)])>2:
                        pass

                else:
                    if len([*filter(lambda x: x <= 0, x)])>2 and len([*filter(lambda y: y <= 0, y)])>2 and len([*filter(lambda z: z <= 0, z)])>2 :

                        vara = "LAYDOWN"
                        dictpatient.update({patient:1})
                       
                    elif len([*filter(lambda x: x <= 0, x)])>2 and len([*filter(lambda y: y >= 0, y)])>2 and len([*filter(lambda z: z <= 0, z)])>2 :

                        vara = "LAYDOWN"
                        dictpatient.update({patient:1})
                       
                    elif ((abs(max(x)-min(x))>2100 and abs(max(x)-min(x))<9900)) and ((abs(max(y)-min(y))>2100 and abs(max(y)-min(y))<9900)) and ((abs(max(z)-min(z))>2100 and abs(max(z)-min(z))<9900)):
                            vara = "WALKING3"       
                            dictpatient.update({patient:5})
                            


                    elif len([*filter(lambda x: x >= 32767, x)]) > 0 and abs(max(x)-min(x))>30000:
                        vara = "RUNNING"
                        dictpatient.update({patient:6})
                        
                    elif len([*filter(lambda x: x >= 32767, x)]) > 0:
                        vara = "RUNNING"
                        dictpatient.update({patient:6})
                       
                    elif len([*filter(lambda x: x > 21000,x)])>=2 and len([*filter(lambda x: x <= 40000, x)])>0:
                        vara = "RUNNING"
                        dictpatient.update({patient:6})
                        

                    else:
                        vara = "STAND_SIT1"
                        dictpatient.update({patient:2})

            elif len([*filter(lambda x: x >= 16900, x)])>1 and len([*filter(lambda x: x < 24000, x)])>0:
                if len([*filter(lambda x: x <= 0, x)])>2 and len([*filter(lambda y: y >= 0, y)])>2 and len([*filter(lambda z: z <= 0, z)])>2 :
                    vara = "LAYDOWN4"
                    dictpatient.update({patient:1})
                   


                elif len([*filter(lambda y: y <= 0, y)])>=2 and ava(x)>15500 and len([*filter(lambda z: z <= 0, z)])>3:
                    vara = "WALKING2"
                    dictpatient.update({patient:5})

            else:
                if len([*filter(lambda x: x >= 17000, x)])>len([*filter(lambda x: x < 16500, x)]):
                    walking=1
                    idle=0
                else:
                    idle=1
                    walking=0
                if len([*filter(lambda x: x < 16800, maindata)])>0 and idle==1:
                    
                    if len([*filter(lambda y: y >= -1000, y)])>=2 and len([*filter(lambda y: y <= 5000, y)])>0:
                        if len([*filter(lambda x: x >= 0, x)])>2 and len([*filter(lambda y: y >= 0, y)])>2 and len([*filter(lambda z: z <= -15000, z)])>2 :
                            vara = "LAYDOWN"
                            dictpatient.update({patient:1})

                        elif len([*filter(lambda x: x >= 0, x)])>2 and len([*filter(lambda y: y >= 0, y)])>2 and len([*filter(lambda z: z <= 0, z)])>2:
                            vara = "STAND_SIT"
                            dictpatient.update({patient:2})
                           

                    
                    elif len([*filter(lambda y: y >= 5000, y)])>=2 and len([*filter(lambda y: y <= 16000, y)])>0:
                        vara = "LAYDOWN_RIGHT"
                        dictpatient.update({patient:4})
                       

                    elif len([*filter(lambda y: y <=-7000, y)])>=2 and len([*filter(lambda y: y >=-15900, y)])>0:
                        vara = "LAYDOWN_LEFT"
                        dictpatient.update({patient:3})
                        

                    elif len([*filter(lambda z: z <= -7000, z)])>2 and len([*filter(lambda z: z >= -17000, z)])>2 :
                        vara = "LAYDOWN"

                        dictpatient.update({patient:1})

            return dictpatient            


def LBBB_RBBB(ecg,rpeaks,imageresource):
        lis=[]
        ecg = pd.DataFrame(ecg)
        count = 1
        for i in glob.glob(imageresource+"/*.jpg"):
            os.remove(i)

        for i in rpeaks:
            lis.append(i)

            if i == rpeaks[0]:
                count += 1
                lis.append(i)
                window_start = int(lis[0]) - 16
                window_end = int(lis[0]) + 90
            elif i == rpeaks[1]:
                count += 1
                lis.append(i)
                window_start = int(lis[0]) - 50
                window_end = int(lis[0]) + 120
            else:
                count += 1
                lis.append(i)
                window_start = int(lis[0]) - 50
                window_end = int(lis[0]) + 80

            aa = pd.DataFrame(ecg.iloc[window_start:window_end])
            plt.plot(aa)
            plt.axis("off")
            plt.savefig(f"{imageresource}/p_{int(lis[0])}.jpg")            
            aq = cv2.imread(f"{imageresource}/p_{int(lis[0])}.jpg")
            aq = cv2.resize(aq, (360, 720))
            cv2.imwrite(f"{imageresource}/p_{int(lis[0])}.jpg", aq)
            lis.clear()
            plt.close()


        LBBB=[]
        RBBB=[]
        files = sorted(glob.glob(imageresource+"/*.jpg"), key=sorting_key)
        for pvcfilename in files:
            predictions,ids = prediction_model(pvcfilename)
##            print(predictions,ids)

            if str(ids) == "LBBB" and float(predictions[0])>0.78:
                LBBB.append(1)
            else:
                LBBB.append(0)

            if str(ids) == "RBBB" and float(predictions[4])>0.78:
                RBBB.append(1)
            else:
                RBBB.append(0)


        if LBBB.count(1)>5:
            label = "LBBB"
            return label
        elif RBBB.count(1)>5:
            label = "RBBB"
            return label
        else:
            label = "Normal"
            return label


def BPM(rpeaks):
        rr_intervals = np.diff(rpeaks)
        hrv_diff = abs(np.diff(rr_intervals)).tolist()
        mean_rr = np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        cv = round((sdnn / mean_rr) * 100,2)  
        if np.isnan(cv):
                HRV = []
        else:
                HRV = hrv_diff
        return round(mean_rr/10), HRV

# ----------------------------------- MI detection model base -----------------------------------------------

def prediction_model_mi(input_arr):
    classes = ['Noise', 'Normal', 'STDEP', 'STELE', 'TAB']
    input_arr = tf.io.decode_jpeg(tf.io.read_file(input_arr), channels=3)
    input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
    input_arr = (tf.expand_dims(input_arr, axis=0),)
    model_pred = predict_tflite_model(let_inf_moedel, input_arr)[0]
    # print(model_pred)
    idx = np.argmax(model_pred)
    return model_pred, classes[idx]

def extract_number(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else float('inf')

def check_st_model(low_ecg_signal, imageresource):
    label = 'Abnormal'
    for i in glob.glob(imageresource + "/*.jpg"):
        os.remove(i)
    try:
        randome_number = random.randint(200000, 1000000)
        plt.plot(low_ecg_signal)
        plt.axis("off")
        plt.savefig(f"{imageresource}/p_{randome_number}.jpg")
        aq = cv2.imread(f"{imageresource}/p_{randome_number}.jpg")
        aq = cv2.resize(aq, (1080, 460))
        cv2.imwrite(f"{imageresource}/p_{randome_number}.jpg", aq)
        plt.close()
        stdep = []
        stele = []
        t_abn = []
        stdeplist = []
        stelelist = []
        t_abn_list = []
        label = 'Abnormal'

        files = sorted(glob.glob(imageresource +"/*.jpg"), key=extract_number)
        for pvcfilename in files:
        
            predictions, ids = prediction_model_mi(pvcfilename)
            label = "Normal"
            if str(ids) == "STDEP" and float(predictions[2]) > 0.91:
                stdep.append(1)
                label = "STDEP"
                stdeplist.append(int(pvcfilename.split("_")[1].split(".jpg")[0]))
            else:
                stdep.append(0)

            if str(ids) == "STELE" and float(predictions[3]) > 0.78:
                stele.append(1)
                label = "STELE"
                stelelist.append(int(pvcfilename.split("_")[1].split(".jpg")[0]))
            else:
                stele.append(0)

            if str(ids) == "TAB" and float(predictions[4]) > 0.78:
                label = "TAB"
                t_abn.append(1)
                t_abn_list.append(int(pvcfilename.split("_")[1].split(".jpg")[0]))
            else:
                t_abn.append(0)

        return label
    except Exception as e:
        print('Inf_lat_error', e)
        return label

def check_mi_model(all_leads_data, imageresource):
    print("-------------------- MI detection -------------------------")
    mi_result = 'Abnormal'
    all_lead_det_data = {}
    t_abn = []
    for lead in all_leads_data.columns:
        lead_data = {}
        if lead in ['II', 'III', 'aVF', 'I', 'aVL', 'v5']:
            ecg_signal = all_leads_data[lead].values
            baseline_signal = baseline_construction_200(ecg_signal)
            lowpass_signal = lowpass(baseline_signal)
            mi_result = check_st_model(lowpass_signal, imageresource)
            lead_data['mi_result'] = mi_result
            all_lead_det_data[lead] = lead_data
            t_abn.append(mi_result)
    if all_lead_det_data['II'] == 'STDEP' and all_lead_det_data['III'] == 'STDEP' and all_lead_det_data['aVF'] == 'STDEP':
        mi_result = 'Inferior STEMI'
    if all_lead_det_data['I'] == 'STDEP' and all_lead_det_data['aVL'] == 'STDEP' and all_lead_det_data['v5'] == 'STDEP':
        mi_result = 'Lateral STEMI'
    flat_list = []
    for element in t_abn:
        if isinstance(element, list):
            flat_list.extend(element)
        else:
            flat_list.append(element)
    counts = Counter(flat_list)
    repeated_elements = [item for item, count in counts.items() if count >= 4]
    mi_t_lab = ' '.join(repeated_elements)
    if mi_t_lab == "TAB" and mi_result != "Inferior STEMI" and mi_result != "Lateral STEMI":
        mi_result = "T_wave_Abnormality"
    return mi_result

def data_convert_MI(sorted_data): # patient_id, path=None
    A = pd.DataFrame(sorted_data)[['dateTime', 'data']]
    try:
        tog_arr = 0
        temp_A = pd.DataFrame(sorted_data)[['data1', 'data5']]
        if temp_A['data1'][0]=="": tog_arr = 1
        A1 = temp_A['data1'].to_list()
        A5 = temp_A['data5'].to_list()
    except Exception as e:
        tog_arr = 1
    A2 = A['data'].to_list()
    
    l, l1, l5 = [], [], []
    data_dict = {"I":[],"II":[],"III":[],"aVR":[],"aVL":[],"aVF":[],"v5":[]} # "DateTime":[],
    if tog_arr == 0:
        for i in range(len(A2)):
            # d = A["dateTime"][i]
            d = datetime.datetime.fromtimestamp(A["dateTime"][i]/1000)
            # print(d)
            start = 0
            stop = 4
            kk= 0
            while True:
                l = A2[i][start:stop]
                l1 = A1[i][start:stop]
                l5 = A5[i][start:stop]
                
                high = l[2]+l[3]
                low = l[0]+ l[1]
                high1 = l1[2]+l1[3]
                low1 = l1[0]+ l1[1]
                high5 = l5[2]+l5[3]
                low5 = l5[0]+ l5[1]

                highdec = int(str(high), 16) # 18
                lowdec = int(str(low),16)
                val = (int(highdec)*256)+(int(lowdec))
                val = ((val + 32768) % 65536) - 32768
                voltage2 = (4.6/4095)*val/4
                
                highdec1 = int(str(high1), 16)
                lowdec1 = int(str(low1),16)
                val1 = (int(highdec1)*256)+(int(lowdec1))
                val1 = ((val1 + 32768) % 65536) - 32768
                voltage1 = (4.6/4095)*val1/4
                
                highdec5 = int(str(high5), 16)
                lowdec5 = int(str(low5),16)
                val5 = (int(highdec5)*256)+(int(lowdec5))
                val5 = ((val5 + 32768) % 65536) - 32768
                voltage5 = (4.6/4095)*val5/4
                
                voltage3 = voltage2 - voltage1
                
                aVR = -(voltage1 + voltage2)  / 2

                aVL = (voltage1 - voltage3) / 2

                aVF = (voltage2 - voltage3)/ 2
                
                data_dict["I"].append(voltage1)
                data_dict["II"].append(voltage2)
                data_dict["III"].append(voltage3)
                data_dict["aVR"].append(aVR)
                data_dict["aVL"].append(aVL)
                data_dict["aVF"].append(aVF)
                data_dict["v5"].append(voltage5)
                
                # data_dict["DateTime"].append(d)
                start+=4
                stop+=4
                kk+=1
                if stop> len(A2[i]):
                    break

    df = {}
    if tog_arr == 0:
        print("7 lead data")
        # print('I:', len(data_dict["I"]), "II:", len(data_dict["II"]),  "III:",len(data_dict["III"]),  "aVR:", len(data_dict["aVR"]),"aVL:", len(data_dict["aVL"]),  "aVF:", len(data_dict["aVF"]), "v5: ", len(data_dict["v5"]))
        df = pd.DataFrame(data_dict)
    else:
        print("Lead II data")

    if tog_arr == 0:
        return df

    return df


def rrirrAB(rpeaks):
    rpeak_diff = np.diff(rpeaks)

    # rr_intervals = rpeak_diff/ 1000 # old
    # percentage_diff = np.abs(np.diff(rr_intervals) / rr_intervals[:-1]) * 100
    mean_percentage_diff = irrgular_per_r = 0
    if len(rpeak_diff) >= 3:
        percentage_diff = np.abs(np.diff(rpeak_diff) / rpeak_diff[:-1]) * 1003
        list_per_r = [value for value in percentage_diff if value >14]
        irrgular_per_r = (len(list_per_r)/ len(percentage_diff)) * 100
        mean_percentage_diff = np.mean(percentage_diff)
        
    threshold = 50  # Adjust this threshold as needed

    # if np.any(percentage_diff > threshold) 
    if (mean_percentage_diff > threshold)  and (irrgular_per_r > 40):
        print("Irregular R-R intervals detected. Possible AFib or AFL.")
        label = "IRREGULAR" # Possible AFib or AFL
    else:
        label = "REGULAR"
    return label

       
def subscribe(client):
    while True:
        item = queue.pop()
        client.loop_start()
        if item is not None:
            try:    
                print(f'Queue size: {queue.size()}')
                print("New Data Received")
                global mycol,topic_y
                start = time.time()
##                decoded_message=str(msg.payload.decode("utf-8",errors="replace"))
##                dd = json.dumps(item)
                #dd = json.loads(item)
                dd = item
                rawdata = dd["data"]
                
                
                try:
                    if dd["version"] == 2:
                        valueOfData = 23
                    elif dd["version"]== 5:
                        valueOfData = 24
                    else:
                        valueOfData = 23
                except:
                    valueOfData = 21
                    
                grouped_data = [rawdata[i:i + valueOfData] for i in range(0, len(rawdata), valueOfData)]
                for group in grouped_data:
#                    try:
#                      print(len(group))
#                    except Exception as e:
#                      print(e)
                    filtered_data = []
                    seconddata = group
                    for i in range(1, len(group)): 
                        current_entry = group[i] 
                        
                        previous_entry = group[i - 1] 
                        if i== 1:
                            filtered_data.append(previous_entry)
                        if current_entry["dateTime"] != previous_entry["dateTime"]:
                            filtered_data.append(current_entry)
                    seconddata = filtered_data
                    sorted_data = sorted(seconddata, key=lambda x: x.get('dateTime', ''))
                    newsublist = funcs(sorted_data)
                    seconddata = sorted_data
                    print("SECOND",len(seconddata))
                    dataerror =0
                    l = []
                    vol = []
                    co = []
                    newlist = []
                    imageresource = folder_path
                    datetimee = []
                    leadlist=[]
                    allarr =''
                    trigger = False
                    mobileBaterry = None
                    rpmId=''
                    versionList = []
                    version = 0
                    patientData = {}
                    coordinates=[]
                    datalength = 0
                    newlist1 = []
                    positionX = []
                    positionY = []
                    positionZ = []
                    battery = None
                    memoryUtilized = None
                    sysncDataReaming = None

                    positionFinal = 0
                    for i in range(0,len(seconddata)):
                        patient = dd['patient']
                        allarr = dd['ecgPackage']
                        try:
                            try:
                              version = dd['version']
                            except:
                              print("version not getting...")
                              pass
                            
                            try:
                              trigger = dd['trigger']
                            except:
                              print("trigger not getting...")
                              pass
                            try:
                              mobileBaterry = dd['mobileBaterry']
                            except:
                              print("mobileBaterry not getting...")
                              pass
                            try:
                              rpmId = dd['rpmId']
                            except:
                              print("rpmId not getting...")
                              pass
    ##                        versionList.append(dd['data'][i]['version'])
                            try:
                              patientData = dd['patientData']
                            except:
                              print("patientData not getting...")
                              pass  
    
                            try:
                              datalength = len(group)
                            except:
                              print("datalength not getting...")
                              pass                          
                              
                            try:
                              coordinates = dd["coordinates"]
                            except:
                              print("coordinates not getting...")
                              pass
                            try:
                              battery = dd['battery']
                            except:
                              print("battery not getting...")
                              pass  
                            try:
                              memoryUtilized = dd['memoryUtilized']
                            except:
                              print("memoryUtilized not getting...")
                              pass
                            try:
                              sysncDataReaming = dd['sysncDataReaming']
                            except:
                              print("sysncDataReaming not getting...")
                              pass                         
                         
                            #version = dd['version']
                        except Exception as e:
                            #print(e)
                            pass
                        
                        try:
                            positionX.append(seconddata[i]['positionX'])
                            positionY.append(seconddata[i]['positionY'])
                            positionZ.append(seconddata[i]['positionZ'])
                        except:
                            print("position Not found")

                        datetimee.append(seconddata[i]['dateTime']) 
                        newlist.append(seconddata[i]['data'])
##                        try:
##                            newlist1.append(seconddata[i]['data1'])  
##                        except:
##                            pass
                        leadlist.append(seconddata[i]['lead'])
                    allstring = ''.join(newlist)
                    print("DATALENGHT:",datalength)  
                    for i in allstring:
                        if len(l)!=4:
                            l.append(str(i))
                        if len(l)==4:
                            high = l[2]+l[3]
                            low = l[0]+ l[1]
                            highdec = int(str(high), 16)
                            lowdec = int(str(low),16)
                            val = (int(highdec)*256)+int(lowdec)

                            if int(version)==5:
                              val = ((val + 32768) % 65536) - 32768
                              voltage = (4.6/4095)*val/4
                            else:
                              #val = ((val + 32768) % 65536) - 32768
                              voltage = (4.6/4095)*val
                            vol.append(str(voltage))
                            l.clear()
                    print(leadlist)
                    if 0 not in leadlist:
                        pass
                    else:            
                        sample_rate = 200
                        try:
                            positionOutput = position_management(patient,positionX,positionY,positionZ)
                            positionFinal = positionOutput[patient]
                        except Exception as pe:
                            print("Position Processing Problem",pe)
                            
                        date_time = np.array(newsublist["DateTime"])
                        OriginalSignal = [float(s) for s in vol]
                        OriginalSignal = MinMaxScaler(feature_range=(0,4)).fit_transform(np.array(OriginalSignal).reshape(-1,1)).squeeze()
                        ecgdata = pd.DataFrame({"ECG":OriginalSignal})
                        final_output = noise_engine(flag = "200",ecgdata=ecgdata)

                        mintime = min(datetimee)
                        maxtime = max(datetimee)
                        maxtimes = datetime.datetime.fromtimestamp(int(maxtime)/1000)
                        maxtimesnewtime =maxtimes.strftime("%Y-%m-%d %H:%M:%S")
                        DT2 = parser.parse(maxtimesnewtime)
                        mintimes = datetime.datetime.fromtimestamp(int(mintime)/1000)
                        mintimesnewtime = mintimes.strftime("%Y-%m-%d %H:%M:%S")
                        DT1 = parser.parse(mintimesnewtime)
                        timetaken = int((DT2 - DT1).total_seconds())
                        print("timetakenFinal:",timetaken)
                        loss_data = np.diff(datetimee)
                        print("EPOCH",datetimee)
                        if timetaken<4:
                            print("Data less than 2 second, half beat solution")
                        elif (loss_data > 60000).any():
                            print("Long Data Recording Found")
                        elif timetaken>50:
                            print("Data Greter than 50 second, Artifacts solution")
                        else:
                            if (loss_data > 800).any():
                                result_data = [{"patient":dd["patient"],"HR":0,"starttime":mintime,"endtime":maxtime,"Arrhythmia":'Artifacts','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":[],"RR":0,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]
                                print("LOG:",result_data)
                                client.publish(topic_y,json.dumps(result_data),qos=2)

                            elif final_output == "high_noise":
                                print("GPT Output")
                                result_data = [{"patient":dd["patient"],"HR":0,"starttime":mintime,"endtime":maxtime,"Arrhythmia":'Artifacts','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":[],"RR":0,"offline":True,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]
                                print("LOG:",result_data)
                                client.publish(topic_y,json.dumps(result_data),qos=2)
                                
        ##                        item = queue.pop()
                            elif dd['ecgPackage']=="Free":
                                print("Free")
                                try:
                                    fa=200
                                    naa = np.array(OriginalSignal)
                                    rpeaks = detect_beats(naa, float(fa))
                                    try:
                                        br,hrv = BPM(rpeaks)
                                    except:
                                        br,hrv = 0,[]
                                    
                                    HR = int(60*int(len(rpeaks))/(timetaken))
                                    _, waves_peak = nk.ecg_delineate(naa, rpeaks, sampling_rate=fa, method="peak")
                                    signal_dwt, waves_dwt = nk.ecg_delineate(naa, rpeaks, sampling_rate=fa, method="dwt")
                                    RRintervallist=[]
                                    for i in rpeaks:
                                        RRintervallist.append(i)
                                    SAf = []
                                    RRpeaks="0"
                                    for i in range(len(RRintervallist)):
                                        try:
                                            RRpeaks = abs(int(RRintervallist[i])*5-int(RRintervallist[i+1])*5)
                                            SAf.append(RRpeaks)
                                        except:
                                            SAf.append(0)
                                            RRpeaks="0"


                                    try:
                                        Ppeak = waves_peak['ECG_P_Peaks'][5]
                                        Rpeak = rpeaks[5]
                                        Ppeak = int(Ppeak)*5
                                        Rpeak = int(Rpeak)*5
                                        PRpeaks = abs(Rpeak-Ppeak)
                                    except:
                                        PRpeaks = "0"
                                    try:
                                        Tpeak = waves_peak['ECG_T_Peaks'][5]
                                        Qpeak = waves_peak['ECG_Q_Peaks'][5]
                                        Tpeak = int(Tpeak)*5
                                        Qpeak = int(Qpeak)*5
                                        QTpeaks = abs(Tpeak-Qpeak)
                                    except:
                                        QTpeaks="0"
                                        
                                
                                    try:
                                        Speak = waves_peak['ECG_S_Peaks'][5]
                                        Qpeak = waves_peak['ECG_Q_Peaks'][5]
                                        Speak = int(Speak)*5
                                        Qpeak = int(Qpeak)*5
                                        SQpeaks = abs(Speak-Qpeak)
                                    except:
                                        SQpeaks = "0"

                                    try:
                                        Spa = waves_peak['ECG_S_Peaks'][5]
                                        Ton = waves_dwt['ECG_T_Onsets'][5]
                                        Spa = int(Spa)*5
                                        Ton = int(Ton)*5
                                        STseg = abs(Ton-Spa)
                                    except:
                                        STseg = "0"

                                        
                                    
                                    try:
                                        PP = waves_dwt['ECG_P_Offsets']
                                        RRO = waves_dwt['ECG_R_Onsets']
                                        if math.isnan(PP[5]) or math.isnan(RRO[5]):
                                            PRseg = "0"
                                        else:
                                            PPIn = int(PP[5])*5
                                            RRon = int(RRO[5])*5
                                            PRseg =  abs(PPIn - RRon)
                                    
                                    except:
                                            PRseg = "0"
                                            
                                    try:
                                        beatss = int(int(HR)/4)
                                    except:
                                        beatss = 0
    ##                                aped=[]
    ##                                for i in range(len(rpeaks)-1):
    ##                                    m=rpeaks[i+1]-rpeaks[i]
    ##                                    aped.append(m*5/1000)
    ##
    ##                                variation=[]
    ##                                rrint=''
    ##                                for i in range(len(aped)-1):
    ##                                    
    ##                                    variation.append(get_percentage_diff(aped[i+1],aped[i]))
    ##                                
    ##                                if Average(variation)>12:
    ##                                    rrint = "IRREGULAR"
    ##                                else:
    ##                                    rrint = "REGULAR"
                                    rrint = rrirrAB(rpeaks)
                                    finddata=[]
                                    try:
                                        result_data = {"patient":dd["patient"],"HR":str(HR),"starttime":mintime,"endtime":maxtime,"Arrhythmia":'','kit':dd["kit"],'position':positionFinal,"beats":len(rpeaks),"RRInterval":str(SAf[0]),"PRInterval":str(PRpeaks),"QTInterval":str(QTpeaks),"QRSComplex":str(SQpeaks),"STseg":str(STseg),"PRseg":str(PRseg),"noOfPause":0,"noOfPauseList":[],"ecgPackage":allarr,"trigger":trigger,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":hrv,"RR":br,"offline":True,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}
                                    except:
                                        result_data = {"patient":dd["patient"],"HR":str(HR),"starttime":mintime,"endtime":maxtime,"Arrhythmia":'','kit':dd["kit"],'position':positionFinal,"beats":len(rpeaks),"RRInterval":str(0),"PRInterval":str(0),"QTInterval":str(0),"QRSComplex":str(0),"STseg":str(0),"PRseg":str(0),"noOfPause":0,"noOfPauseList":[],"ecgPackage":allarr,"trigger":trigger,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":hrv,"RR":br,"offline":True,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}


                                    noise_cc = noise_check_again(naa)

                                        
                                    if int(HR)<60 and noise_cc == "Normal":
                                        rpeakss = hamilton_segmenter(signal = naa)["rpeaks"]
                                        timetakens = round((np.sum(np.diff(datetimee))+500)/1000)
                                        HRss = int(60*int(len(rpeakss))/(timetakens))
                                        HRs = int(60*int(len(rpeakss))/(timetaken))
                                        if HRs>=60 and HRss>=60:
                                            result_data.update({"Arrhythmia":'Artifacts',"HR":0})
                                            d2 = result_data
                                            finddata.append(d2)
                                            newdata = [i for n, i in enumerate(finddata) if i not in finddata[n + 1:]]
                                            print("LOG:",newdata)
                                            client.publish(topic_y,json.dumps(newdata),qos=2)
                                        else:
                                            result_data.update({"Arrhythmia":'Normal',"HR":str(HRs)})
                                            d1 = result_data
                                            finddata.append(d1)
                                            newdata = [i for n, i in enumerate(finddata) if i not in finddata[n + 1:]]
                                            print("LOG:",newdata)
                                            client.publish(topic_y,json.dumps(newdata),qos=2)
                                        
                                    elif int(HR)>100 and noise_cc == "Normal":
                                        result_data.update({"Arrhythmia":'Normal'})
                                        d2 = result_data
                                        finddata.append(d2)
                                        newdata = [i for n, i in enumerate(finddata) if i not in finddata[n + 1:]]
                                        print("LOG:",newdata)
                                        client.publish(topic_y,json.dumps(newdata),qos=2)
                                    elif rrint=="IRREGULAR":
                                        result_data.update({"Arrhythmia":'ABNORMAL'})
                                        d1 = result_data
                                        finddata.append(d1)
                                        newdata = [i for n, i in enumerate(finddata) if i not in finddata[n + 1:]]
                                        print("LOG:",newdata)
                                        client.publish(topic_y,json.dumps(newdata),qos=2)

                                    else:
                                        if int(HR)<60 and noise_cc != "Normal":
                                            result_data.update({"Arrhythmia":'Artifacts',"HR":0})
                                            d2 = result_data
                                            finddata.append(d2)
                                            newdata = [i for n, i in enumerate(finddata) if i not in finddata[n + 1:]]
                                            print("LOG:",newdata)
                                            client.publish(topic_y,json.dumps(newdata),qos=2)
                                        elif int(HR)>100 and noise_cc != "Normal":
                                            result_data.update({"Arrhythmia":'Artifacts'})
                                            d2 = result_data
                                            finddata.append(d2)
                                            newdata = [i for n, i in enumerate(finddata) if i not in finddata[n + 1:]]
                                            print("LOG:",newdata)
                                            client.publish(topic_y,json.dumps(newdata),qos=2)
                                        else:
                                            result_data.update({"Arrhythmia":'Normal'})
                                            d1 = result_data
                                            finddata.append(d1)
                                            newdata = [i for n, i in enumerate(finddata) if i not in finddata[n + 1:]]
                                            print("LOG:",newdata)
                                            client.publish(topic_y,json.dumps(newdata),qos=2)


                                except Exception as e:
                                    result_data = [{"patient":dd["patient"],"HR":0,"starttime":mintime,"endtime":maxtime,"Arrhythmia":'Artifacts','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":[],"RR":0,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]
                                    print("LOG5:",result_data,e)
                                    client.publish(topic_y,json.dumps(result_data),qos=2)



                            else:
    ##                            if isinstance(final_output, type(None)) or isinstance(final_output, str):
    ##                                pass
    ##                            elif len(final_output)==2 and int(timetaken)>=5 and int(timetaken)<=20:
    ##                                pass
    ##
    ##                            else:
                                try:
                                    naa = np.array(OriginalSignal)
                                except:
                                    naa = np.array(OriginalSignal)
                                # final_label, percentage, model_data = vfib_model_check_new(naa, vfib_vfl_model, fs=200)
                                final_label = check_vfib_vfl_model(naa, vfib_vfl_model)

                                if final_label == "VFIB/Vflutter":
                                        na = np.array(OriginalSignal)
                                        rpeaks = detect_beats(na, float(100))
                                        beats = []
                                        for nnn in rpeaks:
                                            beats.append({"index":int(nnn),"dateTime":int(date_time[nnn])})

                                        try:
                                            br,hrv = BPM(rpeaks)
                                        except:
                                            br,hrv = 0,[]
                                        
                                        mintime = min(datetimee)
                                        maxtime = max(datetimee)
                                        maxtimes = datetime.datetime.fromtimestamp(int(maxtime)/1000)
                                        maxtimesnewtime =maxtimes.strftime("%Y-%m-%d %H:%M:%S")
                                        DT2 = parser.parse(maxtimesnewtime)
                                        mintimes = datetime.datetime.fromtimestamp(int(mintime)/1000)
                                        mintimesnewtime = mintimes.strftime("%Y-%m-%d %H:%M:%S")
                                        DT1 = parser.parse(mintimesnewtime)
                                        timetaken = int((DT2 - DT1).total_seconds())
                                        try:
                                            HR = int(60*int(len(rpeaks))/(timetaken))
                                            if abs(timetaken)<5:
                                                result_data = [{"patient":dd["patient"],"HR":0,"starttime":mintime,"endtime":maxtime,"Arrhythmia":'Artifacts','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":[],"RR":0,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]
                                                print("LOG:",result_data)
                                                client.publish(topic_y,json.dumps(result_data),qos=2)
            ##                                    item = queue.pop()
                                            else:
                                                result_data = [{"patient":dd["patient"],"HR":int(HR),"starttime":mintime,"endtime":maxtime,"Arrhythmia":'VFIB','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":hrv,"RR":br,"templateBeat":beats,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]
                                                print("LOG:",result_data)
                                                for i in result_data:
                                                    x = mycol.insert_one(dict(i))

                                                client.publish(topic_y,json.dumps(result_data),qos=2)
            ##                                    item = queue.pop()
                                        except:
                                                result_data = [{"patient":dd["patient"],"HR":0,"starttime":mintime,"endtime":maxtime,"Arrhythmia":'Artifacts','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":[],"RR":0,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]                                                   
                                                print("LOG:",result_data)
                                                client.publish(topic_y,json.dumps(result_data),qos=2)


                                elif final_label == "ASYS" and int(timetaken)>=5:
                                        mintime = min(datetimee)
                                        maxtime = max(datetimee)
                                        result_data = [{"patient":dd["patient"],"HR":0,"starttime":mintime,"endtime":maxtime,"Arrhythmia":'ASYSTOLE','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":[],"RR":0,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]
                                        print("LOG:",result_data)
                                        for i in result_data:
                                            x = mycol.insert_one(dict(i))

                                        client.publish(topic_y,json.dumps(result_data),qos=2)
        ##                                item = queue.pop()

                                elif final_label == "Noise":
                                        print("VFIB Model Noise")
                                        mintime = min(datetimee)
                                        maxtime = max(datetimee)
                                        try:
                                            na = np.array(OriginalSignal)
                                            rpeaks = detect_beats(na, float(200))
                                            beats = []
                                            for nnn in rpeaks:
                                                    beats.append({"index":int(nnn),"dateTime":int(date_time[nnn])})

                                            try:
                                                br,hrv = BPM(rpeaks)
                                            except:
                                                br,hrv = 0,[]


                                            HR = int(60*int(len(rpeaks))/(timetaken))
                                            if HR>60 and HR<100:
                                                if int(version) == 5:
                                                    label_rlbbb = LBBB_RBBB(na,rpeaks,imageresource)
                                                    if label_rlbbb=="LBBB":
                                                        result_data = [{"patient":dd["patient"],"HR":str(HR),"starttime":mintime,"endtime":maxtime,"Arrhythmia":'Normal','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":hrv,"RR":br,"MI":"LBBB","templateBeat":beats,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]
                                                        print("LOG:",result_data)
                                                        client.publish(topic_y,json.dumps(result_data),qos=2)
                                                    elif label_rlbbb=="RBBB":
                                                        result_data = [{"patient":dd["patient"],"HR":str(HR),"starttime":mintime,"endtime":maxtime,"Arrhythmia":'Normal','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":hrv,"RR":br,"MI":"RBBB","templateBeat":beats,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]
                                                        print("LOG:",result_data)
                                                        client.publish(topic_y,json.dumps(result_data),qos=2)
                                                    else:
                                                        result_data = [{"patient":dd["patient"],"HR":str(HR),"starttime":mintime,"endtime":maxtime,"Arrhythmia":'Normal','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":hrv,"RR":br,"templateBeat":beats,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]
                                                        print("LOG:",result_data)
                                                        client.publish(topic_y,json.dumps(result_data),qos=2)
                                                else:
                                                    result_data = [{"patient":dd["patient"],"HR":str(HR),"starttime":mintime,"endtime":maxtime,"Arrhythmia":'Normal','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":hrv,"RR":br,"templateBeat":beats,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]
                                                    print("LOG:",result_data)
                                                    client.publish(topic_y,json.dumps(result_data),qos=2)

                                            else:
                                                result_data = [{"patient":dd["patient"],"HR":0,"starttime":mintime,"endtime":maxtime,"Arrhythmia":'Artifacts','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":[],"RR":0,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]
                                                print("LOG:",result_data)
                                                client.publish(topic_y,json.dumps(result_data),qos=2)
        ##                                        item = queue.pop()
                                                

                                        except:
                                            result_data = [{"patient":dd["patient"],"HR":0,"starttime":mintime,"endtime":maxtime,"Arrhythmia":'Artifacts','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":[],"RR":0,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]
                                            print("LOG:",result_data)
                                            client.publish(topic_y,json.dumps(result_data),qos=2)
        ##                                    item = queue.pop()

                                else:
                                    try:
                                        aboutdata = pd.DataFrame({"ECG":OriginalSignal})
                                        with warnings.catch_warnings():
                                            warnings.filterwarnings("ignore", category=RuntimeWarning)
                                        for i in glob.glob(imageresource+"/*.jpg"):
                                            os.remove(i)


                                        newdada = aboutdata["ECG"]
                                        naa = np.array(newdada)

                                        rrint=''
                                        fa=200
                                        rpeaks = detect_beats(newdada, float(fa))
                                        beats = []
                                        for nnn in rpeaks:
                                                beats.append({"index":int(nnn),"dateTime":int(date_time[nnn])})


                                        try:
                                            br,hrv = BPM(rpeaks)
                                        except:
                                            br,hrv = 0,[]
                                        r_index = rpeaks


                                        b_es = baseline_construction_200(newdada,101)
                                        low_es = lowpass(b_es)
                                        r = list(r_index)
                                        r_index = np.array(r)
                                        s_index = find_s_indexs(b_es,r_index,20)
                                        q_index = find_q_indexs(b_es, r_index, 15)
                                        j_index = find_j_indexs(b_es,s_index)
                                        wideqrs=[]
                                        for iii in range(len(q_index)-1):
                                            if np.isnan(q_index[iii]):
                                                q_index[iii]=0
                                                wideqrs.append(0)
                                            elif np.isnan(s_index[iii]):
                                                s_index[iii]=0
                                                wideqrs.append(0)
                                                
                                            else:
                                                difff = s_index[iii]-q_index[iii]
                                                wideqrs.append((difff*5)/1000)
                                        
                                        newpvcs = []
                                        newpvcswide = []
                                        
                                        for ias in wideqrs:
                                            if ias>=0.09:
                                                newpvcs.append(1)
                                            else:
                                                newpvcs.append(0)
                                        
                                        for iass in wideqrs:
                                            if iass>=0.13:
                                                newpvcswide.append(1)
                                            else:
                                                newpvcswide.append(0)

                                        print(newpvcs)
                                        ss2 = low_es
                                        pt1 = []
                                        p_t1 = []
                                        pt, p_t = find_p_t(low_es, r_index, q_index, s_index)
                                        try:
                                          try:
                                              _, waves_peak = nk.ecg_delineate(newdada, rpeaks, sampling_rate=fa, method="peak")
                                              signal_dwt, waves_dwt = nk.ecg_delineate(newdada, rpeaks, sampling_rate=fa, method="dwt")
                                          except Exception as rr:
                                              _, waves_peak = nk.ecg_delineate(newdada, rpeaks, sampling_rate=fa, method="peak")
                                              signal_dwt, waves_dwt = nk.ecg_delineate(newdada, rpeaks, sampling_rate=fa, method="cwt")
                                        except Exception as ee:
                                              print("neurokit2 Error:",ee)
    ##                                    aped=[]
    ##                                    for i in range(len(rpeaks)-1):
    ##                                        m=rpeaks[i+1]-rpeaks[i]
    ##                                        aped.append(m*5/1000)
    ##
    ##
    ##
    ##
    ##                                    varitionforAFib=[]
    ##
    ##                                    for i in range(len(aped)-1):
    ##                                        
    ##                                        varitionforAFib.append(get_percentage_diff(aped[i+1],aped[i]))
    ##                                    if Average(varitionforAFib)>=12:
    ##                                        rrint = "IRREGULAR"
    ##                                    else:
    ##                                        rrint = "REGULAR"
                                        # print(rrint)
                                        rrint = rrirrAB(rpeaks)
                                        RRintervallist = []
                                        for i in rpeaks:
                                            RRintervallist.append(i)
                                        SAf = []
                                        
                                        for i in range(len(RRintervallist)):
                                            try:
                                                RRpeaks = abs(int(RRintervallist[i])*5-int(RRintervallist[i+1])*5)
                                                SAf.append(RRpeaks)
                                            except:
                                                SAf.append(0)
                                                RRpeaks="0"
                                        print(SAf)


                                        try:
                                            Ppeak = waves_peak['ECG_P_Peaks'][5]
                                            Rpeak = rpeaks[5]
                                            Ppeak = int(Ppeak)*5
                                            Rpeak = int(Rpeak)*5
                                            PRpeaks = abs(Rpeak-Ppeak)
                                        except:
                                            PRpeaks = "0"
                                        try:
                                            Tpeak = waves_peak['ECG_T_Peaks'][5]
                                            Qpeak = waves_peak['ECG_Q_Peaks'][5]
                                            Tpeak = int(Tpeak)*5
                                            Qpeak = int(Qpeak)*5
                                            QTpeaks = abs(Tpeak-Qpeak)
                                        except:
                                            QTpeaks="0"
                                            
                                    
                                        try:
                                            Speak = waves_peak['ECG_S_Peaks'][5]
                                            Qpeak = waves_peak['ECG_Q_Peaks'][5]
                                            Speak = int(Speak)*5
                                            Qpeak = int(Qpeak)*5
                                            SQpeaks = abs(Speak-Qpeak)
                                        except:
                                            SQpeaks = "0"

                                        try:
                                            Spa = waves_peak['ECG_S_Peaks'][5]
                                            Ton = waves_dwt['ECG_T_Onsets'][5]
                                            Spa = int(Spa)*5
                                            Ton = int(Ton)*5
                                            STseg = abs(Ton-Spa)
                                        except:
                                            STseg = "0"

                                            
                                        
                                        try:
                                            PP = waves_dwt['ECG_P_Offsets']
                                            RRO = waves_dwt['ECG_R_Onsets']
                                            if math.isnan(PP[5]) or math.isnan(RRO[5]):
                                                PRseg = "0"
                                            else:
                                                PPIn = int(PP[5])*5
                                                RRon = int(RRO[5])*5
                                                PRseg =  abs(PPIn - RRon)
                                        
                                        except:
                                            PRseg = "0"





                                        pvc = ''
                                        afib = ''
                                        var=''
                                        brad = ''
                                        tachy = ''
                                        VT=''
                                        mintime = min(datetimee)
                                        maxtime = max(datetimee)
                                        maxtimes = datetime.datetime.fromtimestamp(int(maxtime)/1000)
                                        maxtimesnewtime =maxtimes.strftime("%Y-%m-%d %H:%M:%S")
                                        DT2 = parser.parse(maxtimesnewtime)
                                        mintimes = datetime.datetime.fromtimestamp(int(mintime)/1000)
                                        mintimesnewtime = mintimes.strftime("%Y-%m-%d %H:%M:%S")
                                        DT1 = parser.parse(mintimesnewtime)
                                        timetaken = int((DT2 - DT1).total_seconds())
                                        finddata = []

                                        HR = int(60*int(len(rpeaks))/(timetaken))
                                        print("HRRRR:",HR)
                                        OrHR = HR
                                        try:
                                            result_data = {"patient":dd["patient"],"HR":str(HR),"starttime":mintime,"endtime":maxtime,"Arrhythmia":'','kit':dd["kit"],'position':positionFinal,"beats":len(rpeaks),"RRInterval":str(SAf[0]),"PRInterval":str(PRpeaks),"QTInterval":str(QTpeaks),"QRSComplex":str(SQpeaks),"STseg":str(STseg),"PRseg":str(PRseg),"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"noOfPauseList":[],"ecgPackage":"All-Arrhythmia","trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":hrv,"RR":br,"templateBeat":beats,"threeLatter":[],"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}
                                        except Exception as e:
                                            print(e)
                                            result_data = {"patient":dd["patient"],"HR":str(HR),"starttime":mintime,"endtime":maxtime,"Arrhythmia":'','kit':dd["kit"],'position':0,"beats":len(rpeaks),"RRInterval":str(0),"PRInterval":str(0),"QTInterval":str(0),"QRSComplex":str(0),"STseg":str(0),"PRseg":str(0),"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"noOfPauseList":[],"ecgPackage":"All-Arrhythmia","trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":hrv,"RR":br,"templateBeat":beats,"threeLatter":[],"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}






                                        naaaaa = np.array(newdada)
                                        r_index_s = find_r_peakss(naaaaa,fs=200)
    ##                                    apeds=[]
    ##                                    for i in range(len(r_index_s)-1):
    ##                                        ma=r_index_s[i+1]-r_index_s[i]
    ##                                        apeds.append(ma*5/1000)
    ##                                    varitionforAFibs=[]
    ##                                    for i in range(len(apeds)-1):
    ##                                        varitionforAFibs.append(get_percentage_diff(apeds[i+1],apeds[i]))
    ##
    ##                                    if Average(varitionforAFibs)>=9:
    ##                                        rrints = "IRREGULAR"
    ##                                    else:
    ##                                        rrints = "REGULAR"
            ##                            print(Average(varitionforAFibs),rrints,rrint)
                                        rrints = rrirrAB(r_index_s)
                                        if int(timetaken)>6:
                                            try:
                                                newafib,ei_ti = afib_fultter_model_check(naaaaa, afib_load_model, 200,newsublist)
                                                rpeaks = detect_beats(naaaaa,200)
                                                HR = int(60*int(len(rpeaks))/(timetaken))
                                                if HR<40:
                                                    pass
                                                else:
                                                    if newafib == "Aflutter": # and int(HR)>=62
                                                        result_data.update({"Arrhythmia":'AFL',"PRInterval":"0","HR":str(HR)})
                                                        last_label_afl = ''
                                                        for ei_ti_afib_afl in ei_ti:
                                                            if last_label_afl!="Aflutter":
                                                              last_label_afl = "Aflutter"
                                                              result_data["threeLatter"].append(ei_ti_afib_afl)
                                                        d4 = result_data
                                                        finddata.append(d4)
                                                    if rrints == "IRREGULAR" and rrint=="IRREGULAR":
                                                        if newafib == "Afib":
                                                            result_data.update({"Arrhythmia":'AFIB',"PRInterval":"0","HR":str(HR)})
                                                            last_label = ''
                                                            for ei_ti_afib_afl in ei_ti:
                                                                if last_label!='afib':
                                                                    last_label="afib"
                                                                    result_data["threeLatter"].append(ei_ti_afib_afl)
                                                            d4 = result_data
                                                            finddata.append(d4)
                                                    
                                                        # if (newafib == "Afib" or newafib == "Aflutter") and int(HR)<65:
                                                        #     result_data.update({"Arrhythmia":'SINUS-ARR',"PRInterval":"0","HR":str(HR)})
                                                        #     for ei_ti_afib_afl in ei_ti:
                                                        #         result_data["threeLatter"].append(ei_ti_afib_afl)
                                                        #     d4 = result_data
                                                        #     finddata.append(d4)
    
                                                        if newafib == "Abnormal":
                                                            result_data.update({"Arrhythmia":'SINUS-ARR',"PRInterval":"0","HR":str(HR)})
                                                            d4 = result_data
                                                            finddata.append(d4)
                                            except Exception as e:
                                                print("AFIB_AFL issue",e)
                            
                                        if float(HR)<60 and result_data["Arrhythmia"]!="AFIB" and result_data["Arrhythmia"]!="AFL" and result_data["Arrhythmia"]!="ABNORMAL":
                                            if(SACompare(SAf, 5000)) and (rrints == "IRREGULAR" or rrints == "REGULAR"):
                                                result_data.update({"Arrhythmia":'Long Pause'})
                                                d1 = result_data
                                                finddata.append(d1)
                                            else:
                                                # rpeaksss = QRS_detection(newdada,200,350)
                                                rpeaksss = hamilton_segmenter(signal = low_es)["rpeaks"]
                                                timetakens = round((np.sum(np.diff(datetimee))+500)/1000)
                                                HRs = int(60*int(len(rpeaksss))/(timetaken))
                                                HRss = int(60*int(len(rpeaksss))/(timetakens))
                                                if HRs>=60 and HRss>=60:
                                                    pass
                                                elif rrints == "IRREGULAR" and rrint=="IRREGULAR":
                                                    result_data.update({"Arrhythmia":'SINUS-ARR',"HR":str(HRs)})
                                                    d1 = result_data
                                                    finddata.append(d1)
                                                else:
                                                    result_data.update({"Arrhythmia":'BR',"HR":str(HRs)})
                                                    d1 = result_data
                                                    finddata.append(d1)
                                        
                                        if float(HR)>100 and result_data["Arrhythmia"]!="AFIB" and result_data["Arrhythmia"]!="AFL" and rrints != "IRREGULAR" and rrint != "IRREGULAR":
                                            result_data.update({"Arrhythmia":'TC'})
                                            d2 = result_data
                                            finddata.append(d2)





                                        if (SACompare(SAf, 3000)):
                                            l=[]
                                            for x in SAf:
                                                if x>=3000:
                                                    l.append(1)
                                                else:
                                                    l.append(0)
                                            if 1 in l:
                                                noofpause = l.count(1)
                                            else:
                                                noofpause = 0
                                            
                                            
                                            if result_data["Arrhythmia"]=="AFIB":  
                                              result_data.update({"noOfPause":noofpause,"noOfPauseList":[a/1000 for a in SAf if a>3000]})
                                              d1 = result_data
                                              finddata.append(d1)
                                            else:
                                              result_data.update({"Arrhythmia":'Long Pause',"noOfPause":noofpause,"noOfPauseList":[a/1000 for a in SAf if a>3000]})
                                              d1 = result_data
                                              finddata.append(d1)


                                        if SACompareShort(SAf,2000,2900):
                                            l=[]
                                            for x in SAf:
                                                if x>=2000 and x<=2900:
                                                    l.append(1)
                                                else:
                                                    l.append(0)
                                            if 1 in l:
                                                noofpause = l.count(1)
                                            else:
                                                noofpause = 0
                                                
                                            if result_data["Arrhythmia"]=="AFIB":
                                              result_data.update({"noOfPause":noofpause,"noOfPauseList":[a/1000 for a in SAf if a>=2000 and a<=2900 ]})
                                              d1 = result_data
                                              finddata.append(d1)
                                            else:
                                              result_data.update({"Arrhythmia":'Short Pause',"noOfPause":noofpause,"noOfPauseList":[a/1000 for a in SAf if a>=2000 and a<=2900 ]})
                                              d1 = result_data
                                              finddata.append(d1)


                                        layer2 = newpvcs
                                        if 1 in layer2:
                                            lis=[]
                                            count=1
                                            pvc_data = lowpass_1(b_es)
                                            aboutdatas = pd.DataFrame(pvc_data)
                                            for i in rpeaks:
                                                lis.append(i)

                                                if i == rpeaks[0]:
                                                    count += 1
                                                    lis.append(i)
                                                    try:
                                                        window_start = int(lis[0]) - 10
                                                    except:
                                                        window_start = int(lis[0]) - 20
                                                    window_end = int(lis[0]) + 100
                                                elif i == rpeaks[1]:
                                                    count += 1
                                                    lis.append(i)
                                                    window_start = int(lis[0]) - 50
                                                    window_end = int(lis[0]) + 130
                                                else:
                                                    count += 1
                                                    lis.append(i)
                                                    window_start = int(lis[0]) - 50
                                                    window_end = int(lis[0]) + 80

                                                aa = pd.DataFrame(aboutdatas.iloc[window_start:window_end])
        ##                                        plt.ion()
                                                plt.plot(aa,color='blue')
                                                plt.axis("off")
                                                plt.savefig(f"{imageresource}/p_{int(lis[0])}.jpg")
                                                aq = cv2.imread(f"{imageresource}/p_{int(lis[0])}.jpg")
                                                aq = cv2.resize(aq, (360, 720))
                                                cv2.imwrite(f"{imageresource}/p_{int(lis[0])}.jpg", aq)
                                                lis.clear()
                                                #plt.pause(0.01)
                                                plt.close()
        ##                                        plt.ioff()


                                            
                                            observer = []
                                            mainpick = []
                                            newdatepvclist=[]
                                            patientid = dd["patient"]
                                            if os.path.exists("pvcs/"+patientid):
                                                pass
                                            else:
                                                os.mkdir("pvcs/"+patientid)

        ##                                    files = sorted(glob.glob(imageresource+"/*.jpg"), key=len)
                                            files = sorted(glob.glob(imageresource+"/*.jpg"), key=sorting_key)
                                            for pvcfilename in files:
                                              predictions,ids = prediction_model(pvcfilename)
                                              if str(ids) == "PVC" and float(predictions[3])>0.92:
                                                  observer.append(1)
                                                  mainpick.append(int(pvcfilename.split("_")[1].split(".jpg")[0]))
                                                  datetimeapp = int(date_time[int(pvcfilename.split("_")[1].split(".jpg")[0])])
                                                  shutil.copy(pvcfilename,"pvcs/"+patientid+"/"+"p_"+str(pvcfilename.split("_")[1].split(".jpg")[0]+"_"+str(datetimeapp)+".jpg"))
                                              else:
                                                  observer.append(0)
                                                  
                                            
                                            for nnn in mainpick:
                                                newdatepvclist.append(str(date_time[nnn]))
                                            peaksdefined = newdatepvclist
                                            bb = observer
                                            actaulPVC = observer
                                            bigem = []
                                            bigem_count= 0
                                            for q,k in enumerate(bb):
                                                if len(bigem) == 3:
                                                    bigem_count+=1
                                                    try:
                                                        if bb[q] ==0 and bb[q+1]==1:
                                                            bigem.clear()
                                                            bigem.append(1)
                                                        else:
                                                            bigem.clear()
                                                    except:
                                                        bigem.clear()


                                                if len(bigem ) ==0 and k ==1:
                                                    bigem.append(1)
                                                elif len(bigem) ==1 and k ==0:

                                                    bigem.append(0)
                                                elif len(bigem) ==2 and k ==1:
                                                    bigem.append(1)
                                                else:
                                                    if len(bigem)==1 and (1 in bigem) and k==1:
                                                        bigem.clear()
                                                        bigem.append(1)
                                                    elif len(bigem)>1: 
                                                        bigem.clear()
                                                        if k ==1:
                                                            bigem.append(1)
                                                        
                                                        
                                            if len(bigem) == 3:
                                                bigem_count+=1
                                                bigem.clear()



                                            # Trigeminy 


                                            Trigem = []
                                            Trigem_count = 0
                                            for m,l in enumerate(bb):
                                                if len(Trigem) == 4:
                                                    Trigem_count+=1
                                                    try:
                                                        if bb[m] ==0 and bb[m+1]==0 and bb[m+2]==1:
                                                            Trigem.clear()
                                                            Trigem.append(1)
                                                        else:
                                                            Trigem.clear()
                                                    except:
                                                        Trigem.clear()

                                                if len(Trigem) ==0 and l ==1:
                                                    Trigem.append(1)
                                                elif len(Trigem) ==1 and l ==0:

                                                    Trigem.append(0)
                                                elif len(Trigem) ==2 and l ==0:
                                                    Trigem.append(0)
                                                elif len(Trigem) ==3 and l ==1:
                                                    Trigem.append(1)
                                                else:
                                                    if len(Trigem)==1 and (1 in Trigem) and l==1:
                                                        Trigem.clear()
                                                        Trigem.append(1)
                                                    elif len(Trigem)>1: 
                                                        Trigem.clear()
                                                        if l ==1:
                                                            Trigem.append(1)
                                            if len(Trigem) == 4:
                                                Trigem_count+=1
                                                Trigem.clear()


                                            # Quadrageminy


                                            Quadgem = []
                                            Quadgem_count = 0
                                            for p,o in enumerate(bb):
                                                if len(Quadgem) == 5:
                                                    Quadgem_count+=1
                                                    try:
                                                        if bb[p] ==0 and bb[p+1]==0 and bb[p+2]==0 and bb[p+3]==1:
                                                            Quadgem.clear()
                                                            Quadgem.append(1)
                                                        else:
                                                            Quadgem.clear()
                                                    except:
                                                        Quadgem.clear()


                                                if len(Quadgem) ==0 and o ==1:
                                                    Quadgem.append(1)
                                                    
                                                elif len(Quadgem) ==1 and o ==0:

                                                          
                                                    Quadgem.append(0)
                                                elif len(Quadgem) ==2 and o ==0:
                                                    
                                                    Quadgem.append(0)
                                                elif len(Quadgem) ==3 and o ==0:
                                                    
                                                    Quadgem.append(0)
                                                elif len(Quadgem) ==4 and o ==1:
                                                   
                                                    Quadgem.append(1)
                                                else:
                                                    if len(Quadgem)==1 and (o in Quadgem) and o==1:
                                                        Quadgem.clear()
                                                        Quadgem.append(1)
                                                    elif len(Quadgem)>1: 
                                                    
                                                        Quadgem.clear()
                                                        if o ==1:
                                                            Quadgem.append(1)
                                            if len(Quadgem) == 5:
                                                Quadgem_count+=1
                                                Quadgem.clear()

                                            ll=bb
                                            couplet = []
                                            c_count=0
                                            for i in ll:
                                                if i==1:
                                                    couplet.append(1)
                                                    if len(couplet)==3:
                                                        c_count-=1
                                                        couplet.clear()

                                                    if len(couplet)==2: 
                                                        c_count+=1
                                                        
                                                    if 0 in couplet:
                                                        if c_count==0:
                                                            pass
                                                        else:
                                                            c_count-=1
                                                        couplet.clear()
                                                else:
                                                    couplet.clear()

                                                        
                                            triplet = []
                                            t_count=0
                                            for i in ll:
                                                if i==1:
                                                    triplet.append(1)
                                                    if len(triplet)>=4:
                                                        t_count-=1
                                                        triplet.clear()
                                                    if len(triplet)==3:
                                                        t_count+=1

                                                    if 0 in triplet:
                                                        if t_count==0:
                                                            pass
                                                        else:
                                                            t_count-=1
                                                        triplet.clear()
                                                        
                                                else:
                                                    triplet.clear()


                                            if int(HR)>100:
                                                vt = []
                                                vt_count=0
                                                for i in ll:
                                                    if i==1:
                                                        vt.append(1)
                                                        if len(vt)>=4:
                                                            vt_count+=1
                                                            vt.clear()
                                                        if 0 in vt:
                                                            if vt_count==0:
                                                                pass
                                                            else:
                                                                vt_count-=1
                                                            vt.clear()
                                                            
                                                    else:
                                                        vt.clear()
                                            

                                            if int(HR)>60 and int(HR)<=300:
                                                aivr = []
                                                aivr_count=0
                                                for i in ll:
                                                    if i==1:
                                                        aivr.append(1)
                                                        if len(aivr)>=4:
                                                            aivr_count+=1
                                                            aivr.clear()
                                                        if 0 in aivr:
                                                            if aivr_count==0:
                                                                pass
                                                            else:
                                                                aivr_count-=1
                                                            aivr.clear()
                                                            
                                                    else:
                                                        aivr.clear()


                                            if int(HR)<=60:
                                                ivr = []
                                                ivr_count=0
                                                for i in ll:
                                                    if i==1:
                                                        ivr.append(1)
                                                        if len(ivr)>=4:
                                                            ivr_count+=1
                                                            ivr.clear()
                                                        if 0 in ivr:
                                                            if ivr_count==0:
                                                                pass
                                                            else:
                                                                ivr_count-=1
                                                            ivr.clear()
                                                            
                                                    else:
                                                        ivr.clear()



                                            finaliso = actaulPVC.count(1) - Quadgem_count*2 - Trigem_count*2 - bigem_count*2 - c_count*2 - t_count*3
                                            if actaulPVC.count(1)>0:
    #                                            for pek in peaksdefined:
    #                                                result_data["threeLatter"].append({"Arrhythmia":"PVC-Isolated","startTime":pek,"endTime":pek,"percentage":actaulPVC.count(1)/len(rpeaks)*100})
    #                                            
                                                result_data.update({"Arrhythmia":'PVC-Isolated',"Vbeats":actaulPVC.count(1),"HR":int(HR),"ISOLATEDCOUNT":actaulPVC.count(1),"peakslocation":peaksdefined})
                                            if Quadgem_count>=1:
    #                                            for pek in peaksdefined:
    #                                                result_data["threeLatter"].append({"Arrhythmia":"PVC-Quadrigeminy","startTime":pek,"endTime":pek,"percentage":Quadgem_count*2/len(rpeaks)*100})

                                                result_data.update({"Arrhythmia":'PVC-Quadrigeminy',"Vbeats":bb.count(1),"HR":int(HR),"peakslocation":peaksdefined})
                                            if Trigem_count>=1:                                            
    #                                            for pek in peaksdefined:
    #                                                result_data["threeLatter"].append({"Arrhythmia":"PVC-Trigeminy","startTime":pek,"endTime":pek,"percentage":Trigem_count*2/len(rpeaks)*100})
                                                result_data.update({"Arrhythmia":'PVC-Trigeminy',"Vbeats":bb.count(1),"HR":int(HR),"peakslocation":peaksdefined})

                                            if bigem_count>=1:
    #                                            for pek in peaksdefined:
    #                                                result_data["threeLatter"].append({"Arrhythmia":"PVC-Bigeminy","startTime":pek,"endTime":pek,"percentage":bigem_count*2/len(rpeaks)*100})
                                                
                                                result_data.update({"Arrhythmia":'PVC-Bigeminy',"Vbeats":bb.count(1),"HR":int(HR),"peakslocation":peaksdefined})
                                            if c_count>=1:
    #                                            for pek in peaksdefined:
    #                                                result_data["threeLatter"].append({"Arrhythmia":"PVC-Couplet","startTime":pek,"endTime":pek,"percentage":c_count*2/len(rpeaks)*100})

                                                
                                                result_data.update({"Arrhythmia":'PVC-Couplet',"Vbeats":bb.count(1),"HR":int(HR),"COUPLETCOUNT":c_count,"peakslocation":peaksdefined})
                                            if t_count>=1:
    #                                            for pek in peaksdefined:
    #                                                result_data["threeLatter"].append({"Arrhythmia":"PVC-Triplet","startTime":pek,"endTime":pek,"percentage":t_count*3/len(rpeaks)*100})
                                                
                                                result_data.update({"Arrhythmia":'PVC-Triplet',"Vbeats":bb.count(1),"HR":int(HR),"TRIPLETCOUNT":t_count,"peakslocation":peaksdefined})
                                            if float(HR)>100.0:
                                                    if vt_count>=1 and bb.count(1)>12:
                                                        result_data.update({"Arrhythmia":'VT',"Vbeats":bb.count(1),"HR":int(HR),"peakslocation":peaksdefined})
                                                    if aivr_count>=1 and bb.count(1)<=12:
                                                        result_data.update({"Arrhythmia":'NSVT',"Vbeats":bb.count(1),"HR":int(HR),"peakslocation":peaksdefined})

                                            if float(HR)>60.0 and float(HR)<=100.0:
                                                    if aivr_count>=1:
                                                        result_data.update({"Arrhythmia":'NSVT',"Vbeats":bb.count(1),"HR":int(HR),"peakslocation":peaksdefined})
                                            
                                            if float(HR)<=60.0:
                                                    if ivr_count>=1:
                                                        result_data.update({"Arrhythmia":'IVR',"Vbeats":bb.count(1),"HR":int(HR),"peakslocation":peaksdefined})


                                            else:
                                                wideq = wide_qrs_detection(low_es, fs=200)
                                                if wideq["wideqrs_label"]=="Wide_QRS" and result_data["Arrhythmia"] not in ["PVC-Isolated", "PVC-Quadrigeminy", "PVC-Trigeminy","PVC-Bigeminy","PVC-Couplet","PVC-Triplet","VT","IVR","NSVT","AFIB","AFL"]:
                                                    widelens = len(wideq["wideqrs_index"])
                                                    if widelens>5 and (1 in newpvcswide):
                                                        result_data.update({"Arrhythmia":'WIDE-QRS',"Count":widelens,"peakslocation":peaksdefined})
                                                        d3 = result_data
                                                        finddata.append(d3)

                                            d3 = result_data
                                            finddata.append(d3)



                                        if result_data["Arrhythmia"] not in ["PVC-Isolated", "PVC-Quadrigeminy", "PVC-Trigeminy","PVC-Bigeminy","PVC-Couplet","PVC-Triplet","VT","IVR","NSVT","AFIB","AFL"] and int(timetaken)>6:
                                            apeds=[]
                                            r_index = detect_beats(low_es,200)
                                            for i in range(len(r_index)-1):
                                                m=r_index[i+1]-r_index[i]
                                                apeds.append(m*5/1000)

                                            variations=[]
                                            rrints=''
                                            for i in range(len(apeds)-1):
                                                
                                                variations.append(get_percentage_diff(apeds[i+1],apeds[i]))
                                            
                                            forPAC = Average(variations)
                                            jr_label = "Abnormal"
                                            try:
                                                r_peaks = r_index
                                                baseline_signal = baseline_construction_200(OriginalSignal, 131)
                                                pqrst_data = pqrst_detection(baseline_signal, fs=fs, thres=0.37, lp_thres=0.1, rr_thres = 0.15)
                                                junc_r_label = pqrst_data['R_Label']
                                                p_index = pqrst_data['P_Index']
                                                p_t = pqrst_data['P_T List']
                                                updated_union, junc_union = [], []
                                                pac_detect, junc_index = [], []
                                                for i in range(len(r_peaks) - 1):
                                                    fig, ax = plt.subplots(num=1, clear=True)
                                                    segment = low_es[r_peaks[i]-16:r_peaks[i + 1]+20]
                                                    ax.plot(segment,color='blue')
                                                    ax.axis(False)
                                                    fig.canvas.draw()
                                                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                                                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                                                    image = Image.fromarray(data)
                                                    resized_image = image.resize((360, 720), Image.LANCZOS)
                                                    plt.close(fig)
                                                    predictions,ids = prediction_model_PAC(resized_image)
                                                    
                                                    if str(ids) == "PAC" and float(predictions[3])>0.93: # 0.91
                                                        updated_union.append(1)
                                                        junc_union.append(0)
                                                        pac_detect.append(int(r_peaks[i]))
                                                        pac_detect.append(int(r_peaks[i+1]))
                                                    elif str(ids) == "Junctional" and float(predictions[1]) > 0.90:
                                                        junc_union.append(1)
                                                        updated_union.append(0)
                                                        junc_index.append(int(r_peaks[i]))
                                                        junc_index.append(int(r_peaks[i+1]))  
                                                    else:
                                                        updated_union.append(0)
                                                        junc_union.append(0)
                                                if junc_r_label == "Regular" and HR <= 60:
                                                    if junc_union:
                                                      junc_count = junc_union.count(1)
                                                      total_index = len(junc_union)
                                                      jr_model_percent = junc_count/ total_index
                                                      count = 0
                                                      new_threshold = 0.065 if HR > 50 else 0.06
                                                      for i in range(len(p_t)-1):
                                                          dis = (r_index[i+1]-p_t[i][-1])/fs
                                                          if dis <= new_threshold: count += 1
                                                      jr_abstraction_per = ((len(r_index)-1 - (len(p_index)-count))/(len(r_index)-1))
                                                      combined_percent = (jr_model_percent *0.2) + (jr_abstraction_per *0.8)
                                                      if combined_percent >= 0.75 and jr_model_percent >= 0.2:
                                                          jr_label = "Junctional_Rhythm" if HR > 40 else "Junctional_Bradycardia"
                                                      else:
                                                          jr_label = "Abnormal" 
                                            except Exception as e:
                                                print(e)
                                                updated_union=[0,0,0,0,0,0,0,0]
                                            if Average(variations)>0.20:
                                                actaulPAC = updated_union
                                                bbs = updated_union
                                            else:
                                                actaulPAC = [0,0,0,0,0,0,0,0]
                                                bbs = [0,0,0,0,0,0,0,0]
                                            bigem = []
                                            bigem_count= 0
                                            for q,k in enumerate(bbs):
                                                if len(bigem) == 3:
                                                    bigem_count+=1
                                                    try:
                                                        if bbs[q] ==0 and bbs[q+1]==1:
                                                            bigem.clear()
                                                            bigem.append(1)
                                                        else:
                                                            bigem.clear()
                                                    except:
                                                        bigem.clear()
                                                if len(bigem ) ==0 and k ==1:
                                                    bigem.append(1)
                                                elif len(bigem) ==1 and k ==0:
                                                    bigem.append(0)
                                                elif len(bigem) ==2 and k ==1:
                                                    bigem.append(1)
                                                else:
                                                    if len(bigem)==1 and (1 in bigem) and k==1:
                                                        bigem.clear()
                                                        bigem.append(1)
                                                    elif len(bigem)>1: 
                                                        bigem.clear()
                                                        if k ==1:
                                                            bigem.append(1)            
                                            if len(bigem) == 3:
                                                bigem_count+=1
                                                bigem.clear()

                                            # Trigeminy 
                                            Trigem = []
                                            Trigem_count = 0
                                            for m,l in enumerate(bbs):
                                                if len(Trigem) == 4:
                                                    Trigem_count+=1
                                                    try:
                                                        if bbs[m] ==0 and bbs[m+1]==0 and bbs[m+2]==1:
                                                            Trigem.clear()
                                                            Trigem.append(1)
                                                        else:
                                                            Trigem.clear()
                                                    except:
                                                        Trigem.clear()

                                                if len(Trigem) ==0 and l ==1:
                                                    Trigem.append(1)
                                                elif len(Trigem) ==1 and l ==0:
                                                    Trigem.append(0)
                                                elif len(Trigem) ==2 and l ==0:
                                                    Trigem.append(0)
                                                elif len(Trigem) ==3 and l ==1:
                                                    Trigem.append(1)
                                                else:
                                                    if len(Trigem)==1 and (1 in Trigem) and l==1:
                                                        Trigem.clear()
                                                        Trigem.append(1)
                                                    elif len(Trigem)>1: 
                                                        Trigem.clear()
                                                        if l ==1:
                                                            Trigem.append(1)
                                            if len(Trigem) == 4:
                                                Trigem_count+=1
                                                Trigem.clear()

                                            # Quadrageminy
                                            Quadgem = []
                                            Quadgem_count = 0
                                            for p,o in enumerate(bbs):
                                                if len(Quadgem) == 5:
                                                    Quadgem_count+=1
                                                    try:
                                                        if bbs[p] ==0 and bbs[p+1]==0 and bbs[p+2]==0 and bbs[p+3]==1:
                                                            Quadgem.clear()
                                                            Quadgem.append(1)
                                                        else:
                                                            Quadgem.clear()
                                                    except:
                                                        Quadgem.clear()
                                                if len(Quadgem) ==0 and o ==1:
                                                    Quadgem.append(1)             
                                                elif len(Quadgem) ==1 and o ==0:                                     
                                                    Quadgem.append(0)
                                                elif len(Quadgem) ==2 and o ==0:            
                                                    Quadgem.append(0)
                                                elif len(Quadgem) ==3 and o ==0:            
                                                    Quadgem.append(0)
                                                elif len(Quadgem) ==4 and o ==1:                           
                                                    Quadgem.append(1)
                                                else:
                                                    if len(Quadgem)==1 and (o in Quadgem) and o==1:
                                                        Quadgem.clear()
                                                        Quadgem.append(1)
                                                    elif len(Quadgem)>1:                                      
                                                        Quadgem.clear()
                                                        if o ==1:
                                                            Quadgem.append(1)
                                            if len(Quadgem) == 5:
                                                Quadgem_count+=1
                                                Quadgem.clear()

                                            # couplet
                                            ll=bbs
                                            couplet = []
                                            pac_c_count=0
                                            for i in ll:
                                                if i==1:
                                                    couplet.append(1)
                                                    if len(couplet)==3:
                                                        pac_c_count-=1
                                                        couplet.clear()
                                                    if len(couplet)==2: 
                                                        pac_c_count+=1    
                                                    if 0 in couplet:
                                                        if pac_c_count==0:
                                                            pass
                                                        else:
                                                            pac_c_count-=1
                                                        couplet.clear()
                                                else:
                                                    couplet.clear()

                                            # triplet            
                                            triplet = []
                                            pac_t_count=0
                                            for i in ll:
                                                if i==1:
                                                    triplet.append(1)
                                                    if len(triplet)>=4:
                                                        pac_t_count-=1
                                                        triplet.clear()
                                                    if len(triplet)==3:
                                                        pac_t_count+=1
                                                    if 0 in triplet:
                                                        if pac_t_count==0:
                                                            pass
                                                        else:
                                                            pac_t_count-=1
                                                        triplet.clear()    
                                                else:
                                                    triplet.clear()

                                            if int(HR)>=120:
                                                at = []
                                                at_count=0
                                                for i in ll:
                                                    if i==1:
                                                        at.append(1)
                                                        if len(at)>=4:
                                                            at_count+=1
                                                            at.clear()
                                                        if 0 in at:
                                                            if at_count==0:
                                                                pass
                                                            else:
                                                                at_count-=1
                                                            at.clear()
                                                            
                                                    else:
                                                        at.clear()
                                            
                                            finaliso = actaulPAC.count(1) - Quadgem_count*2 - Trigem_count*2 - bigem_count*2 - pac_c_count*2 - pac_t_count*3
                                            if actaulPAC.count(1)>0 and forPAC>6:
                                                result_data.update({"Arrhythmia":'PAC-Isolated',"PACTOTALCOUNT":actaulPAC.count(1),"HR":int(HR),"ISOPAC":abs(finaliso)})
                                                d3 = result_data
                                                finddata.append(d3)
                                            if Quadgem_count>=1 and forPAC>6:
                                                result_data.update({"Arrhythmia":'PAC-Quadrigeminy',"PACTOTALCOUNT":bbs.count(1),"HR":int(HR),"PACQUADRIGEMCOUNT":Quadgem_count})
                                                d3 = result_data
                                                finddata.append(d3)

                                            if Trigem_count>=1 and forPAC>6:
                                                result_data.update({"Arrhythmia":'PAC-Trigeminy',"PACTOTALCOUNT":bbs.count(1),"HR":int(HR),"PACTRIGEMCOUNT":Trigem_count})
                                                d3 = result_data
                                                finddata.append(d3)

                                            if bigem_count>=1 and forPAC>6:
                                                result_data.update({"Arrhythmia":'PAC-Bigeminy',"PACTOTALCOUNT":bbs.count(1),"HR":int(HR),"PACBIGEMCOUNT":bigem_count})
                                                d3 = result_data
                                                finddata.append(d3)

                                            if pac_c_count>=1 and forPAC>6:
                                                result_data.update({"Arrhythmia":'PAC-Couplet',"PACTOTALCOUNT":bbs.count(1),"HR":int(HR),"PACCOUPLETCOUNT":pac_c_count})
                                                d3 = result_data
                                                finddata.append(d3)

                                            if pac_t_count>=1 and forPAC>6:
                                                result_data.update({"Arrhythmia":'PAC-Triplet',"PACTOTALCOUNT":bbs.count(1),"HR":int(HR),"PACTRIPLETCOUNT":pac_t_count})
                                                d3 = result_data
                                                finddata.append(d3)

                                            if float(HR)>=150.0:
                                                if at_count>=1:
                                                    result_data.update({"Arrhythmia":'SVT',"PACTOTALCOUNT":bbs.count(1),"HR":int(HR)})
                                                    d3 = result_data
                                                    finddata.append(d3)

                                        tf.keras.backend.clear_session()
                                        if int(HR)<=80 and int(timetaken)>6 and result_data["Arrhythmia"] not in ["PVC-Isolated", "PVC-Quadrigeminy", "PVC-Trigeminy","PVC-Bigeminy","PVC-Couplet","PVC-Triplet","VT","IVR","NSVT","AFIB","AFL"]:
                                            try:
                                                # baseline_signal = baseline_construction_200(OriginalSignal, 131)
                                                # jnrhy = jr_detection(baseline_signal,tf_cwt_model, 200)
                                                jnrhy = jr_label
                                                if jnrhy=="Junctional_Rhythm":
                                                    result_data.update({"Arrhythmia":'JN-RHY'})
                                                    d3 = result_data
                                                    finddata.append(d3)
                                                elif jnrhy=="Junctional_Bradycardia":                                
                                                    result_data.update({"Arrhythmia":'JN-BR'})
                                                    d3 = result_data
                                                    finddata.append(d3)
                                            except:
                                                print("JN ISSUE")

                                        if result_data['Arrhythmia']=='' and rrint == "REGULAR":
                                            result_data.update({"Arrhythmia":'Normal'})
                                            d3 = result_data
                                            finddata.append(d3)
                                            
                                            
                                        #datetimee.clear()
                                        #newlist.clear()

                                        if (result_data['Arrhythmia']=='BR' or result_data['Arrhythmia']=='Short Pause' or result_data['Arrhythmia']=='Normal' or result_data['Arrhythmia']=='') and int(HR)<80 and int(timetaken)>=7:
                                            try:
                                                block_na = lowpass_11(naa)
                                                labelss = block_process(block_na, 200)
                                                final_label,ei_ti_block = block_model_check(block_na, 200, labelss)
                                                if final_label == "III Degree": 
                                                    result_data.update({"Arrhythmia":'III Degree'})
                                                    for ei_ti_blockdata in ei_ti_block:
                                                            result_data["threeLatter"].append(ei_ti_blockdata)
                                                    d3 = result_data
                                                    finddata.append(d3)
                                                elif final_label == "MOBITZ-I":
                                                    result_data.update({"Arrhythmia":'MOBITZ-I'})
                                                    for ei_ti_blockdata in ei_ti_block:
                                                            result_data["threeLatter"].append(ei_ti_blockdata)

                                                    d3 = result_data
                                                    finddata.append(d3)
                                                elif final_label == "MOBITZ-II":
                                                    result_data.update({"Arrhythmia":'MOBITZ-II'})
                                                    for ei_ti_blockdata in ei_ti_block:
                                                            result_data["threeLatter"].append(ei_ti_blockdata)
                                                    
                                                    d3 = result_data
                                                    finddata.append(d3)

                                                else:
                                                    if result_data["Arrhythmia"] not in ["PVC-Isolated", "PVC-Quadrigeminy", "PVC-Trigeminy","PVC-Bigeminy","PVC-Couplet","PVC-Triplet","VT","IVR","NSVT","WIDE-QRS","SVT",'PAC-Triplet','PAC-Couplet','PAC-Bigeminy','PAC-Trigeminy','PAC-Quadrigeminy','PAC-Isolated','ABNORMAL','Long Pause','Short Pause'] and rrint == "IRREGULAR":
                                                        result_data.update({"Arrhythmia":'SINUS-ARR'})
                                                        d3 = result_data
                                                        finddata.append(d3)
    #                                                else:
    #                                                    result_data.update({"Arrhythmia":'ABNORMAL'})
    #                                                    d3 = result_data
    #                                                    finddata.append(d3)

                                            except Exception as e:
                                                print("MOBITZ-I,MOBITZ-II,III Degree issue",e)
                                                
                                        if result_data['Arrhythmia']=='' and rrint == "IRREGULAR":
                                                result_data.update({"Arrhythmia":'SINUS-ARR'})
                                                d3 = result_data
                                                finddata.append(d3)

    ##                                    if result_data['Arrhythmia']=='':
    ##                                            result_data.update({"Arrhythmia":'ABNORMAL'})
    ##                                            d3 = result_data
    ##                                            finddata.append(d3)

                                        if (result_data['Arrhythmia']=='Normal' or rrint=="REGULAR") and int(HR)<100 and int(timetaken)>7:
                                            try:
                                                b, a = signal.butter(3, 0.3, btype='lowpass', analog=False)
                                                low_passed = signal.filtfilt(b, a, newdada)
                                                rpeaksnew = detect_beats(low_passed, float(fa))
                                                _, waves_peak = nk.ecg_delineate(low_passed, unique(rpeaksnew), sampling_rate=fa, method="peak")
                                                signal_dwt, waves_dwt = nk.ecg_delineate(low_passed, unique(rpeaksnew), sampling_rate=fa, method="dwt")
                                                firststblock =[]
                                                rpeaks1 = unique(rpeaksnew)
                                                Ppeaks = unique(waves_peak['ECG_P_Peaks'])
                                                

                                                #dwt method
                                                for iii in range(len(Ppeaks)):
                                                    if np.isnan(Ppeaks[iii]):
                                                        Ppeaks[iii]=0
                                                        firststblock.append(0)
                                                    else:
                                                        difff = rpeaksnew[iii]-Ppeaks[iii]
                                                        firststblock.append((difff*5)/1000)
                                                firstblockfinal=[]
                    ##                            print("FIRST",firststblock)
                                                for findtime in firststblock:
                                                    if findtime>0.26:
                                                        firstblockfinal.append(findtime)

                                                if len(firstblockfinal)>5:
                                                    result_data.update({"Arrhythmia":'I DEGREE',"PRInterval":firstblockfinal[0]})
                                                    d3 = result_data
                                                    finddata.append(d3)
                                                else:
                                                    pass
                                            except:
                                                print("I degree issue")




    ##                                    if result_data['Arrhythmia']=='Normal' and int(HR)<100 and int(timetaken)>7:
    ##                                        try:
    ##                                            labelss = block_process(naa, 200)
    ##                                            if labelss=="3rd Degree block":
    ##                                                result_data.update({"Arrhythmia":'III Degree'})
    ##                                                d3 = result_data
    ##                                                finddata.append(d3)
    ##                                            else:
    ##                                                pass
    ##                                        except:
    ##                                            print("MOBITZ-I,MOBITZ-II,III Degree issue")
                                            
    #                                    if int(HR)<80 and int(timetaken)>7:
    #                                        try:
    #                                            labelss = block_process(naa, 200)
    #                                            if labelss=="3rd Degree block":
    #                                                result_data.update({"Arrhythmia":'III Degree'})
    #                                                d3 = result_data
    #                                                finddata.append(d3)
    #                                            elif labelss=="Mobitz I": # ------------ NEW added
    #                                                result_data.update({"Arrhythmia":'MOBITZ-I'})
    #                                                d3 = result_data
    #                                                finddata.append(d3)
    #                                            elif labelss=="Mobitz II":
    #                                                result_data.update({"Arrhythmia":'MOBITZ-II'})
    #                                                d3 = result_data
    #                                                finddata.append(d3)
    #                                            else:
    #                                                pass
    #                                        except Exception as e:
    #                                            print("MOBITZ-I,MOBITZ-II,III Degree issue",e)                                            
                                            
                                        try:    
                                            pace_label, pacemaker_index, r_index, q_index, s_index, p_index = pacemaker_detect(low_es, fs= 200)
                                            if pace_label == "Atrial_Pacemaker":
                                                result_data.update({"ATRIAL_PACEMAKER":True})
                                                d3 = result_data
                                                finddata.append(d3)


                                            if pace_label == "Ventricular_Pacemaker":
                                                result_data.update({"VENTRICULAR_PACEMAKER":True})
                                                d3 = result_data
                                                finddata.append(d3)


                                            if pace_label == "Atrial_&_Ventricular_pacemaker":
                                                result_data.update({"AV_PACEMAKER":True})
                                                d3 = result_data
                                                finddata.append(d3)
                                        except:
                                            print("PACEMAKER ISSUE")

                                        try:
                                            naMI = np.array(newdada)
                                            if int(HR)<100 and int(timetaken)>6 and int(version)==5 and rrint=="REGULAR":
                                                #label_mi = process_signal(naMI, 200)
                                                label_rlbbb = LBBB_RBBB(b_es,rpeaks,imageresource)

                                                all_lead_data = data_convert_MI(sorted_data) #, path=save, patient_id
                                                if len(all_lead_data) != 0 and len(all_lead_data['II'].values) > 500:
                                                    # label_mi = MIDetection_MI(all_lead_data, 200).check_MI()
                                                    label_mi = check_mi_model(all_lead_data, imageresource)
                                                    if lbbb_rbbb_label == "LBBB": # if label_mi == "LBBB":
                                                        result_data.update({"Arrhythmia":'ABNORMAL',"MI":"LBBB"})
                                                        d3 = result_data
                                                        finddata.append(d3)
                                                    elif lbbb_rbbb_label == 'RBBB': #label_mi == "RBBB":
                                                        result_data.update({"Arrhythmia":'ABNORMAL',"MI":"RBBB"})
                                                        d3 = result_data
                                                        finddata.append(d3)                                               
                                                    elif label_mi == "Inferior STEMI":
                                                        result_data.update({"Arrhythmia":'ABNORMAL',"MI":"Inferior MI"})
                                                        d3 = result_data
                                                        finddata.append(d3)
                                                    elif label_mi == "Lateral STEMI":
                                                        result_data.update({"Arrhythmia":'ABNORMAL',"MI":"Lateral MI"})
                                                        d3 = result_data
                                                        finddata.append(d3)
                                                    elif label_mi == "T_wave_Abnormality":
                                                        result_data.update({"Arrhythmia":'ABNORMAL',"MI":"T wave Abnormality"})
                                                        d3 = result_data
                                                        finddata.append(d3)
                                                #    elif label_mi == "STEMI":
                                                #        result_data.update({"Arrhythmia":'ABNORMAL',"MI":"STEMI"})
                                                #        d3 = result_data
                                                #        finddata.append(d3)
                                                #    elif label_mi == "NSTEMI":
                                                #        result_data.update({"Arrhythmia":'ABNORMAL',"MI":"NSTEMI"})
                                                #        d3 = result_data
                                                #        finddata.append(d3)
                                                #    elif label_mi == "Anterior STEMI":
                                                #        result_data.update({"Arrhythmia":'ABNORMAL',"MI":"Anterior STEMI"})
                                                #        d3 = result_data
                                                #        finddata.append(d3)
                                                #    elif label_mi == "Posterior STEMI":
                                                #        result_data.update({"Arrhythmia":'ABNORMAL',"MI":"Posterior STEMI"})
                                                #        d3 = result_data
                                                #        finddata.append(d3)
                                                
                                                
                                                
                                        except:
                                            print("Error in MI")
                                            
                                        if result_data['Arrhythmia']=='' and int(HR)>100:
                                                result_data.update({"Arrhythmia":'TC'})
                                                d3 = result_data
                                                finddata.append(d3)
                                            
                                        if result_data['Arrhythmia']=='' and int(HR)<60:
                                                result_data.update({"Arrhythmia":'BR'})
                                                d3 = result_data
                                                finddata.append(d3)
                                            
                                        if result_data['Arrhythmia']=='' and int(HR)>60 and int(HR)<100:
                                                result_data.update({"Arrhythmia":'Normal'})
                                                d3 = result_data
                                                finddata.append(d3)
                                        
                                        newdata = [i for n, i in enumerate(finddata) if i not in finddata[n + 1:]]

                                        print("LOG:",newdata)
                                        if allarr == 'All-Arrhythmia':
                                            if int(result_data['HR'])<=20:
                                                mintime = min(datetimee)
                                                maxtime = max(datetimee)
                                                result_data = [{"patient":dd["patient"],"HR":0,"starttime":mintime,"endtime":maxtime,"Arrhythmia":'Artifacts','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":[],"RR":0,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]
                                                print("FinalLog:",result_data)
                                                client.publish(topic_y,json.dumps(result_data),qos=2)
        ##                                        item = queue.pop()
                                            else:
                                                
                                                for i in newdata:
                                                    x = mycol.insert_one(dict(i))
                                                client.publish(topic_y,json.dumps(newdata),qos=2)
        ##                                        item = queue.pop()
                                                end = time.time()
                                                print("Time Taken:",end-start)
                                        else:
                                            client.publish(topic_y,json.dumps(newdata),qos=2)
        ##                                    item = queue.pop()
                                            end = time.time()
                                            print("Time Taken:",end-start)

                                    except Exception as e:
                                                print("DATA Corrupted inside",e)
                                                tb = traceback.extract_tb(e.__traceback__)
                                                line_number = tb[-1][1]  # Extract the line number from the last entry in the traceback
                                                print(f"Exception occurred on line {tb}: {tb}")
                                                result_data = [{"patient":dd["patient"],"HR":0,"starttime":mintime,"endtime":maxtime,"Arrhythmia":'Artifacts','kit':dd["kit"],'position':positionFinal,"beats":0,"RRInterval":0,"PRInterval":0,"QTInterval":0,"QRSComplex":0,"STseg":0,"PRseg":0,"Vbeats":0,"noOfPause":0,"ISOLATEDCOUNT":0,"COUPLETCOUNT":0,"TRIPLETCOUNT":0,"PACTRIPLETCOUNT":0,"PACCOUPLETCOUNT":0,"ISOPAC":0,"PACTOTALCOUNT":0,"trigger":trigger,"offline":True,"rpmId":rpmId,"version":version,"patientData":patientData,"coordinates":coordinates,"datalength":datalength,"HRV":[],"RR":0,"battery":battery ,"memoryUtilized": memoryUtilized,"sysncDataReaming":sysncDataReaming,"mobileBaterry":mobileBaterry}]                                                   
                                                print("LOG:",result_data)
                                                client.publish(topic_y,json.dumps(result_data),qos=2)
    ##                                        client.on_message = on_message


            except Exception as e:
                print("Data Failure Inside,"+str(e))
                tb = traceback.extract_tb(e.__traceback__)
                line_number = tb[-1][1]  # Extract the line number from the last entry in the traceback
                print(f"Exception occurred on line {tb}: {tb}")

        else:
            time.sleep(0.1)
            pass

##                client.on_message = on_message

##        client.on_message = on_message
        



def run():
    while True:
        try:
            try:
                client = connect_mqtt()
                subscribe(client)
                client.loop_forever()
            except:
                print("Data Failure Outside")
                client = connect_mqtt()
                subscribe(client)
                client.loop_forever()
        except Exception as e:
            print("Outside:",e)

if __name__ == '__main__':
    run()

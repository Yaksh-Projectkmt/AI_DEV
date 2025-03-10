from paho.mqtt import client as mqtt_client
from queue import Queue
from collections import deque
import concurrent.futures
import ssl
import json
import threading
import time
import random
from paho.mqtt import client as mqtt_client
import ast
import time
from dateutil import parser
import matplotlib.pyplot as plt
import time
import csv
import math
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import ssl
import numpy as np
import random
import tools as st
import utils
from scipy.signal import medfilt
from scipy.stats import pearsonr
from scipy import sparse, signal
from scipy.sparse.linalg import spsolve
from statistics import pvariance
import scipy.fftpack
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import scipy.signal as signal
import os
import pandas
from math import sqrt
import numpy as np
import datetime
import math
from scipy.ndimage import label
import json
import pywt as pw
import neurokit2 as nk
import seaborn as sns
import collections
import logging

#configuration
##broker = 'oomcardiotest.projectkmt.com'
##port = 8883
##topic = "oom/ecg/rawData"
##client_id = f'python-mqtt-{random.randint(0, 100)}'
##username = 'kmt'
##password = 'dVBbS3NxMtmzD438'

broker = 'oomcardiodev.projectkmt.com'
port = 8883
client_id = f'python-mqtt-{random.randint(900000, 1000000)}'
username = 'kmt'
password = 'Kmt123'


##broker = 'oomcardiotest.projectkmt.com'
##port = 8883
##client_id = f'{random.randint(1000000000000000, 2000000000000000)}'
##username = 'kmt'
##password = 'dVBbS3NxMtmzD438'


futures = {}
executor = concurrent.futures.ThreadPoolExecutor(max_workers=15)

collect_data_queues = {}
send_data_queues = {}
vol_data = {}
vol_data_list = {}
hr_list = {}
counter = {}
new_hex_data = {}
client=''
def connect_mqtt() -> mqtt_client:
    global client
    def on_connect(client, userdata, flags, rc,protocol):
        if rc == 0:
            print("Connected to MQTT Broker!")
            topic = "oom/ecg/rawData"
            client.subscribe(topic,qos=2)
        else:
            print("Failed to connect, return code %d\n", rc)
            time.sleep(0.1)
            connect_mqtt()
    client = mqtt_client.Client(client_id,protocol=mqtt_client.MQTTv5)
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    client.tls_set_context(context)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    try:
        client.connect(broker, port)
    except Exception as e:
        print("Error connecting to MQTT Broker", e)
        time.sleep(0.1)
        connect_mqtt()
    return client

def subscribe(client: mqtt_client):

    def on_message(client, userdata, msg):

        try:
            new_data = json.loads(msg.payload.decode("utf-8"))
            new_patient_id = new_data["patient"]
            data = new_data["data"]
            new_list = []
            leadlist = []
            version=[]
            for i in range(len(data)):
                new_list.append(data[i]["data"])
                leadlist.append(data[i]['lead'])
                version.append(data[i]['version'])
            if 0 not in leadlist:
                pass
            else:
                if new_patient_id in collect_data_queues.keys():
                    collect_data_queues[new_patient_id].extend(new_list)

                else:
                    send_data_queues[new_patient_id] = deque([])
                    collect_data_queues[new_patient_id] = deque(new_list)

                    futures[new_patient_id] = executor.submit(collect_data, new_patient_id,version)
                    futures[new_patient_id].add_done_callback(lambda f: cleanup_thread(f.result(),new_patient_id))
        except Exception as e:
            print("Error processing MQTT message: %s", e, "\n", msg.payload)

    client.on_message = on_message

def collect_data(patient_id,version):
    
    hr_list[patient_id] = deque([])
    counter[patient_id] = 0
    new_hex_data[patient_id] = "".join(collect_data_queues[patient_id])
    
    vol_data[patient_id] = voltage_converter(new_hex_data[patient_id],version)
    
    vol_data[patient_id] = np.array(vol_data[patient_id])
##    hr =  one_min_hr(patient_id)
    one_min_hr(patient_id)
    collect_data_queues[patient_id].clear()

    while True:

        if len(collect_data_queues[patient_id]) != 0:
            send_data_queues[patient_id].clear()
            
            if len(collect_data_queues[patient_id]) >= 6:
                for i in range (6):
                    send_data_queues[patient_id].append(collect_data_queues[patient_id].popleft())
            elif len(collect_data_queues[patient_id]) < 6 and len(collect_data_queues[patient_id]) != 0 :
                while len(collect_data_queues[patient_id]) != 0:
                    send_data_queues[patient_id].append(collect_data_queues[patient_id].popleft())
                    
            new_hex_data[patient_id] = "".join(send_data_queues[patient_id])
            new_hex_data[patient_id] = voltage_converter(new_hex_data[patient_id],version)
            vol_data[patient_id] = list(vol_data[patient_id])
            vol_data[patient_id].extend(new_hex_data[patient_id])
            vol_data[patient_id] = np.array(vol_data[patient_id][-3000:])
            hr =  one_min_hr(patient_id)
            
            counter[patient_id] = 0            
            
        counter[patient_id] += 1

        if counter[patient_id] > 600:
            del collect_data_queues[patient_id]
            del send_data_queues[patient_id]
            del vol_data[patient_id]
            del hr_list[patient_id]
            del counter[patient_id]
            del new_hex_data[patient_id]
##            del vol_data_list[patient_id]
            
            return f'Done with thread {patient_id}'

        time.sleep(0.1)

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
    init_ecg = 6  # seconds for initialization
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

def hr_count(ecg_signal, fs=200):
    r_index = hamilton_segmenter(signal = ecg_signal)["rpeaks"]
    
    cal_sec = ecg_signal.shape[0]/fs
    if cal_sec != 0:
        hr = round(len(r_index)*60/cal_sec)
        return hr
    return 0

def detect_rpeaks_eq(ecg, rate, ransac_window_size=3.35, lowfreq=5.0, highfreq=13.0):
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
    lp_energy = gaussian_filter1d(lp_energy, rate / 14.0)
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
    
def one_min_hr(patient_id, fs = 200):
    global client
    try:
        hr = hr_count(vol_data[patient_id], fs=fs)
        if len(hr_list[patient_id]) == 5:
            hr_list[patient_id].append(hr)
            hr_list[patient_id].popleft()
        elif len(hr_list[patient_id]) < 5:
            hr_list[patient_id].append(hr)
        mean_hr = round(np.mean(hr_list[patient_id]))


        ecg_signal = vol_data[patient_id]
        fa=200
        naa = np.array(ecg_signal)
        ecg_signal = MinMaxScaler(feature_range=(0,4)).fit_transform(naa.reshape(-1,1)).squeeze()

        #rpeaks = hamilton_segmenter(signal = ecg_signal)["rpeaks"]
        rpeaks = detect_rpeaks_eq(ecg_signal,200)
    ##    rpeaks = detect_beats(naa, float(fa))
    ##    times = len(dataconversion)/200
    ##    HR = int(60*int(len(rpeaks))/times)

        _, waves_peak = nk.ecg_delineate(naa, rpeaks, sampling_rate=200, method="peak")
        signal_dwt, waves_dwt = nk.ecg_delineate(naa, rpeaks, sampling_rate=200, method="dwt")
        RRintervallist=[]
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


        try:
            Ppeak = waves_peak['ECG_P_Peaks'][2]
            Rpeak = rpeaks[2]
            Ppeak = int(Ppeak)*5
            Rpeak = int(Rpeak)*5
            PRpeaks = abs(Rpeak-Ppeak)
        except:
            PRpeaks = "0"
        try:
            Tpeak = waves_peak['ECG_T_Peaks'][2]
            Qpeak = waves_peak['ECG_Q_Peaks'][2]
            Tpeak = int(Tpeak)*5
            Qpeak = int(Qpeak)*5
            QTpeaks = abs(Tpeak-Qpeak)
        except:
            QTpeaks="0"
            

        try:
            Speak = waves_peak['ECG_S_Peaks'][2]
            Qpeak = waves_peak['ECG_Q_Peaks'][2]
            Speak = int(Speak)*5
            Qpeak = int(Qpeak)*5
            SQpeaks = abs(Speak-Qpeak)
        except:
            SQpeaks = "0"

        try:
            Spa = waves_peak['ECG_S_Peaks'][2]
            Ton = waves_dwt['ECG_T_Onsets'][2]
            Spa = int(Spa)*5
            Ton = int(Ton)*5
            STseg = abs(Ton-Spa)
        except:
            STseg = "0"

            
        
        try:
            PP = waves_dwt['ECG_P_Offsets']
            RRO = waves_dwt['ECG_R_Onsets']
            if math.isnan(PP[2]) or math.isnan(RRO[2]):
                PRseg = "0"
            else:
                PPIn = int(PP[2])*5
                RRon = int(RRO[2])*5
                PRseg =  abs(PPIn - RRon)
        
        except:
                PRseg = "0"
    ##    mintime = min(datetimee)
    ##    maxtime = max(datetimee)
        result_data = {"patient":patient_id,"HR":str(mean_hr),"beats":len(rpeaks),"RRInterval":str(SAf[0]),"PRInterval":str(PRpeaks),"QTInterval":str(QTpeaks),"QRSComplex":str(SQpeaks),"STseg":str(STseg),"PRseg":str(PRseg),"ecgPackage":"All-Arrhythmia"}
        topic_y = "oom/ecg/processedDataThreeSec/"+str(patient_id)
        print("LOG of "+topic_y+":",result_data)
        client.publish(topic_y,json.dumps(result_data),qos=2)

    except Exception as e:
        print(e)
        pass
        #print("Unknown Execption Occur")
        


    
##    return mean_hr

def voltage_converter(data,version):
    vol_list = []
    start = 0
    step = 4
    while start < len(data)-4:
        temp_list = data[start:start+step]
        high = temp_list[2]+temp_list[3]
        low = temp_list[0]+ temp_list[1]
        highdec = int(str(high), 16)
        lowdec = int(str(low),16)
        val = (int(highdec)*256)+(int(lowdec))
        if 5 in version:
          val = ((val + 32768) % 65536) - 32768
          voltage = (4.6/4095)*val/4
        else:
          voltage = (4.6/4095)*val
        vol_list.append(voltage)
        start+=4
    return vol_list

def cleanup_thread(result, patient_id):
    del futures[patient_id]

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
            time.sleep(0.1)
if __name__ == '__main__':
    run()

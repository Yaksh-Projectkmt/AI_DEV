# -*- coding: utf-8 -*-
import time
import pymongo
from paho.mqtt import client as mqtt_client
import ast
import json
import ssl
import random
import datetime
from dateutil import parser

##myclient = pymongo.MongoClient("mongodb://hopsadmin:fN6ZPg02713u8X9l@139.59.75.40:27017/ecgs1?authSource=admin&retryWrites=true&w=majority")
myclient = pymongo.MongoClient("mongodb://localhost:27017")

mydb = myclient["test"]
test1 = mydb["Test1"]
# test2 = mydb["Test2"]
# test3 = mydb["Test3"]


##broker = 'oomcardiotest.projectkmt.com'
##port = 8883
##client_id = f'{random.randint(6500, 10000)}'
##username = 'kmt'
##password = 'dVBbS3NxMtmzD438'

broker = 'oomcardiodev.projectkmt.com'
port = 8883
client_id = f'python-mqtt-{random.randint(0, 500000)}'
username = 'kmt'
password = 'Kmt123'

def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc, protocol):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id,protocol=mqtt_client.MQTTv5)
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    client.tls_set_context(context)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(client,msg):
        topic = "oom/ecg/episodes"
        time.sleep(1)
        msg = f"{msg}"
        print(msg)
        result = client.publish(topic,msg)
        status = result[0]
        print(status)
        print("Data Write Successfully to channel:oom/ecg/episodes")


def countdown(t):
        while t:
                mins, secs = divmod(t, 60)
                timer = '{:02d}:{:02d}'.format(mins, secs)
                time.sleep(1)
                t -= 1
        return True
##        mainanalysis()


def mainanalysis():
    # print("start")
    while True:
        if countdown(10):
##            print("start")
            try:
                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"AFIB"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"AFIB"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("1:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                      
                    starttime = []
                    endtime = []
                    hr = []
                    pos = []
                    arr = ''
                    try:
                        hr.clear()
                        # print("AFIB CHECK")
                        for i in j:
                            # print(i)
                            if i["Arrhythmia"]=="AFIB":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "AFIB"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        # print("maindict:",maindict,hr)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "AFIB":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("AFIB",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"AFIB"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                # print(json.dumps(maindict))
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except:
                        pass
            
                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"PVC-Bigeminy"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"PVC-Bigeminy"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("2:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos = []
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="PVC-Bigeminy":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "PVC-Bigeminy"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "PVC-Bigeminy":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("PVC-Bigeminy",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"PVC-Bigeminy"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                # print(maindict)
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except:
                        pass



                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"PVC-Trigeminy"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"PVC-Trigeminy"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("3:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos = []
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="PVC-Trigeminy":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "PVC-Trigeminy"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "PVC-Trigeminy":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("PVC-Trigeminy",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"PVC-Trigeminy"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except:
                        pass



                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"PVC-Quadrigeminy"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"PVC-Quadrigeminy"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("4:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos = []
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="PVC-Quadrigeminy":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "PVC-Quadrigeminy"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "PVC-Quadrigeminy":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("PVC-Quadrigeminy",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"PVC-Quadrigeminy"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except:
                        pass








                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"VT"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"VT"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("5:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="VT":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "VT"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "VT":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("VT",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"VT"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except:
                        pass




                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"IVR"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"IVR"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("6:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="IVR":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "IVR"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "IVR":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("IVR",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"IVR"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except:
                        pass




                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"NSVT"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"NSVT"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("7:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="NSVT":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "NSVT"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "NSVT":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("NSVT",datadata)
                            if datadata>10:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"NSVT"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except:
                        pass

                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"MOBITZ-II"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"MOBITZ-II"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("8:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="MOBITZ-II":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "MOBITZ-II"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "MOBITZ-II":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("MOBITZ-II",datadata)
                            if datadata> 15: # 40
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"MOBITZ-II"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass


                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"I DEGREE"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"I DEGREE"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("9:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="I DEGREE":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "I DEGREE"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "I DEGREE":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("I_DEGREE",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"I DEGREE"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass



                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"SINUS-ARR"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"SINUS-ARR"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("10:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="SINUS-ARR":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "SINUS-ARR"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "SINUS-ARR":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("SINUS-ARR",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"SINUS-ARR"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass



                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"JN-RHY"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"JN-RHY"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("11:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="JN-RHY":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "JN-RHY"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "JN-RHY":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("JN-RHY",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"JN-RHY"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass



                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"JN-BR"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"JN-BR"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("12:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="JN-BR":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "JN-BR"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "JN-BR":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("JN-BR",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"JN-BR"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass



                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"PAC-Quadrigeminy"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"PAC-Quadrigeminy"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("13:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="PAC-Quadrigeminy":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "PAC-Quadrigeminy"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "PAC-Quadrigeminy":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("PAC-Quadrigeminy",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"PAC-Quadrigeminy"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass


                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"PAC-Trigeminy"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"PAC-Trigeminy"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("14:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="PAC-Trigeminy":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "PAC-Trigeminy"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "PAC-Trigeminy":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("PAC-Trigeminy",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"PAC-Trigeminy"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass




                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"PAC-Bigeminy"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"PAC-Bigeminy"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("16:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="PAC-Bigeminy":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "PAC-Bigeminy"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "PAC-Bigeminy":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("PAC-Bigeminy",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"PAC-Bigeminy"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass

                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"SVT"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"SVT"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("17:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="SVT":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "SVT"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "SVT":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("SVT",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"SVT"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass



                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"III Degree"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"III Degree"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("18:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="III Degree":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "III Degree"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "III Degree":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("III Degree",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"III Degree"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass




                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"MOBITZ-I"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"MOBITZ-I"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("19:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="MOBITZ-I":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "MOBITZ-I"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "MOBITZ-I":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("MOBITZ-I",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"MOBITZ-I"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass



                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"AFL"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"AFL"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("20:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="AFL":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "AFL"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "AFL":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("AFL",datadata)
                            if datadata>40:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"AFL"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass




                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"VFIB"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"VFIB"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("21:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="VFIB":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "VFIB"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "VFIB":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("VFIB",datadata)
                            if datadata>100:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"VFIB"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass


                insert_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"ASYSTOLE"}},{"_id":0})]
                deleted_data  = [x for x in test1.find({'Arrhythmia':{'$regex':"ASYSTOLE"}})]
                delete_id = []
                #test3.insert_many(insert_data)
                for r in deleted_data:
                    delete_id.append(r['_id'])
                list1=[]
                for i in insert_data:
                    if i["patient"] in list1:
                        pass
                    else:
                        list1.append(i["patient"])
                dict1={}
                for j in list1:
                    list2=[]
                    for i in insert_data:
                        if j in i["patient"]:
                            list2.append(i)
                    dict1.update({j:list2})

                for i,j in dict1.items():
                    try:
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"],"version":j[0]["version"]}
                    except Exception as e:
                      print("22:",e)
                      maindict = {"patient":j[0]["patient"],"kit":j[0]["kit"],"position":j[0]["position"]}
                    starttime = []
                    endtime = []
                    hr = []
                    arr = ''
                    pos=[]
                    try:
                        hr.clear()
                        for i in j:
                            if i["Arrhythmia"]=="ASYSTOLE":
                                starttime.append(i["starttime"])
                                endtime.append(i["endtime"])
                                hr.append(int(i["HR"]))
                                arr = "ASYSTOLE"
                                pos.append(i["position"])
                        newst = min(starttime)
                        newend = max(endtime)
                        try:
                            minhr = min(hr)
                            maxhr = max(hr)
                        except:
                            break
                        if int(minhr)>int(maxhr):
                            minhr,maxhr = maxhr,minhr
                        else:
                            pass
                        
                        maindict.update({"Arrhythmia":arr,"MinHR":minhr,"MaxHR":maxhr,"StartTime":newst,"EndTime":newend,"position":pos})
                        if maindict["Arrhythmia"] == "ASYSTOLE":
                            timestamp = newend
                            your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)
                            newtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
                            DT = parser.parse(newtime)
                            now = datetime.datetime.now()
                            datadata = abs(int((now - DT).total_seconds()))
                            print("ASYSTOLE",datadata)
                            if datadata>100:
                                manyuser_data  = [x for x in test1.find({'patient':{'$regex':maindict["patient"]},'Arrhythmia':{'$regex':"ASYSTOLE"}},{"_id":1})]
                                manyuser = []
                                for iii in manyuser_data:
                                    manyuser.append(iii["_id"])
                                test1.delete_many({'_id':{'$in':manyuser}})
                                client = connect_mqtt()
                                publish(client,json.dumps(maindict)) 
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass


            except:
                print("NO data Found")

##    countdown(20)



if __name__ == "__main__":
    print("START")
    while True:
        if mainanalysis():
            pass



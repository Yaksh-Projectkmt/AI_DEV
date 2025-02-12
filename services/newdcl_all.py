import datetime
import pymongo
import time

def sctn():
    myclient = pymongo.MongoClient("mongodb://localhost:27017")
    mydb = myclient["test"]
    mycol = mydb["Test1"]
    
    mycol.delete_many({"$or":[{"Arrhythmia":"ABNORMAL"},{"Arrhythmia":"Artifacts"},{"Arrhythmia":"Non-ECG"},{"Arrhythmia":"Normal"}, {"Arrhythmia":"TC"},{"Arrhythmia":"BR"},{"Arrhythmia":"PVC-Isolated"},{"Arrhythmia":"PVC-Couplet"},{"Arrhythmia":"PVC-Triplet"},{"Arrhythmia":"SA"},{"Arrhythmia":"WIDE-QRS"},{"Arrhythmia":"WIDE-QRS"},{"Arrhythmia":"PAC-Couplet"},{"Arrhythmia":"PAC-Isolated"},{"Arrhythmia":"PAC-Triplet"}]})
    mycols = mydb["Test6"]
    mycols.delete_many({"$or":[{"Arrhythmia":"ABNORMAL"},{"Arrhythmia":"Artifacts"},{"Arrhythmia":"Non-ECG"},{"Arrhythmia":"Normal"}, {"Arrhythmia":"TC"},{"Arrhythmia":"BR"},{"Arrhythmia":"PVC-Isolated"},{"Arrhythmia":"PVC-Couplet"},{"Arrhythmia":"PVC-Triplet"},{"Arrhythmia":"SA"},{"Arrhythmia":"WIDE-QRS"},{"Arrhythmia":"WIDE-QRS"},{"Arrhythmia":"PAC-Couplet"},{"Arrhythmia":"PAC-Isolated"},{"Arrhythmia":"PAC-Triplet"}]})
    return 100

def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        time.sleep(1)
        t -= 1
    return sctn()

t = 10
print("DATA CLEANING START...")
while True:
    t = countdown(t)
    if t is None:
        break

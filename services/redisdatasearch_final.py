import json
import redis
import time
from paho.mqtt import client as mqtt_client
import random,ssl

broker = 'oomcardiodev.projectkmt.com'
port = 8883
client_id = f'AI-PHY{random.randint(100000000, 200000000)}'
username = 'kmt'
password = 'Kmt123'

##broker = 'oomcardiotest.projectkmt.com'
##port = 8883
##client_id = f'{random.randint(0, 999999)}'
##username = 'kmt'
##password = 'dVBbS3NxMtmzD438'

topic_x = "oom/ecg/offlineUnanalyzedData"
def connect_mqtt() -> mqtt_client:

    global client

    def on_connect(client, userdata, flags, rc, protocol):
        if rc == 0:
            print("Connected to Offline MQTT Client")
        else:
            print("Failed to connect, return code %d\n" % rc)

    client = mqtt_client.Client(client_id, protocol=mqtt_client.MQTTv5)
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    client.tls_set_context(context)
    client.enable_shared_subscription = True
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


class RedisQueue:
    def __init__(self, name, redis_conn):
        self.name = name
        self.redis = redis_conn
        self.patient_counter_key = f"{name}_patient_counter"

    def push(self, item):
        # Push data to the list
        self.redis.lpush(self.name, json.dumps(item))

        # Increment patient counter
        item = json.loads(item)
        patient_id = item.get("patient")
        self.redis.hincrby(self.patient_counter_key, patient_id, 1)

    def peek(self):
        item = self.redis.lindex(self.name, -1)
        return json.loads(item) if item is not None else None

    def pop(self):
        item = self.redis.rpop(self.name)
        #item = json.dumps(item)
        if item is not None:
            # Decrement patient counter when popping an item
            patient_id = json.loads(item).get("patient")
            self.redis.hincrby(self.patient_counter_key, patient_id, -1)
        return item if item is not None else None


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

# Connect to Redis
redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)
queue = RedisQueue('devOfflinechannel', redis_conn)
#redis_conn.flushall()
while True:
    try:
        client = connect_mqtt()
        print("MQTT connected,",client)
        break
    except:
        print("MQTT NOT CONNECTED")
        time.sleep(1)

while True:
    try:
        try:
            while True:
                client.loop_start()
                patients = queue.get_all_patient_keys()
                dic = {}
                keys_to_delete = []

                for patientname in patients:        
                    p = queue.get_patient_data_count(patientname)
                    dic[patientname] = p  # Cleaner update syntax

                for ii, jj in dic.items():
                    if jj == -1 or jj == 0:  # If jj is -1 or 0, mark for deletion
                        keys_to_delete.append(ii)
                    else:
                        client.publish(topic_x + "/" + ii, jj, qos=2, retain=True)  # Publish only valid values

                for iii in keys_to_delete:
                    queue.hkeydelete(iii)
                    del dic[iii]

                time.sleep(2)    
        except Exception as e:
            print(e)
            client = connect_mqtt()
    except Exception as e:
        print("Connection Error:", e)


#while True:
#    try:
#        try:
#            while True:
#                client.loop_start()
#                patients = queue.get_all_patient_keys()
###                data = queue.pop()
#                dic = {}
#                keys_to_delete = []
#                for patientname in patients:        
#                    p = queue.get_patient_data_count(patientname)
#                    dic.update({patientname:p})
###                print(dic)
#                for ii,jj in dic.items():
#                    if jj==-1:
#                      keys_to_delete.append(ii)
#                    else:
#                      client.publish(topic_x+"/"+ii,jj,qos=2,retain=True)
#                    if jj==0:
#                        keys_to_delete.append(ii)
#                for iii in keys_to_delete:
#                    queue.hkeydelete(iii)
#                    del dic[iii]
#                time.sleep(2)    
#        except Exception as e:
#            print(e)
#            client = connect_mqtt()
#    except Exception as e:
#        print("Connection Error:",e)

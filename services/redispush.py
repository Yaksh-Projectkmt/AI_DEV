import redis
import time
import json,random
import random
import time
import ssl
import json
from paho.mqtt import client as mqtt_client
import concurrent.futures

broker = 'oomcardiodev.projectkmt.com'
port = 8883
client_id = f'python-mqtt-{random.randint(1000, 2000)}'
username = 'kmt'
password = 'Kmt123'

##broker = 'oomcardiotest.projectkmt.com'
##port = 8883
##client_id = f'{random.randint(0, 999999)}'
##username = 'kmt'
##password = 'dVBbS3NxMtmzD438'

def connect_mqtt() -> mqtt_client:

    global client

    def on_connect(client, userdata, flags, rc, protocol):
        if rc == 0:
            print("Connected")
##            topic = "oom/ecg/rawData"
            topic = "oom/ecg/rawDataOffline"
            client.subscribe(topic)
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
##        self.redis.lpush(self.name, json.dumps(item))
        self.redis.lpush(self.name, item)
        # Increment patient counter
        items = json.loads(item)
        patient_id = items.get("patient")
        self.redis.hincrby(self.patient_counter_key, patient_id, 1)

    def peek(self):
        item = self.redis.lindex(self.name, -1)
        return json.loads(item) if item is not None else None

    def pop(self):
        item = self.redis.rpop(self.name)
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

#class RedisQueue:
#    def __init__(self, name, redis_conn):
#        self.name = name
#        self.redis = redis_conn
#
#    def push(self, item):
#        self.redis.lpush(self.name, json.dumps(item))
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

# Connect to Redis
redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)
queue = RedisQueue('devOfflinechannel', redis_conn)

#redis_conn.flushall()

def on_message(client, userdata, message):
    try:
        payload = message.payload.decode()
        #payload=str(message.payload.decode("utf-8",errors="replace"))
        #payload = json.dumps(payload)
        queue.push(payload)
        print(f'Queue size: {queue.size()}')
    except Exception as e:
        print("Corrupted data received",e)
##    print(f"Received message on topic {message.topic}: {payload}")

    
def subscribe(client):
    client.on_message = on_message  # Set the on_message callback function
    client.loop_forever()   


if __name__ == '__main__':
    while True:
        try:
            client = connect_mqtt()
            subscribe(client)
            break
        except Exception as e:
            print("MQTT NOT CONNETED:",e)
            time.sleep(1)

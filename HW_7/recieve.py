from kafka import KafkaConsumer
import pandas as pd
import time




topicname='csvdata'


consumer = KafkaConsumer(topicname,bootstrap_servers='localhost:9092',auto_offset_reset='latest'  )

for message in consumer:
    mas = message.value.decode('utf-8')
    print(mas)

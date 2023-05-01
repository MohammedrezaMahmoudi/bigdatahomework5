from kafka import KafkaProducer
import pandas as pd
import time




topicname='csvdata'
df = pd.read_csv('f.csv')

producer = KafkaProducer(bootstrap_servers='localhost:9092')

for _ ,row in df.iterrows():
    massage = str(row.values.tolist())
    producer.send(topicname,massage.encode('utf-8'))
    print(massage)
producer.close()

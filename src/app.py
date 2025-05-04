from flask import Flask, Response, render_template
import cv2
import numpy as np
import json
from loguru import logger
from dotenv import load_dotenv
from src.utils.config import load_config
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType
import os
import threading
import time

# Load env và config
load_dotenv()
config = load_config()

app = Flask(__name__)

# Spark config
SPARK_APP_NAME = "ImageStructuredStreaming"
SPARK_MASTER = os.environ.get("SPARK_MASTER", "local[*]")
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_TOPICS = config.kafka.topics["camera_streams"]
KAFKA_GROUP_ID = config.kafka.group_id

if not KAFKA_BOOTSTRAP_SERVERS:
    logger.error("KAFKA_BOOTSTRAP_SERVERS is not set.")
    exit(1)

# Spark Session
spark = SparkSession.builder \
    .appName(SPARK_APP_NAME) \
    .master(SPARK_MASTER) \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# Kafka frame schema
frame_schema = StructType([
    StructField("frame", StringType(), True)
])

# Global biến chứa frame mới nhất (hex string)
latest_frame_hex = None

def process_batch(batch_df, batch_id):
    global latest_frame_hex
    if not batch_df.isEmpty():
        rows = batch_df.select("payload.frame").collect()
        if rows:
            latest_frame_hex = rows[-1]["frame"]  # Lấy frame cuối cùng trong batch

# Đọc từ Kafka và parse JSON
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", KAFKA_TOPICS) \
    .option("startingOffsets", "latest") \
    .load() \
    .selectExpr("CAST(value AS STRING)")

parsed_df = kafka_df.withColumn("payload", from_json(col("value"), frame_schema))

# Ghi stream xử lý batch
query_sink = parsed_df.writeStream \
    .outputMode("append") \
    .foreachBatch(process_batch) \
    .start()

# Flask hiển thị frame đã xử lý
def generate_frames():
    global latest_frame_hex
    while True:
        if latest_frame_hex:
            try:
                frame_bytes = bytes.fromhex(latest_frame_hex)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Chuyển ảnh grayscale thành BGR để encode JPEG đúng
                    gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                    ret, buffer = cv2.imencode('.jpg', gray_bgr)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
        time.sleep(0.05)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Thread để giữ Spark Streaming
    spark_thread = threading.Thread(target=query_sink.awaitTermination)
    spark_thread.daemon = True
    spark_thread.start()

    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    finally:
        spark.stop()
        spark_thread.join()

from flask import Flask, Response, render_template
import cv2
import numpy as np
import json
from kafka import KafkaConsumer
from loguru import logger
from dotenv import load_dotenv
from src.utils.config import load_config
from pyspark.sql import SparkSession
import io
from collections import deque
import os

# Load environment variables and configuration
load_dotenv()
config = load_config()

app = Flask(__name__)

# Kafka Consumer (dùng đúng tên service trong docker-compose)
consumer = KafkaConsumer(
    config.kafka.topics["camera_streams"],
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"),
    group_id=config.kafka.group_id,
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='latest',
    enable_auto_commit=False
)

# Spark Session (kết nối Spark master trong Docker)
spark = SparkSession.builder \
    .appName("ImageProcessingBatchApp") \
    .master(os.getenv("SPARK_MASTER", "spark://spark:7077")) \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()


def _decode_frame(frame_data: str) -> np.ndarray:
    try:
        frame_bytes = bytes.fromhex(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise Exception("Failed to decode image")
        return frame
    except Exception as e:
        logger.error(f"Error decoding frame: {e}")
        return None

def convert_to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

def process_batch_with_spark(batch_frames: list) -> list:
    try:
        rdd = spark.sparkContext.parallelize(batch_frames)
        gray_rdd = rdd.map(convert_to_grayscale)
        return gray_rdd.collect()
    except Exception as e:
        logger.error(f"Spark batch processing error: {e}")
        return []

def generate_frames(batch_size=1):
    buffer = deque(maxlen=batch_size)

    try:
        while True:
            messages = consumer.poll(timeout_ms=1000)
            if not messages:
                logger.warning("No messages received from Kafka.")
                continue

            for _, msgs in messages.items():
                for msg in msgs:
                    try:
                        data = msg.value
                        frame_data = data.get("frame")
                        if not frame_data:
                            continue

                        frame = _decode_frame(frame_data)
                        if frame is not None:
                            buffer.append(frame)

                        if len(buffer) == batch_size:
                            processed_frames = process_batch_with_spark(list(buffer))
                            buffer.clear()

                            for frame_gray in processed_frames:
                                if frame_gray is not None:
                                    ret, buffer_img = cv2.imencode('.jpg', frame_gray)
                                    if ret:
                                        yield (b'--frame\r\n'
                                               b'Content-Type: image/jpeg\r\n\r\n' + buffer_img.tobytes() + b'\r\n')
                                else:
                                    logger.warning("Invalid frame skipped")
                    except Exception as e:
                        logger.error(f"Message handling error: {e}")
                        continue

    except Exception as e:
        logger.error(f"Frame generation error: {e}")
    finally:
        consumer.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video_feed():
    return Response(generate_frames(batch_size=5),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

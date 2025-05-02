from flask import Flask, Response, render_template
import cv2
import numpy as np
import json
from kafka import KafkaConsumer
from loguru import logger
from dotenv import load_dotenv
from src.utils.config import load_config
from pyspark.sql import SparkSession
from pyspark import RDD
import io

# Load environment variables and configuration
load_dotenv()
config = load_config()

app = Flask(__name__)

# Initialize Kafka consumer
consumer = KafkaConsumer(
    config.kafka.topics["camera_streams"],
    bootstrap_servers=config.kafka.bootstrap_servers,
    group_id=config.kafka.group_id,
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='latest'
)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("ImageProcessingApp") \
    .getOrCreate()

def _decode_frame(frame_data: str) -> np.ndarray:
    """Decode frame from hex string."""
    frame_bytes = bytes.fromhex(frame_data)
    nparr = np.frombuffer(frame_bytes, np.uint8)
    
    # Decode the image using OpenCV
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise Exception("Failed to decode the image")
    return frame

def convert_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert frame to grayscale using OpenCV."""
    # Kiểm tra ảnh màu (3 kênh) hoặc ảnh đen trắng (1 kênh)
    if len(frame.shape) == 3 and frame.shape[2] == 3:  # Ảnh màu với 3 kênh (RGB)
        # Chuyển ảnh màu sang ảnh grayscale bằng cách lấy trung bình của 3 kênh RGB
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray_frame
    else:
        # Nếu ảnh đã là ảnh grayscale (1 kênh)
        return frame

def process_image_with_spark(image_data: str) -> np.ndarray:
    """Process the image using Spark (convert to grayscale in this case)."""
    try:
        # Decode the frame
        frame = _decode_frame(image_data)
        
        # Convert the frame to grayscale using Spark
        rdd = spark.sparkContext.parallelize([frame])
        gray_rdd = rdd.map(lambda x: convert_to_grayscale(x))
        
        # Collect the result from the RDD (this could be further optimized for large scale)
        gray_frame = gray_rdd.collect()[0]
        
        return gray_frame
    except Exception as e:
        logger.error(f"Error processing image with Spark: {str(e)}")
        return None

def generate_frames():
    """Generate frames from Kafka stream."""
    try:
        while True:
            # Poll for messages
            messages = consumer.poll(timeout_ms=1000)
            
            for topic_partition, msgs in messages.items():
                for msg in msgs:
                    try:
                        # Decode message
                        data = msg.value
                        frame_data = data["frame"]
                        
                        # Process frame using Spark
                        frame_gray = process_image_with_spark(frame_data)
                        
                        if frame_gray is None:
                            raise Exception("Failed to process image with Spark")
                        
                        # Convert the grayscale frame to JPEG
                        ret, buffer = cv2.imencode('.jpg', frame_gray)
                        if not ret:
                            raise Exception("Failed to encode image")
                        frame_gray = buffer.tobytes()
                        
                        # Stream the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_gray + b'\r\n')
                        
                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}")
                        continue
                        
    except Exception as e:
        logger.error(f"Error in frame generation: {str(e)}")
    finally:
        consumer.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

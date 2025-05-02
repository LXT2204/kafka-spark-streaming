from flask import Flask, Response, render_template
import cv2
import numpy as np
import json
from kafka import KafkaConsumer
from loguru import logger
from dotenv import load_dotenv
from src.utils.config import load_config

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

def _decode_frame(frame_data: str) -> np.ndarray:
    """Decode frame from hex string."""
    frame_bytes = bytes.fromhex(frame_data)
    nparr = np.frombuffer(frame_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
                        
                        # Decode frame
                        frame = _decode_frame(frame_data)
                        
                        # Encode frame to JPEG for streaming
                        ret, buffer = cv2.imencode('.jpg', frame)
                        frame = buffer.tobytes()
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        
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
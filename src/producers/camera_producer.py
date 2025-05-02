import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from kafka import KafkaProducer
from loguru import logger

from src.utils.config import load_config
from src.utils.logger import setup_logger

class VideoProducer:
    """Producer class for streaming video data to Kafka."""
    
    def __init__(self, config):
        """Initialize the video producer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.video = None
        self.producer = None
        self._setup_kafka()
        self._setup_video()
    
    def _setup_kafka(self) -> None:
        """Set up Kafka producer."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.kafka.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all'
            )
            logger.info("Successfully connected to Kafka")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {str(e)}")
            raise
    
    def _setup_video(self) -> None:
        """Set up video capture."""
        try:
            # video_path = Path(self.config.video.path)
            video_path = Path('videos/camera1.mp4')
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            self.video = cv2.VideoCapture(str(video_path))
            if not self.video.isOpened():
                raise RuntimeError(f"Failed to open video file: {video_path}")
            
            # Set video properties
            self.video.set(cv2.CAP_PROP_FPS, self.config.video.fps)
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video.resolution['width'])
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video.resolution['height'])
            
            # Get video info
            total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Successfully opened video: {video_path}")
            logger.info(f"Total frames: {total_frames}, FPS: {fps}, Duration: {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to setup video: {str(e)}")
            raise
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame to hex string.
        
        Args:
            frame: Image frame
            
        Returns:
            Hex encoded string
        """
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes().hex()
    
    def stream(self) -> None:
        """Stream video data to Kafka."""
        logger.info("Starting video stream")
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.video.read()
                if not ret:
                    if self.config.video.loop:
                        # Reset video to beginning
                        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        logger.info("Reached end of video, looping...")
                        continue
                    else:
                        logger.info("Reached end of video")
                        break
                
                # Encode frame
                frame_data = self._encode_frame(frame)
                
                # Prepare message
                message = {
                    "timestamp": time.time(),
                    "frame": frame_data,
                    "frame_number": frame_count,
                    "video_path": str(self.config.video.path)
                }
                
                # Send to Kafka
                self.producer.send(
                    self.config.kafka.topics['camera_streams'],
                    value=message
                )
                self.producer.flush()
                
                # Log progress
                frame_count += 1
                if frame_count % 100 == 0:  # Log every 100 frames
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    logger.info(f"Processed {frame_count} frames at {fps:.2f} FPS")
                
                # Control frame rate
                time.sleep(1.0 / self.config.video.fps)

                # extra sleep, debugging only

        except KeyboardInterrupt:
            logger.info("Stopping video stream")
        except Exception as e:
            logger.error(f"Error in video stream: {str(e)}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.video:
            self.video.release()
        if self.producer:
            self.producer.close()
        logger.info("Cleaned up resources")

def main():
    """Main function to run the video producer."""
    # Load configuration
    config = load_config()
    
    # Setup logger
    setup_logger(config.logging.file, config.logging.level)
    
    # Create and run producer
    producer = VideoProducer(config)
    producer.stream()

if __name__ == "__main__":
    main() 

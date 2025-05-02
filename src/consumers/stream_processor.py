import json
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from kafka import KafkaConsumer
from loguru import logger

from src.models.base_model import BaseModel
from src.utils.config import load_config
from src.utils.logger import setup_logger

class StreamProcessor:
    """Consumer class for processing camera streams from Kafka."""
    
    def __init__(self, config, model: Optional[BaseModel] = None):
        """Initialize the stream processor.
        
        Args:
            config: Configuration object
            model: Optional PyTorch model for processing frames
        """
        self.config = config
        self.model = model
        self.consumer = None
        self._setup_kafka()
    
    def _setup_kafka(self) -> None:
        """Set up Kafka consumer."""
        try:
            self.consumer = KafkaConsumer(
                self.config.kafka.topics["camera_streams"],
                bootstrap_servers=self.config.kafka.bootstrap_servers,
                group_id=self.config.kafka.group_id,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest'
            )
            logger.info("Successfully connected to Kafka")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {str(e)}")
            raise
    
    def _decode_frame(self, frame_data: str) -> np.ndarray:
        """Decode frame from hex string.
        
        Args:
            frame_data: Hex encoded frame data
            
        Returns:
            Decoded image frame
        """
        frame_bytes = bytes.fromhex(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input.
        
        Args:
            frame: Image frame
            
        Returns:
            Preprocessed tensor
        """
        # Resize if needed
        frame = cv2.resize(frame, (224, 224))  # Example size, adjust as needed
        
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        
        # Convert to tensor
        frame = torch.from_numpy(frame).permute(2, 0, 1)
        frame = frame.unsqueeze(0)
        
        return frame
    
    def _process_frame(self, frame: torch.Tensor) -> Dict:
        """Process frame using the model.
        
        Args:
            frame: Preprocessed frame tensor
            
        Returns:
            Processing results
        """
        if self.model is None:
            return {"status": "no_model"}
        
        try:
            with torch.no_grad():
                output = self.model.predict(frame)
            
            # Process model output (customize based on your model)
            return {
                "status": "success",
                "predictions": output.tolist()
            }
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def process_stream(self) -> None:
        """Process camera stream from Kafka."""
        logger.info("Starting stream processing")
        try:
            while True:
                # Poll for messages
                messages = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, msgs in messages.items():
                    for msg in msgs:
                        try:
                            # Decode message
                            data = msg.value
                            frame_data = data["frame"]
                            timestamp = data["timestamp"]
                            frame_number = data.get("frame_number", 0)
                            video_path = data.get("video_path", "unknown")
                            
                            # Decode and preprocess frame
                            frame = self._decode_frame(frame_data)
                            frame_tensor = self._preprocess_frame(frame)
                            
                            # Process frame
                            results = self._process_frame(frame_tensor)

                            # Store the frame to logs
                            frame_path = f"logs/frames/frame_{frame_number}.jpg"
                            cv2.imwrite(frame_path, frame)
                            
                            # Log results
                            logger.info(f"Processed frame {frame_number} from {video_path} at {timestamp}")
                            logger.debug(f"Results: {results}")
                            
                        except Exception as e:
                            logger.error(f"Error processing message: {str(e)}")
                            continue
                
        except KeyboardInterrupt:
            logger.info("Stopping stream processing")
        except Exception as e:
            logger.error(f"Error in stream processing: {str(e)}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.consumer:
            self.consumer.close()
        logger.info("Cleaned up resources")

def main():
    """Main function to run the stream processor."""
    # Load configuration
    config = load_config()
    
    # Setup logger
    setup_logger(config.logging.file, config.logging.level)
    
    # Create and run processor
    processor = StreamProcessor(config)
    processor.process_stream()

if __name__ == "__main__":
    # main() 
    import os
    os.system("python src/app.py")
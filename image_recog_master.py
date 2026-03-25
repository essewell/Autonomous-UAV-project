"""
BeagleY-AI Real-Time Human Detection System
Receives JPEG frames via SPI and performs YOLO detection without intermediate files
"""

import spidev
import time
import struct
from datetime import datetime
import os
import onnxruntime as ort
import cv2
import numpy as np
import argparse

# SPI Configuration
SPI_BUS = 0
SPI_DEVICE = 0
SPI_SPEED = 8000000  # 8 MHz for faster transfer
SPI_MODE = 0
BUFFER_SIZE = 4096

# Protocol commands
CMD_GET_FRAME_SIZE = 0x01
CMD_GET_FRAME_DATA = 0x02
CMD_FRAME_COMPLETE = 0x03


class YOLODetectorONNX:
    def __init__(self, model_path='yolov8n.onnx', input_size=320):
        """Initialize YOLO detector with ONNX Runtime"""
        self.input_size = input_size
        
        print("Initializing ONNX Runtime...")
        
        # Session options for optimization
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create inference session with CPU
        load_start = time.time()
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        print(f"Model loaded in {time.time() - load_start:.2f}s")
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
    
    def preprocess(self, img):
        """Preprocess image for YOLO - accepts numpy array"""
        if img is None:
            raise ValueError("Invalid image data")
        
        self.orig_shape = img.shape[:2]  # (height, width)
        
        # Resize to input size
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and transpose to CHW format
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
        
        # Add batch dimension
        img_batch = np.expand_dims(img_transposed, axis=0)  # (1, 3, H, W)
        
        return img_batch, img
    
    def postprocess(self, outputs, conf_threshold=0.5):
        """Parse YOLO outputs and extract the highest confidence person detection"""
        # YOLOv8 output format: (1, 84, N) where N is number of detections
        # 84 = 4 (bbox) + 80 (COCO classes)
        output = outputs[0]  # Get first output
        
        # Transpose to (N, 84)
        if len(output.shape) == 3:
            output = output[0].T  # Remove batch dim and transpose
        
        # Extract boxes and scores
        boxes = output[:, :4]  # (x_center, y_center, width, height)
        scores = output[:, 4:]  # Class scores
        
        # Get person class (index 0 in COCO)
        person_scores = scores[:, 0]
        
        # Filter by confidence
        mask = person_scores > conf_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = person_scores[mask]
        
        # If no detections, return None
        if len(filtered_scores) == 0:
            return None
        
        # Get the highest confidence detection only
        best_idx = np.argmax(filtered_scores)
        best_box = filtered_boxes[best_idx]
        best_score = filtered_scores[best_idx]
        
        x_center, y_center, width, height = best_box
        
        # Convert to corner coordinates and scale to original image
        x1 = (x_center - width / 2) * self.orig_shape[1] / self.input_size
        y1 = (y_center - height / 2) * self.orig_shape[0] / self.input_size
        x2 = (x_center + width / 2) * self.orig_shape[1] / self.input_size
        y2 = (y_center + height / 2) * self.orig_shape[0] / self.input_size
        
        detection = {
            'confidence': float(best_score),
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'size': (float(x2 - x1), float(y2 - y1))
        }
        
        return detection
    
    def detect(self, img, save_output=False, output_path=None):
        """Run detection on image (numpy array)"""
        # Preprocess
        preprocess_start = time.time()
        input_tensor, orig_img = self.preprocess(img)
        preprocess_time = time.time() - preprocess_start
        
        # Run inference
        inference_start = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        inference_time = time.time() - inference_start
        
        # Postprocess
        postprocess_start = time.time()
        detection = self.postprocess(outputs)
        postprocess_time = time.time() - postprocess_start
        
        total_time = preprocess_time + inference_time + postprocess_time
        
        # Print results
        # print(f"\n{'='*60}")
        # print(f"Timing Breakdown:")
        # print(f"  Preprocess:  {preprocess_time*1000:.1f}ms")
        # print(f"  Inference:   {inference_time*1000:.1f}ms ⭐")
        # print(f"  Postprocess: {postprocess_time*1000:.1f}ms")
        # print(f"  TOTAL:       {total_time*1000:.1f}ms ({total_time:.4f}s)")
        # print(f"{'='*60}")
        
        count = 0 
        # Print detection
        if detection:
            w, h = detection['size']
            count += 1
            # Draw bounding box on the original image
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(orig_img, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (0, 255, 0), 2)
            label = f"Person {detection['confidence']:.2f}"
            cv2.putText(orig_img, label, 
                      (int(x1), int(y1) - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the image with bounding box if output path is provided
            if save_output and output_path:
                cv2.imwrite(output_path, orig_img)
                # print(f"Output saved to {output_path}")
        else:
            print("✗ No person detected")
            # Even if no detection, we still want to save the original image
            if save_output and output_path:
                cv2.imwrite(output_path, orig_img)
                # print(f"Output saved to {output_path}")
                
        return total_time, detection is not None, detection


class RealtimeDetectionSystem:
    def __init__(self, model_path='yolov8n.onnx', save_detections=True):  # Changed default to True
        self.spi = None
        self.frame_count = 0
        self.detection_count = 0
        self.output_dir = "detected_frames"  # Changed folder name
        self.save_detections = save_detections
        
        # Create output directory if saving
        if self.save_detections and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
        
        # Initialize YOLO detector
        self.detector = YOLODetectorONNX(model_path, input_size=320)
    
    def setup_spi(self):
        """Initialize SPI communication"""
        self.spi = spidev.SpiDev()
        self.spi.open(SPI_BUS, SPI_DEVICE)
        self.spi.max_speed_hz = SPI_SPEED
        self.spi.mode = SPI_MODE
        self.spi.bits_per_word = 8
        print(f"SPI initialized: {SPI_SPEED} Hz, Mode {SPI_MODE}")
    
    def spi_transfer(self, send_data):
        """Perform SPI transfer"""
        if len(send_data) < BUFFER_SIZE:
            send_data = send_data + [0] * (BUFFER_SIZE - len(send_data))
        return bytes(self.spi.xfer2(send_data))
    
    def get_frame_size(self):
        """Request frame size from ESP32"""
        time.sleep(0.01)  # Give ESP32 time to prepare response
        send_buffer = [CMD_GET_FRAME_SIZE] + [0] * (BUFFER_SIZE - 1)
        response = self.spi_transfer(send_buffer)
        
        # Parse frame size (first 4 bytes)
        frame_size = struct.unpack('<I', response[0:4])[0]
        return frame_size
    
    def receive_frame(self):
        """Receive complete JPEG frame from ESP32"""
        # Start timing
        frame_start_time = time.time()
        
        # Step 1: Get frame size
        frame_size = self.get_frame_size()
        
        if frame_size == 0 or frame_size > 1000000:  # Sanity check (max 1MB)
            print(f"Invalid frame size: {frame_size}")
            return None

        # Step 2: Receive frame data in chunks
        frame_data = bytearray()
        total_received = 0
        
        while total_received < frame_size:
            # Request next chunk
            send_buffer = [CMD_GET_FRAME_DATA] + [0] * (BUFFER_SIZE - 1)
            response = self.spi_transfer(send_buffer)
            
            # Parse chunk size (first 4 bytes)
            chunk_size = struct.unpack('<I', response[0:4])[0]
            
            if chunk_size == 0 or chunk_size > BUFFER_SIZE - 4:
                print(f"Invalid chunk size: {chunk_size}")
                break
            
            # Extract chunk data
            chunk_data = response[4:4+chunk_size]
            frame_data.extend(chunk_data)
            total_received += chunk_size
            
            # Progress indicator
            #progress = (total_received / frame_size) * 100
            #print(f"  Received: {total_received}/{frame_size} bytes ({progress:.1f}%)", end='\r')
            
            time.sleep(0.001)  # Small delay between chunks
        
        # Calculate total frame receive time
        frame_total_time = time.time() - frame_start_time
        
        #print(f"\n⏱️  Frame receive time: {frame_total_time:.3f}s")
        #print(f"Transfer rate: {(total_received / 1024) / frame_total_time:.1f} KB/s")
        
        # Step 3: Confirm frame received
        send_buffer = [CMD_FRAME_COMPLETE] + [0] * (BUFFER_SIZE - 1)
        self.spi_transfer(send_buffer)
        
        return bytes(frame_data) if total_received == frame_size else None
    
    def verify_jpeg(self, data):
        """Verify if data is a valid JPEG"""
        if len(data) < 2:
            return False
        # JPEG starts with FF D8 and ends with FF D9
        return data[0:2] == b'\xff\xd8' and data[-2:] == b'\xff\xd9'
    
    def bytes_to_image(self, jpeg_bytes):
        """Convert JPEG bytes to OpenCV image (numpy array)"""
        # Decode JPEG bytes to numpy array
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_height, frame_width = img.shape[:2]
        print(f"✓ Frame dimensions: {frame_width} x {frame_height} pixels")
        return img
    
    def run(self, frame_interval=0.01): #originally 0.75
        """
        Main loop to receive frames and perform detection
        
        Args:
            frame_interval: Delay between frames in seconds (TODO: how fast can I go without fucking it up?)
        """
        print("\n=== Real-Time Human Detection System Started ===\n")
        
        try:
            self.setup_spi()
            
            print("Waiting for ESP32 to be ready...")
            time.sleep(2)
            
            # Continuous frame capture
            print(f"Starting continuous capture + detection (interval: {frame_interval}s)")
            print("Press Ctrl+C to stop\n")
            
            while True:
                try:
                    #print(f"\n{'='*60}")
                    #print(f"--- Frame #{self.frame_count + 1} ---")
                    
                    # Receive frame
                    frame_data = self.receive_frame()
                    
                    if frame_data:
                        # Verify JPEG
                        if self.verify_jpeg(frame_data):
                            print("✓ Valid JPEG received")
                            
                            # Convert to OpenCV image
                            img = self.bytes_to_image(frame_data)
                            
                            if img is not None:
                                # Always generate output path for saving
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                output_path = f"{self.output_dir}/frame_{self.frame_count:04d}_{timestamp}.jpg"
                                
                                # Run detection - this will now save ALL frames with bounding boxes
                                det_time, found_person, detection = self.detector.detect(
                                    img, 
                                    save_output=True,  # Force saving for all frames
                                    output_path=output_path
                                )
                                
                                # Final summary printout
                                print(f"\n{'='*60}")
                                print(f"DETECTION SUMMARY - Frame #{self.frame_count + 1}")
                                print(f"{'='*60}")
                                if found_person and detection:
                                    x1, y1, x2, y2 = detection['bbox']
                                    width, height = detection['size']
                                    area = width * height
                                    center_x = (x1 + x2) / 2
                                    center_y = (y1 + y2) / 2
                                    
                                    print(f"✅ PERSON DETECTED")
                                    print(f"  Confidence:        {detection['confidence']:.2%}")
                                    print(f"  Bounding Box:      ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                                    print(f"  Box Center:        ({center_x:.1f}, {center_y:.1f})")
                                    print(f"  Box Dimensions:    {width:.1f} x {height:.1f} pixels")
                                    print(f"  Box Area:          {area:.0f} pixels²")
                                    self.detection_count += 1
                                else:
                                    print(f"❌ NO PERSON DETECTED")
                                print(f"  Frame saved to:    {output_path}")
                                print(f"{'='*60}\n")
                                
                            else:
                                print("✗ Failed to decode image")
                        else:
                            print("✗ Invalid JPEG data")
                    else:
                        print("✗ Frame receive failed")
                    
                    self.frame_count += 1
                    time.sleep(frame_interval)
                    
                except KeyboardInterrupt:
                    print("\n\nStopping capture...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(1)
        
        except Exception as e:
            print(f"Setup error: {e}")
        
        finally:
            if self.spi:
                self.spi.close()
                print("SPI closed")
            
            print(f"Total frames processed: {self.frame_count}")
            print(f"Total detections: {self.detection_count}")
            if self.frame_count > 0:
                print(f"Detection rate: {(self.detection_count/self.frame_count)*100:.1f}%")


def main():
    system = RealtimeDetectionSystem(model_path="yolov8n.onnx")
    system.run(frame_interval=0.01) 
    #TODO: The frames running through opencv are abt 2 frames behind what is happening in real time. 
    # Fix this shit

def get_detection_data(self, img):
    """Get detection data without saving images"""
    det_time, found_person, detection = self.detector.detect(img, save_output=False)
    return {
        'person_detected': found_person,
        'detection': detection,
        'processing_time': det_time
    }


if __name__ == "__main__":
    main()
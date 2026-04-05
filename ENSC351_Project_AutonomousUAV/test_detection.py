import onnxruntime as ort
import cv2
import numpy as np
import time
import sys

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
    
    def preprocess(self, image_path):
        """Preprocess image for YOLO"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
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
    
    def detect(self, image_path, save_output=True):
        """Run detection on image"""
        # Preprocess
        preprocess_start = time.time()
        input_tensor, orig_img = self.preprocess(image_path)
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
        print(f"\n{'='*60}")
        print(f"Timing Breakdown:")
        print(f"  Preprocess:  {preprocess_time*1000:.1f}ms")
        print(f"  Inference:   {inference_time*1000:.1f}ms ⭐")
        print(f"  Postprocess: {postprocess_time*1000:.1f}ms")
        print(f"  TOTAL:       {total_time*1000:.1f}ms ({total_time:.4f}s)")
        print(f"{'='*60}")
        
        # Print detection
        if detection:
            w, h = detection['size']
            print(f"  Person detected:")
            print(f"  Confidence: {detection['confidence']:.2%}")
            print(f"  Bounding box size: {w:.0f}x{h:.0f}px")
            
            # Draw and save
            if save_output:
                x1, y1, x2, y2 = detection['bbox']
                cv2.rectangle(orig_img, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                label = f"Person {detection['confidence']:.2f}"
                cv2.putText(orig_img, label, 
                          (int(x1), int(y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.imwrite('output_detection.jpg', orig_img)
                print("Output saved to output_detection.jpg")
        else:
            print("✗ No person detected")
        
        # Check if meets requirement
        if total_time < 0.5:
            print("✅ SUCCESS: Meets <0.5s requirement!")
        else:
            print(f"⚠ Warning: Slower than target ({total_time:.3f}s vs 0.5s)")
        
        return total_time, detection is not None, detection

# Main script
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_detection.py <image_path>")
        print("Example: python3 test_detection.py Untitled.jpeg")
        sys.exit(1)
    
    # Initialize detector (done once)
    detector = YOLODetectorONNX('yolov8n.onnx', input_size=320)
    
    # Run detection
    det_time, found_person, detection = detector.detect(sys.argv[1])
    
    if found_person:
        print(f"\n✓ RESULT: Person detected with {detection['confidence']:.1%} confidence")
    else:
        print(f"\n✗ RESULT: No person detected")
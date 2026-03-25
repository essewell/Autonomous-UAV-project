"""
Sensor Fusion Controller for Person-Following Drone
Combines Lidar and Camera data to control drone movements
"""

import time
import signal
import sys
from import_lidar import LidarReader  # Import from your lidar module

# Import OpenCV data - we'll need to modify image_recog_master.py to provide data
# For now, we'll assume we can import detection results
from image_recog_master import RealtimeDetectionSystem, YOLODetectorONNX

class DroneController:
    def __init__(self, lidar_consistency_window=3, lidar_jump_threshold=20):
        """
        Initialize the sensor fusion controller
        
        Args:
            lidar_consistency_window: Number of consistent lidar readings needed
            lidar_jump_threshold: Maximum allowed jump in lidar values (mm)
        """
        self.lidar_reader = None
        self.camera_system = None
        self.running = False
        
        # Lidar configuration
        self.lidar_consistency_window = lidar_consistency_window
        self.lidar_jump_threshold = lidar_jump_threshold
        
        # Camera configuration
        self.camera_frame_width = 640  # Assuming standard resolution
        self.camera_frame_height = 480
        self.confidence_threshold = 0.4  # 40% confidence threshold
        
        # Distance control params
        self.target_bbox_area = 3100
        self.area_tolerance = 200

        # Control states
        self.last_detection = None
        self.last_lidar_data = None
        
    def initialize_sensors(self):
        """Initialize both lidar and camera systems"""
        try:
            print("Initializing Lidar...")
            self.lidar_reader = LidarReader(
                consistency_window=self.lidar_consistency_window,
                jump_threshold=self.lidar_jump_threshold
            )
            print("✓ Lidar initialized")
            
            print("Initializing Camera...")
            self.camera_system = RealtimeDetectionSystem(model_path="yolov8n.onnx", save_detections=True)
            self.camera_system.setup_spi()
            print("✓ Camera system initialized")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize sensors: {e}")
            return False
    
    def get_camera_data(self):
        """
        Get the latest camera detection data
        This needs to be modified from the original image_recog_master.py
        to return data instead of just processing frames
        """
        try:
            # Receive frame from ESP32
            frame_data = self.camera_system.receive_frame()
            
            if frame_data and self.camera_system.verify_jpeg(frame_data):
                # Convert to OpenCV image
                img = self.camera_system.bytes_to_image(frame_data)
                
                if img is not None:
                    # Run detection without saving (we just want the data)
                    det_time, found_person, detection = self.camera_system.detector.detect(
                        img, 
                        save_output=False  # Don't save for real-time control
                    )
                    
                    return {
                        'person_detected': found_person,
                        'detection': detection,
                        'timestamp': time.time(),
                        'frame_size': img.shape
                    }
            
            return {
                'person_detected': False,
                'detection': None,
                'timestamp': time.time(),
                'frame_size': None
            }
            
        except Exception as e:
            print(f"Camera error: {e}")
            return {
                'person_detected': False,
                'detection': None,
                'timestamp': time.time(),
                'frame_size': None
            }
    
    def get_lidar_data(self):
        """Get filtered lidar data"""
        try:
            if self.lidar_reader and self.lidar_reader.has_new_data():
                lidar_data = self.lidar_reader.read_data()
                filtered_distances = self.lidar_reader.apply_bounce_filter(lidar_data['distances'])
                
                return {
                    'distances': filtered_distances,  # First 4 sensors filtered
                    'raw_distances': lidar_data['distances'],  # All 5 sensors raw
                    'timestamp': lidar_data['timestamp'],
                    'sequence': lidar_data['sequence']
                }
            return None
        except Exception as e:
            print(f"Lidar error: {e}")
            return None
    
    def calculate_bbox_center(self, detection):
        """Calculate the center of the bounding box"""
        if not detection or 'bbox' not in detection:
            return None
        
        x1, y1, x2, y2 = detection['bbox']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return center_x, center_y
    
    def calculate_bbox_area(self, detection):
        """Calculate the area of the bounding box"""
        if not detection or 'bbox' not in detection:
            return 0
        
        x1, y1, x2, y2 = detection['bbox']
        width = x2 - x1
        height = y2 - y1
        
        return width * height

    def execute_movement(self, movement_command):
        """Execute a movement command (placeholder for actual drone control)"""
        print(f"DRONE COMMAND: {movement_command}")
        
        # TODO: Replace with actual drone control API calls
        # This is where you'd integrate with your drone's SDK
        # For example:
        # - drone.move_right()
        # - drone.rotate_left()
        # - etc.
    
    def fusion_algorithm(self, lidar_data, camera_data):
        """
        Main sensor fusion algorithm based on your pseudocode
        
        Args:
            lidar_data: Filtered lidar distances [sensor1, sensor2, sensor3, sensor4]
            camera_data: Camera detection results
        """
        lidar_distances = lidar_data['distances']
        
        # Priority 1: Obstacle avoidance using lidar
        if lidar_distances[0] < 300:  # Right sensor
            self.execute_movement("move_left")
            return "obstacle_avoidance"
            
        elif lidar_distances[1] < 300:  # Left sensor  
            self.execute_movement("move_right")
            return "obstacle_avoidance"
            
        elif lidar_distances[2] < 300:  # Front sensor
            self.execute_movement("move_backward") 
            return "obstacle_avoidance"
            
        elif lidar_distances[3] < 300:  # Back sensor
            self.execute_movement("move_forward")
            return "obstacle_avoidance"
        
        # Priority 2: Person tracking using camera
        elif camera_data['person_detected'] and camera_data['detection']:
            detection = camera_data['detection']
            
            if detection['confidence'] > self.confidence_threshold:
                bbox_center = self.calculate_bbox_center(detection)
                bbox_area = self.calculate_bbox_area(detection)
                
                if bbox_center:
                    center_x, center_y = bbox_center
                    frame_width = self.camera_frame_width
                    frame_height = self.camera_frame_height
                    
                    # Print detailed tracking info
                    print(f"Tracking - Center: ({center_x:.1f}, {center_y:.1f}), Area: {bbox_area:.0f} px²")
                    
                    # Distance control based on bounding box area
                    if bbox_area > (self.target_bbox_area + self.area_tolerance):
                        # Person is too close - move backward
                        self.execute_movement("move_backward")
                        return "distance_adjust_backward"
                    
                    elif bbox_area < (self.target_bbox_area - self.area_tolerance):
                        # Person is too far - move forward
                        self.execute_movement("move_forward")
                        return "distance_adjust_forward"
                    
                    # Centering control (only if distance is good)
                    else:
                        # Calculate deadzone in center to prevent oscillation
                        deadzone_x = 50  # pixels
                        deadzone_y = 50  # pixels
                        
                        center_min_x = (frame_width / 2) - deadzone_x
                        center_max_x = (frame_width / 2) + deadzone_x
                        center_min_y = (frame_height / 2) - deadzone_y
                        center_max_y = (frame_height / 2) + deadzone_y
                        
                        if center_x > center_max_x:
                            self.execute_movement("rotate_right")
                            return "centering_rotate_right"
                        elif center_x < center_min_x:
                            self.execute_movement("rotate_left") 
                            return "centering_rotate_left"
                        elif center_y > center_max_y:
                            self.execute_movement("move_down")
                            return "centering_move_down"
                        elif center_y < center_min_y:
                            self.execute_movement("move_up")
                            return "centering_move_up"
                        else:
                            # Person is centered and at correct distance
                            self.execute_movement("hover")
                            return "person_centered_perfectly"
                else:
                    print("Error: Could not calculate bounding box center")
                    return "error"
            else:
                print("Person detected but confidence too low")
                self.execute_movement("hover")
                return "low_confidence"
        else:
            print("Camera: no person detected")
            self.execute_movement("hover")
            return "no_person"
    
    def run(self):
        """Main control loop"""
        print("\n=== Starting Sensor Fusion Controller ===")
        print("Press Ctrl+C to stop\n")
        
        if not self.initialize_sensors():
            print("Failed to initialize sensors. Exiting.")
            return
        
        self.running = True
        
        try:
            while self.running:
                # Get data from both sensors
                lidar_data = self.get_lidar_data()
                camera_data = self.get_camera_data()
                
                # Print sensor status
                if lidar_data:
                    print(f"Lidar: {[f'{d:.1f}' for d in lidar_data['distances']]} mm")
                    self.last_lidar_data = lidar_data
                
                if camera_data['person_detected'] and camera_data['detection']:
                    bbox_center = self.calculate_bbox_center(camera_data['detection'])
                    if bbox_center:
                        center_x, center_y = bbox_center
                        print(f"Camera: Person detected (conf: {camera_data['detection']['confidence']:.2f}, center: ({center_x:.1f}, {center_y:.1f}))")
                
                # Only run fusion algorithm if we have both sensor data
                if lidar_data and camera_data:
                    result = self.fusion_algorithm(lidar_data, camera_data)
                    print(f"Fusion result: {result}")
                else:
                    if not lidar_data:
                        print("Waiting for lidar data...")
                    if not camera_data['frame_size']:
                        print("Waiting for camera data...")
                
                print("-" * 50)
                time.sleep(0.1)  # Control loop frequency
                
        except KeyboardInterrupt:
            print("\nStopping controller...")
        except Exception as e:
            print(f"Controller error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.lidar_reader:
            self.lidar_reader.close()
        print("Controller shutdown complete")

# Global reference for signal handler
controller = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\nReceived interrupt signal, cleaning up...')
    if controller:
        controller.running = False
    sys.exit(0)

def main():
    global controller
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run the controller
    controller = DroneController(
        lidar_consistency_window=3,
        lidar_jump_threshold=20
    )
    
    controller.run()

if __name__ == "__main__":
    main()
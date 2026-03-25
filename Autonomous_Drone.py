from pymavlink import mavutil
import time
import signal
import sys

from import_lidar import LidarReader
from image_recog_master import RealTimeDetectionSystem, YOLODetectorONNX

#everything to do with the drone control and MAVLink
class Drone:
    def __init__(self, connection_string = '/dev/ttyAMA0', baudrate=57600):
        "Initilaizing MAVLink connection"
        self.connection_string = connection_string
        self.baud = baudrate
        self.master = None

        #speed parameters
        self.speed_xy = 0.25
        self.speed_z = 0.25
        self.speed_yaw = 0.25

        #initial velocities
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_z = 0.0
        self.yaw_rate = 0.0

    def connect(self):
        "Connect to flight controller"

        try:
            print(f"connect drone {self.connection_string}")
            self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baud)
    
            print("waiting for heartbeat")
            self.master.wait_heartbeat()
            print(f"connected to drone (system {self.master.target_system} component {self.master.target_component})")
            return True
        except Exception as e:
            print(f"Failed to connect to drone: {e}")
            return False
    
    def set_mode(self, mode_name = "GUIDED"):
        """Set the drone mode"""
        try: 
            mode_mapping = self.master.mode_mapping()
            if mode_name not in mode_mapping:
                print (f"Error: Unknown mode {mode_name} unavailable")
                return False
            
            mode_id = mode_mapping[mode_name]
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id)
            print(f"Mode changed to {mode_name}")
            return True
        except Exception as e:
            print(f"Failed to set mode: {e}")
            return False
        
    def arm(self):
        """Arm the drone"""
        try:
            self.master.mav.command_long_send(
                self.master.target_system, 
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0
            )
            self.master.motors_armed_wait()
            print("Drone armed")
            return True
        except Exception as e:
            print(f"Failed to arm drone: {e}")
            return False
        
    def disarm(self):
        """Disarm the drone"""
        try:
            self.master.mav.command_long_send(
                self.master.target_system, 
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            self.master.motors_disarmed_wait()
            print("Drone disarmed")
            return True
        except Exception as e:
            print(f"Failed to disarm drone: {e}")
            return False
        
    def set_velocity(self, velocity_x, velocity_y, velocity_z, yaw_rate=0.0):
        """Set the drone velocity"""
        try:
            self.master.mav.set_position_target_local_ned_send(
                int(time.time() * 1000),
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000111111000111, #mask
                0,0,0, #ignoring x, y, z
                velocity_x, velocity_y, velocity_z, #not ignoring velocities
                0,0,0, # ignoring accelerations
                0,0,yaw_rate #not ignoring yaw rate
            )
            return True
        except Exception as e:
            print(f"Failed to set velocity: {e}")
            return False
        
    def stop(self):
        """Stop movement and hover in place"""
        self.set_velocity(0.0, 0.0, 0.0, 0.0)

    def forward(self):
        """Drone move forward"""
        self.velocity_x = self.speed_xy
        self.set_velocity(self.velocity_x, self.velocity_y, self.velocity_z, self.yaw_rate)

    def backward(self):
        """Drone move backward"""
        self.velocity_x = -self.speed_xy
        self.set_velocity(self.velocity_x, self.velocity_y, self.velocity_z, self.yaw_rate)

    def left(self):
        """Drone move left"""
        self.velocity_y = -self.speed_xy
        self.set_velocity(self.velocity_x, self.velocity_y, self.velocity_z, self.yaw_rate)
        
    def right(self):
        """Drone move right"""
        self.velocity_y = self.speed_xy
        self.set_velocity(self.velocity_x, self.velocity_y, self.velocity_z, self.yaw_rate)

    def up(self):
        """Drone move up"""
        self.velocity_z = -self.speed_z
        self.set_velocity(self.velocity_x, self.velocity_y, self.velocity_z, self.yaw_rate)

    def down(self):
        """Drone move down"""
        self.velocity_z = self.speed_z
        self.set_velocity(self.velocity_x, self.velocity_y, self.velocity_z, self.yaw_rate)

    def yaw_left(self):
        """Drone rotate left"""
        self.yaw_rate = self.speed_yaw
        self.set_velocity(self.velocity_x, self.velocity_y, self.velocity_z, self.yaw_rate)

    def yaw_right(self):
        """Drone rotate right"""
        self.yaw_rate = -self.speed_yaw
        self.set_velocity(self.velocity_x, self.velocity_y, self.velocity_z, self.yaw_rate)
    
    def hover(self):
        """Drone hovering"""
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_z = 0.0
        self.yaw_rate = 0.0
        self.set_velocity(self.velocity_x, self.velocity_y, self.velocity_z, self.yaw_rate) 

class DroneController:
    def __init__(self, lidar_consistency_window=3, lidar_jump_threshold=20):
        
        self.drone = Drone()
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
        self.obstacle_threshold = 300 

        # Control states
        self.last_detection = None
        self.last_lidar_data = None
        
    def initialize_sensors(self):
        """Initialize both lidar and camera systems"""
        try:
            print("Connecting to drone...")
            if not self.drone.connect():
                return False
            time.sleep(1)

            print("Setting drone to GUIDED mode...")
            if not self.drone.set_mode("GUIDED"):
                return False
            time.sleep(1)

            print("Initializing Lidar...")
            self.lidar_reader = LidarReader(
                consistency_window=self.lidar_consistency_window,
                jump_threshold=self.lidar_jump_threshold
            )
            print("✓ Lidar initialized")
            
            print("Initializing Camera...")
            self.camera_system = RealtimeDetectionSystem(model_path="yolov8n.onnx", save_detections= False)
            self.camera_system.setup_spi()
            print("✓ Camera system initialized")
            
            print("Arming motor...")
            if not self.drone.arm():
                return False
            time.sleep(1)

            print("All systems iare ready")
            return True
        
        except Exception as e:
            print(f"Failed to initialize sensors: {e}")
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
    
    def fusion_algorithm(self, lidar_data, camera_data):
        """
        Main sensor fusion algorithm based on your pseudocode
        
        Args:
            lidar_data: Filtered lidar distances [sensor1, sensor2, sensor3, sensor4]
            camera_data: Camera detection results
        """
        lidar_distances = lidar_data['distances']
        
        # Priority 1: Obstacle avoidance using lidar
        if lidar_distances[0] < self.obstacle_threshold:  # Right sensor
            self.drone.left()
            return "obstacle_avoidance_left"
            
        elif lidar_distances[1] < self.obstacle_threshold:  # Left sensor  
            self.drone.right()
            return "obstacle_avoidance_right"
            
        elif lidar_distances[2] < self.obstacle_threshold:  # Front sensor
            self.drone.backward()
            return "obstacle_avoidance_backward"
            
        elif lidar_distances[3] < self.obstacle_threshold:  # Back sensor
            self.drone.forward()
            return "obstacle_avoidance_forward"
        
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
                        self.drone.backward()
                        return "distance_adjust_backward"
                    
                    elif bbox_area < (self.target_bbox_area - self.area_tolerance):
                        # Person is too far - move forward
                        self.drone.forward()
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
                            self.drone.yaw_right()
                            return "centering_rotate_right"
                        elif center_x < center_min_x:
                            self.drone.yaw_left() 
                            return "centering_rotate_left"
                        elif center_y > center_max_y:
                            self.drone.down()
                            return "centering_move_down"
                        elif center_y < center_min_y:
                            self.drone.up()
                            return "centering_move_up"
                        else:
                            # Person is centered and at correct distance
                            self.drone.hover()
                            return "person_centered_perfectly"
                else:
                    print("Error: Could not calculate bounding box center")
                    self.drone.hover()
                    return "error"
            else:
                print(f"Person detected but confidence too low ({detection['confidence']:.2f})")
                self.drone.hover()
                return "low_confidence"
        else:
            print("Camera: no person detected")
            self.drone.hover()
            return "no_person"
    
    def run(self):
        """Main control loop"""
        print("\n" + "="*60)
        print("\n=== Starting Sensor Fusion Controller ===")
        print("="*60)
        print("Press Ctrl+C to stop\n")
        
        if not self.initialize_sensors():
            print("Failed to initialize sensors. Exiting.")
            return
        
        self.running = True
        control_rate_hz = 10  # Control loop frequency
        last_control_time = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                if current_time - last_control_time >= 1.0 / control_rate_hz:
                # Get data from both sensors
                    lidar_data = self.get_lidar_data()
                    camera_data = self.get_camera_data()
                
                    # Print sensor status
                    if lidar_data:
                        distances = [f'{d:.1f}' for d in lidar_data['distances']]
                        print(f"Lidar: [{', '.join(distances)}] mm")
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
                        self.drone.hover()
                
                    print("-" * 60)
                    last_control_time = current_time
                    
                time.sleep(0.1)  # Control loop frequency
                
        except KeyboardInterrupt:
            print("\nStopping controller...")
        except Exception as e:
            print(f"Controller error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources...")
        self.running = False

        if self.drone and self.drone.master:
            print("stop drone")
            self.drone.stop()
            time.sleep(1)

            print("Landing drone")
            self.drone.set_mode("LAND")
            time.sleep(5)

            print("Disarming drone")
            self.drone.disarm()
            time.sleep(1)

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
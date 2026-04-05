import struct
import time
import signal
import sys
import os
import subprocess
from multiprocessing import shared_memory
from collections import deque

class LidarReader:
    def __init__(self, shm_name="lidar_data", consistency_window=3, jump_threshold=20):
        """
        Initialize connection to shared memory
        
        Args:
            shm_name: Name of the shared memory segment
            consistency_window: Number of consecutive similar readings needed to accept a change (default: 3)
            jump_threshold: Maximum allowed jump in values (default: 20)
        """
        self.shm_name = shm_name
        self.shm = None
        self.last_sequence = 0
        self.cpp_process = None
        
        # Filter parameters
        self.consistency_window = consistency_window
        self.jump_threshold = jump_threshold
        
        # Store last confirmed values and recent history for each of the 4 sensors
        self.confirmed_values = [None] * 4
        self.recent_history = [deque(maxlen=consistency_window) for _ in range(4)]
        
        self.connect()
    
    def connect(self):
        """Connect to the shared memory segment"""
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            print(f"Connected to shared memory: {self.shm_name}")
        except FileNotFoundError:
            print(f"Shared memory '{self.shm_name}' not found. Is the C++ program running?")
            raise
    
    def read_data(self):
        """
        Read lidar data from shared memory
        Returns: dict with 'distances', 'timestamp', 'sequence'
        """
        data = struct.unpack('=5HQI', self.shm.buf[:22])
        return {
            'distances': list(data[0:5]),
            'timestamp': data[5],
            'sequence': data[6]
        }
    
    def apply_bounce_filter(self, new_values):
        """
        Apply bounce filter to remove single spike events
        Only accepts changes when values are consistent across multiple readings
        
        Args:
            new_values: List of 5 values from sensors (only first 4 are filtered)
        
        Returns:
            List of 4 filtered values
        """
        filtered = []
        
        for i in range(4):  # Only process first 4 sensors
            new_val = new_values[i]
            
            # If this is the first reading, accept it immediately
            if self.confirmed_values[i] is None:
                self.confirmed_values[i] = new_val
                self.recent_history[i].append(new_val)
                filtered.append(new_val)
                continue
            
            current_confirmed = self.confirmed_values[i]
            jump = abs(new_val - current_confirmed)
            
            # If the new value is close to current confirmed value, accept it
            if jump <= self.jump_threshold:
                self.confirmed_values[i] = new_val
                self.recent_history[i].clear()  # Reset history
                self.recent_history[i].append(new_val)
                filtered.append(new_val)
            else:
                # Large jump detected - need consistency check
                self.recent_history[i].append(new_val)
                
                # Check if we have enough consistent readings
                if len(self.recent_history[i]) >= self.consistency_window:
                    # Check if all values in recent history are similar to each other
                    hist_vals = list(self.recent_history[i])
                    max_hist = max(hist_vals)
                    min_hist = min(hist_vals)
                    
                    # If recent readings are consistent with each other (within threshold)
                    if (max_hist - min_hist) <= self.jump_threshold:
                        # Accept the change - this is a real transition, not a bounce
                        avg_new = sum(hist_vals) / len(hist_vals)
                        self.confirmed_values[i] = avg_new
                        #print(f"  [Sensor {i}] Confirmed transition: {current_confirmed:.1f} → {avg_new:.1f} mm")
                        filtered.append(avg_new)
                    else:
                        # Values are inconsistent - likely noise, keep current
                        #print(f"  [Sensor {i}] Bounce rejected: inconsistent readings")
                        filtered.append(current_confirmed)
                else:
                    # Not enough data yet, keep current value
                    #print(f"  [Sensor {i}] Large jump detected, waiting for consistency ({len(self.recent_history[i])}/{self.consistency_window})")
                    filtered.append(current_confirmed)
            
        return filtered
    
    def has_new_data(self):
        """Check if there's new data available"""
        current_data = self.read_data()
        if current_data['sequence'] > self.last_sequence:
            self.last_sequence = current_data['sequence']
            return True
        return False
    
    def wait_for_new_data(self, timeout=1.0):
        """
        Wait for new data to arrive
        Returns: data dict if new data arrived, None if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.has_new_data():
                return self.read_data()
            time.sleep(0.001)
        return None
    
    def kill_cpp_process(self):
        """Kill the C++ process if it's still running"""
        try:
            subprocess.run(['sudo', 'killall', 'continuousMultipleSensors'], 
                          check=False, 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL)
            time.sleep(0.5)
        except Exception as e:
            print(f"Note: Could not terminate C++ program: {e}")
    
    def close(self):
        """Close the shared memory connection and kill C++ process"""
        if self.shm:
            self.shm.close()
            print("Closed shared memory connection")
        self.kill_cpp_process()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Global reference for signal handler
lidar_reader = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\nReceived interrupt signal, cleaning up...')
    if lidar_reader:
        lidar_reader.close()
    os._exit(0)

# Example usage
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # consistency_window: how many consistent readings needed to accept a large change
        # jump_threshold: what counts as a "large" jump (in mm)
        with LidarReader(consistency_window=3, jump_threshold=20) as lidar:
            lidar_reader = lidar
            print("Reading lidar data with bounce filter... Press Ctrl+C to stop")
            print(f"Consistency window: {lidar.consistency_window}, Jump threshold: {lidar.jump_threshold} mm\n")
            
            while True:
                new_data = lidar.wait_for_new_data(timeout=0.1)
                
                if new_data:
                    # Apply filter to first 4 sensors
                    filtered_distances = lidar.apply_bounce_filter(new_data['distances'])
                    
                    # Print both raw and filtered values
                    raw_4 = new_data['distances'][:4]
                    #print(f"Raw:      {raw_4} mm")
                    print(f"Filtered: {[f'{v:.1f}' for v in filtered_distances]} mm")
                    #print()
                else:
                    print("No new data received")
                
                time.sleep(0.035)
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if lidar_reader:
            lidar_reader.close()
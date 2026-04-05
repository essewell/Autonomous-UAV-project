#!/usr/bin/env python3
"""
BeagleY-AI SPI Master - Camera Data Receiver
Receives JPEG camera frames from ESP32-S3 via SPI
"""

import spidev
import time
import struct
from datetime import datetime
import os

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

class CameraReceiver:
    def __init__(self):
        self.spi = None
        self.frame_count = 0
        self.output_dir = "captured_frames"
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
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
        # originally 0.01
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
        
        print(f"Frame size: {frame_size} bytes")
        
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
            progress = (total_received / frame_size) * 100
            print(f"  Received: {total_received}/{frame_size} bytes ({progress:.1f}%)", end='\r')
            
            time.sleep(0.001)  # Small delay between chunks
            # originally 0.001
        
        # Calculate total frame receive time
        frame_total_time = time.time() - frame_start_time
        
        print(f"\n⏱️  Frame receive time: {frame_total_time:.3f}s")
        print(f"Transfer rate: {(total_received / 1024) / frame_total_time:.1f} KB/s")
        
        # Step 3: Confirm frame received
        send_buffer = [CMD_FRAME_COMPLETE] + [0] * (BUFFER_SIZE - 1)
        self.spi_transfer(send_buffer)
        
        return bytes(frame_data) if total_received == frame_size else None
    
    def save_frame(self, frame_data):
        """Save frame to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.output_dir}/frame_{self.frame_count:04d}_{timestamp}.jpg"
        
        try:
            with open(filename, 'wb') as f:
                f.write(frame_data)
            print(f"✓ Saved: {filename} ({len(frame_data)} bytes)")
            self.frame_count += 1
            return True
        except Exception as e:
            print(f"✗ Error saving frame: {e}")
            return False
    
    def verify_jpeg(self, data):
        """Verify if data is a valid JPEG"""
        if len(data) < 2:
            return False
        # JPEG starts with FF D8 and ends with FF D9
        return data[0:2] == b'\xff\xd8' and data[-2:] == b'\xff\xd9'
    
    def run(self, continuous=True, frame_interval=1.0): #frame_interval=1.0
        """
        Main loop to receive frames
        
        Args:
            continuous: If True, continuously receive frames
            frame_interval: Delay between frames in seconds
        """
        print("\n=== Camera Frame Receiver Started ===\n")
        
        try:
            self.setup_spi()
            
            print("Waiting for ESP32 to be ready...")
            time.sleep(2)
            
            if continuous:
                print(f"Starting continuous capture (interval: {frame_interval}s)")
                print("Press Ctrl+C to stop\n")
                
                while True:
                    try:
                        print(f"\n--- Frame #{self.frame_count + 1} ---")
                        frame_data = self.receive_frame()
                        
                        if frame_data:
                            # Verify JPEG
                            if self.verify_jpeg(frame_data):
                                print("✓ Valid JPEG received")
                                self.save_frame(frame_data)
                            else:
                                print("✗ Invalid JPEG data")
                        else:
                            print("✗ Frame receive failed")
                        
                        time.sleep(frame_interval)
                        
                    except KeyboardInterrupt:
                        print("\n\nStopping capture...")
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        time.sleep(1)
            else:
                # Single frame capture
                print("Capturing single frame...")
                frame_data = self.receive_frame()
                
                if frame_data:
                    if self.verify_jpeg(frame_data):
                        print("✓ Valid JPEG received")
                        self.save_frame(frame_data)
                    else:
                        print("✗ Invalid JPEG data")
                else:
                    print("✗ Frame receive failed")
        
        except Exception as e:
            print(f"Setup error: {e}")
        
        finally:
            if self.spi:
                self.spi.close()
                print("\nSPI closed")
            
            print(f"\nTotal frames captured: {self.frame_count}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ESP32-CAM Frame Receiver')
    parser.add_argument('-s', '--single', action='store_true', 
                       help='Capture single frame and exit')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                       help='Interval between frames in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    receiver = CameraReceiver()
    receiver.run(continuous=not args.single, frame_interval=args.interval)

if __name__ == "__main__":
    main()
#include "VL53L0X.hpp"

#include <chrono>
#include <csignal>
#include <exception>
#include <iomanip>
#include <iostream>
#include <unistd.h>

#include <stdio.h>
// headers for shared memory
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
// headers for sensor reset
#include <fstream>
#include <string>

/* TODO
 - When running this in the background (`sudo ./continuousMultipleSensors &`) there are still processes running in the background after stopping (CTRL-C)
 - Solved this by running `sudo killall continuousMultipleSensors` in the python script, but sudo may not work or be possible when running only on beagle. 
*/

// Define the shared memory data structure
struct __attribute__((packed)) lidar_data_t {
    uint16_t distances[5];  // Array for up to 5 sensors
    uint64_t timestamp;     // Timestamp in ns
    uint32_t sequence;      // Sequence number to detect new data
};

// function so i2c stops shitting itself after each reset (questionable whether all of the reset shit is necessary)
void reset_sensors(const uint16_t* pins, int sensor_count) {
    std::cout << "Resetting " << sensor_count << " sensors..." << std::endl;
    
    // Step 1: Power off ALL sensors (pull XSHUT low)
    for (int i = 0; i < sensor_count; i++) {
        std::string gpio_num = std::to_string(pins[i]);
        
        // Export GPIO if not already exported
        std::ofstream export_file("/sys/class/gpio/export");
        if (export_file.is_open()) {
            export_file << gpio_num;
            export_file.close();
        }
        
        usleep(50000); // 50ms delay for export to complete
        
        // Set as output
        std::string direction_path = "/sys/class/gpio/gpio" + gpio_num + "/direction";
        std::ofstream direction_file(direction_path);
        if (direction_file.is_open()) {
            direction_file << "out";
            direction_file.close();
        } else {
            std::cerr << "Warning: Could not set direction for GPIO " << gpio_num << std::endl;
        }
        
        usleep(10000); // 10ms
        
        // Pull low to power off
        std::string value_path = "/sys/class/gpio/gpio" + gpio_num + "/value";
        std::ofstream value_file(value_path);
        if (value_file.is_open()) {
            value_file << "0";
            value_file.close();
            std::cout << "  Sensor " << i << " (GPIO " << gpio_num << ") powered off" << std::endl;
        } else {
            std::cerr << "Error: Could not power off sensor " << i << " (GPIO " << gpio_num << ")" << std::endl;
        }
    }
    
    // Step 2: Wait for all sensors to fully power down
    std::cout << "Waiting for sensors to power down..." << std::endl;
    usleep(200000); // 200ms delay - increased from 50ms
    
    // Step 3: Power sensors back on ONE AT A TIME
    // This prevents I2C address conflicts during initialization
    for (int i = 0; i < sensor_count; i++) {
        std::string gpio_num = std::to_string(pins[i]);
        std::string value_path = "/sys/class/gpio/gpio" + gpio_num + "/value";
        
        std::ofstream value_file(value_path);
        if (value_file.is_open()) {
            value_file << "1";
            value_file.close();
            std::cout << "  Sensor " << i << " (GPIO " << gpio_num << ") powered on" << std::endl;
        }
        
        // Small delay between powering on each sensor
        usleep(50000); // 50ms between each sensor
    }
    
    // Step 4: Wait for all sensors to boot up
    std::cout << "Waiting for sensors to boot up..." << std::endl;
    usleep(200000); // 200ms delay - increased from 50ms
    
    std::cout << "Sensor reset complete\n" << std::endl;
}

// Global pointers for cleanup in signal handler
lidar_data_t *global_data = nullptr;
int global_shm_fd = -1;

// SIGINT (CTRL-C) exit flag and signal handler
volatile sig_atomic_t exitFlag = 0;
void sigintHandler(int) {
	exitFlag = 1;

	// Cleanup shared memory 
	if (global_data != nullptr && global_data != MAP_FAILED) {
		munmap(global_data, sizeof(lidar_data_t));
		global_data = nullptr;
	}
	if (global_shm_fd != -1) {
		close(global_shm_fd);
		global_shm_fd = -1;
	}
	shm_unlink("/lidar_data");
	
	std::cout << "\nCleaned up resources. Exiting..." << std::endl;
	std::exit(0);
}

int main() {
	std::cout << "Size of lidar_data_t: " << sizeof(lidar_data_t) << " bytes" << std::endl;

	// Configuration constants
	// Number of sensors. If changed, make sure to adjust pins and addresses accordingly (ie to match size).
	const int SENSOR_COUNT = 4;
	// GPIO pins to use for sensors' XSHUT. As exported by WiringPi.
	const uint16_t pins[SENSOR_COUNT] = { 343, 345, 346, 335}; // addresses for GPIO 5, 6, 13, 
	/* GPIO pins reference:
	    GPIO 5  -> pin 343
	    GPIO 6  -> pin 345
	    GPIO 13 -> pin 346
	    GPIO 16 -> pin 335
	    ~~GPIO 26 -> pin 437~~
	*/
	// Reset all sensors before starting
	reset_sensors(pins, SENSOR_COUNT);
	// Sensors' addresses that will be set and used. These have to be unique.
	const uint8_t addresses[SENSOR_COUNT] = {
		VL53L0X_ADDRESS_DEFAULT + 2,
		VL53L0X_ADDRESS_DEFAULT + 4,
		VL53L0X_ADDRESS_DEFAULT + 6,
		VL53L0X_ADDRESS_DEFAULT + 10,
		//VL53L0X_ADDRESS_DEFAULT + 12,
		//VL53L0X_ADDRESS_DEFAULT + 14
	};

	// Register SIGINT handler
	signal(SIGINT, sigintHandler);

	// === SETUP SHARED MEMORY (DO ONCE AT STARTUP) ===
	close(global_shm_fd);
	shm_unlink("/lidar_data"); // unlink any previous instance
	
	global_shm_fd = shm_open("/lidar_data", O_CREAT | O_RDWR, 0666);
	if (global_shm_fd == -1) {
		std::cerr << "Error opening shared memory" << std::endl;
		return 1;
	}
	
	if (ftruncate(global_shm_fd, sizeof(lidar_data_t)) == -1) {
		std::cerr << "Error setting shared memory size" << std::endl;
		close(global_shm_fd);
		return 1;
	}

	global_data = (lidar_data_t*)mmap(NULL, sizeof(lidar_data_t), PROT_READ | PROT_WRITE, MAP_SHARED, global_shm_fd, 0);
    if (global_data == MAP_FAILED) {
        std::cerr << "Error mapping shared memory" << std::endl;
        close(global_shm_fd);
        return 1;
    }
	
	lidar_data_t *global_data = (lidar_data_t*)mmap(NULL, sizeof(lidar_data_t), 
	                                          PROT_READ | PROT_WRITE, 
	                                          MAP_SHARED, global_shm_fd, 0);
	if (global_data == MAP_FAILED) {
		std::cerr << "Error mapping shared memory" << std::endl;
		close(global_shm_fd);
		return 1;
	}
	
	// Initialize shared memory data
	for (int j = 0; j < 5; j++) {
		global_data->distances[j] = 0;
	}
	global_data->timestamp = 0;
	global_data->sequence = 0;

	// Create sensor objects' array
	VL53L0X* sensors[SENSOR_COUNT];

	// Create sensors (and ensure GPIO pin mode)
	for (int i = 0; !exitFlag && i < SENSOR_COUNT; ++i) {
		sensors[i] = new VL53L0X(pins[i]);
		sensors[i]->powerOff();
	}
	// Just a check for an early CTRL-C
	if (exitFlag) {
		munmap(global_data, sizeof(lidar_data_t));
		close(global_shm_fd);
		shm_unlink("/lidar_data");
		return 0;
	}

	// For each sensor: create object, init the sensor (ensures power on), set timeout and address
	// Note: don't power off - it will reset the address to default!
	for (int i = 0; !exitFlag && i < SENSOR_COUNT; ++i) {
		try {
			// Initialize...
			sensors[i]->initialize();
			// ...set measurement timeout...
			sensors[i]->setTimeout(200);
			// ...set the lowest possible timing budget (high speed mode)...
			sensors[i]->setMeasurementTimingBudget(20000);
			// ...and set I2C address...
			sensors[i]->setAddress(addresses[i]);
			// ...also, notify user.
			std::cout << "Sensor " << i << " initialized, real time budget: " << sensors[i]->getMeasurementTimingBudget() << std::endl;
		} catch (const std::exception & error) {
			std::cerr << "Error initializing sensor " << i << " with reason:" << std::endl << error.what() << std::endl;
			return 1;
		}
	}

	// Start continuous back-to-back measurement
	for (int i = 0; !exitFlag && i < SENSOR_COUNT; ++i) {
		try {
			sensors[i]->startContinuous();
		} catch (const std::exception & error) {
			std::cerr << "Error starting continuous read mode for sensor " << i << " with reason:" << std::endl << error.what() << std::endl;
			return 2;
		}
	}

	// Durations in nanoseconds
	uint64_t totalDuration = 0;
	uint64_t maxDuration = 0;
	uint64_t minDuration = 1000*1000*1000;
	// Initialize reference time measurement
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

	// We need that variable after the for loop
	int j = 0;
	// Also, set fill and width options for cout so that measurements are aligned
	std::cout << std::setw(4) << std::setfill('0');

	// Take the measurements!
	for (; !exitFlag && j < 100000; ++j) {
		//std::cout << "\rReading" << j << " | ";
		for (int i = 0; !exitFlag && i < SENSOR_COUNT; ++i) {
			uint16_t distance;
			try {
				// Read the range. Note that it's a blocking call
				distance = sensors[i]->readRangeContinuousMillimeters();
				// Scaling to read distance in cm
				distance = distance / 13; 
			} catch (const std::exception & error) {
				std::cerr << std::endl << "Error geating measurement from sensor " << i << " with reason:" << std::endl << error.what() << std::endl;
				// You may want to bail out here, depending on your application - error means issues on I2C bus read/write.
				// return 3;
				distance = 8096;
			}

			// Check for timeout
			if (sensors[i]->timeoutOccurred()) {
				// std::cout << "tout | ";
			} else {
				// Display the reading
				// std::cout << distance << " | ";
			}

			// === WRITE TO SHARED MEMORY ===
			global_data->distances[i] = distance;
			//data->distances = distance;
			global_data->timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
							std::chrono::steady_clock::now().time_since_epoch()
							).count();
			global_data->sequence++;

			// Calculate duration of current iteration
			std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
			uint64_t duration = (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)).count();
			// Save current time as reference for next iteration
			t1 = t2;
			// Add total measurements duration
			totalDuration += duration;
		}
		// std::cout << std::endl << std::flush;

		// Calculate duration of current iteration
		std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
		uint64_t duration = (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)).count();
		// Save current time as reference for next iteration
		t1 = t2;
		// Add total measurements duration
		totalDuration += duration;
		// Skip comparing first measurement against max and min as it's not a full iteration
		if (j == 0) {
			continue;
		}
		// Check and save max and min iteration duration
		if (duration > maxDuration) {
			maxDuration = duration;
		}
		if (duration < minDuration) {
			minDuration = duration;
		}
	}

	// Print measurement duration statistics
	std::cout << "\nMax duration: " << maxDuration << "ns" << std::endl;
	std::cout << "Min duration: " << minDuration << "ns" << std::endl;
	std::cout << "Avg duration: " << totalDuration/(j+1) << "ns" << std::endl;
	std::cout << "Avg frequency: " << 1000000000/(totalDuration/(j+1)) << "Hz" << std::endl;

	// Clean-up: delete objects, set GPIO/XSHUT pins to low.
	// Cleanup: delete objects, stop continuous readings, and set GPIO pins to low.
	for (int i = 0; i < SENSOR_COUNT; ++i) {
		if (sensors[i]) {
			sensors[i]->stopContinuous();
			delete sensors[i];
			sensors[i] = nullptr;
		}
	}

	// === CLEANUP SHARED MEMORY ===
	munmap(global_data, sizeof(lidar_data_t));
	close(global_shm_fd);
	shm_unlink("/lidar_data");

	return 0;
}

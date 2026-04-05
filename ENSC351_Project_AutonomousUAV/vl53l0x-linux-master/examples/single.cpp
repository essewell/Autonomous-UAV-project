/** This example shows how to get single-shot range measurements from the VL53L0X.
 * The sensor can optionally be configured with different ranging profiles,
 * as described in the VL53L0X API user manual, to get better performance for a certain application.
 * This code is based on "Single.ino" example from vl53l0x-arduino library,
 * which in turn is based on the four "SingleRanging" examples in the VL53L0X API.
 * The range readings are in units of mm.
 */


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


// Define the shared memory data structure
struct __attribute__((packed)) lidar_data_t {
    uint16_t distances[5];  // Array for up to 5 sensors
    uint64_t timestamp;     // Timestamp in nanoseconds
    uint32_t sequence;      // Sequence number to detect new data
};

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
	exit(0);
}



// Uncomment to enable long range measurements
// #define LONG_RANGE
// Uncomment ONE to enable high speed or high accuracy measurements
// #define HIGH_SPEED
// #define HIGH_ACCURACY
#ifdef HIGH_SPEED
	#ifdef HIGH_ACCURACY
		#error HIGH_SPEED and HIGH_ACCURACY cannot be both enabled at once!
	#endif
#endif

int main() {
	// Register SIGINT handler
	signal(SIGINT, sigintHandler);

	std::cout << "Size of lidar_data_t: " << sizeof(lidar_data_t) << " bytes" << std::endl;


	// === SETUP SHARED MEMORY (DO ONCE AT STARTUP) ===
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

	// Create the sensor with default values
	VL53L0X sensor;
	try {
		// Initialize the sensor
		sensor.initialize();
		// Set measurement timeout value
		sensor.setTimeout(200);
	} catch (const std::exception & error) {
		std::cerr << "Error initializing sensor with reason:" << std::endl << error.what() << std::endl;
		return 1;
	}

	#ifdef LONG_RANGE
		try {
			// Lower the return signal rate limit (default is 0.25 MCPS)
			sensor.setSignalRateLimit(0.1);
			// Increase laser pulse periods (defaults are 14 and 10 PCLKs)
			sensor.setVcselPulsePeriod(VcselPeriodPreRange, 18);
			sensor.setVcselPulsePeriod(VcselPeriodFinalRange, 14);
		} catch (const std::exception & error) {
			std::cerr << "Error enabling long range mode with reason:" << std::endl << error.what() << std::endl;
			return 2;
		}
	#endif

	#if defined HIGH_SPEED
		try {
			// Reduce timing budget to 20 ms (default is about 33 ms)
			sensor.setMeasurementTimingBudget(20000);
		} catch (const std::exception & error) {
			std::cerr << "Error enabling high speed mode with reason:" << std::endl << error.what() << std::endl;
			return 3;
		}
	#elif defined HIGH_ACCURACY
		try {
			// Increase timing budget to 200 ms
			sensor.setMeasurementTimingBudget(200000);
		} catch (const std::exception & error) {
			std::cerr << "Error enabling high accuracy mode with reason:" << std::endl << error.what() << std::endl;
			return 3;
		}
	#endif

	// Highly unprobable but check SIGINT exit flag
	if (exitFlag) {
		munmap(global_data, sizeof(lidar_data_t));
		close(global_shm_fd);
		shm_unlink("/lidar_data");
		return 0;
	}

	// Durations in nanoseconds
	uint64_t totalDuration = 0;
	uint64_t maxDuration = 0;
	uint64_t minDuration = 1000*1000*1000;
	// Initialize reference time measurement
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

	// We need iterator value after the loop
	int i = 0;
	int sensor_id = 0; // NEW SHIT TO BE UPDATED
	// Also, set width/fill for cout stream so that measurements are aligned
	std::cout << "\rReading" << std::setw(4) << std::setfill('0');
	// Take the measurements!
	for (; !exitFlag && i < 100000; ++i) {
		uint16_t distance;
		try {
			// Read the range. Note that it's a blocking call
			distance = sensor.readRangeSingleMillimeters();
		} catch (const std::exception & error) {
			std::cerr << "Error geating measurement with reason:" << std::endl << error.what() << std::endl;
			// You may want to bail out here, depending on your application - error means issues on I2C bus read/write.
			// return 3;
			distance = 8096;
		}

		// Check IO timeout and print range information
		if (sensor.timeoutOccurred()) {
			std::cout << "\rReading" << i << " | timeout!" << std::endl;
		} else {
			// std::cout << "\rReading" << i << " | " << distance << std::endl;
		}
		std::cout << std::flush;

		// === WRITE TO SHARED MEMORY ===
		global_data->distances[sensor_id] = distance;
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
		// Skip comparing first measurement against max and min as it's not a full iteration
		if (i == 0) {
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

	// Print duration data
	std::cout << "\nMax duration: " << maxDuration << "ns" << std::endl;
	std::cout << "Min duration: " << minDuration << "ns" << std::endl;
	std::cout << "Avg duration: " << totalDuration/(i+1) << "ns" << std::endl;
	std::cout << "Avg frequency: " << 1000000000/(totalDuration/(i+1)) << "Hz" << std::endl;

	// === CLEANUP SHARED MEMORY ===
	munmap(global_data, sizeof(lidar_data_t));
	close(global_shm_fd);
	shm_unlink("/lidar_data");

	return 0;
}

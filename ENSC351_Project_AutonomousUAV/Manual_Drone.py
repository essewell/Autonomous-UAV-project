from pymavlink import mavutil
import time
import sys
import select
import tty
import termios

master = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)

print("wating for heartbeat")
master.wait_heartbeat()
print("heartbeat from system (system %u component %u)" % (master.target_system, master.target_component))

master.mav.param_request_read_send(
    master.target_system,
    master.target_component,
    b'ARMING_CHECK',
    -1
)

while True:
    msg = master.recv_match(blocking=True)
    if msg:
        print(msg)

#seeting uyp the mode to be GUIDED
def set_mode(master, mode_name = "GUIDED"):
    mode_mapping = master.mode_mapping()
    #maybe add an error catching statement in case it can't map to the drone
    mode_ID = mode_mapping[mode_name] 

    master.mav.set_mode_send(master.target_system, mavutil.mavlink.mav_mode_flag_custom_mode_enabled, mode_ID)

set_mode(master, "GUIDED")
time.sleep(5)

#arming the drone
def arm(master):
    master.mav.command_long_send(master.target_system, master.target_component, mavutil.mavlink.mav_cmd_component_arm_disarm, 0, 1, 0, 0, 0, 0, 0, 0)

    master.motors_armed_confirmation()
    #maybe add a confirmation printed message

#disarming the drone
def disarm(master):
    master.mav.command_long_send(master.target_system, master.target_component, mavutil.mavlink.mav_cmd_component_arm_disarm, 0, 0, 0, 0, 0, 0, 0, 0)

    master.motors_disarmed_confirmation()
    #add a confirmation message for disarming

#velocity
def drone_velocity(master, velocity_x, velocity_y, velocity_z, yaw_rate = 0.0):
    master.mav.set_position_target_local_ned_send(
        int(time.time() * 1000),
        master.target_system,
        master.target_component,
        mavutil.mavlink.mav_frame_body_ned,
        0b0000111111000111, #mask
        0,0,0, #ignorimg x, y, z
        velocity_x, velocity_y, velocity_z, #not ignoring velocities
        0,0,0, # ignoring accelerations
        0,0,yaw_rate #not ignoring yaw rate
    )

def keyboard(timeout = 0.01):
    drone, _, _ = select.select([sys.stdin], [], [], timeout)
    if drone:
        return sys.stdin.read(1)
    return None

def keyboard_input(master):
    #W/S = forward/backward (velocity_x)
    #A/D = left/right (velocity_y)
    #R/F = up/down  (velocity_z)
    #Q/E = yaw left/yaw right (yaw_rate)
    #X = go to land mode (exiting)
    #Z = disarm drone (exiting)
    #SPACE = hovering (no velocity)

    velocity_x = velocity_y = velocity_z = yaw_rate = 0.0
    speed_xy = 0.5
    speed_z = 0.5
    speed_yaw = 0.25

    #getting the keyboard pressed button (raw mode)
    settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())
    try:
        while True:
            key = keyboard()

            if key: 
                key = key.lower()

                if key == 'w':
                    velocity_x = speed_xy
                elif key == 's':
                    velocity_x = -speed_xy
                elif key == 'd':
                    velocity_y = speed_xy
                elif key == 'a':
                    velocity_y = -speed_xy
                elif key == 'r':
                    velocity_z = -speed_z
                elif key == 'f':
                    velocity_z = speed_z
                elif key == 'q':
                    yaw_rate = speed_yaw
                elif key == 'e':
                    yaw_rate = -speed_yaw
                elif key == 'x':
                    print ("switch to land mode and exiting")
                    set_mode(master, "LAND")
                    break
                elif key == 'z':
                    print("disarming drone and exiting")
                    disarm(master)
                    break
                elif key == ' ':
                    print("drone hovering")
                    velocity_x = velocity_y = velocity_z = yaw_rate = 0.0
            else:
                velocity_x = velocity_y = velocity_z = yaw_rate = 0.0
        
        drone_velocity(master, velocity_x, velocity_y, velocity_z, yaw_rate)
        time.sleep(0.1)


    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        drone_velocity(master, 0,0,0,0)

def main():
    master = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)
    set_mode(master, "GUIDED")
    time.sleep(0.5)

    arm(master)
    time.sleep(0.5)

    keyboard_input(master)

if __name__ == "__main__":
    main()
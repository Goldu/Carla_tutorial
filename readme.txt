import carla
import random
import time
import math

# Function to calculate the distance between two points
def distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

# Function to calculate the angle between two vectors
def angle_between_vectors(vec1, vec2):
    return math.acos(vec1.dot(vec2) / (vec1.length() * vec2.length()))

# Function to calculate the curvature of a road segment between two waypoints
def curvature(waypoint1, waypoint2, waypoint3):
    vector1 = waypoint2.transform.location - waypoint1.transform.location
    vector2 = waypoint3.transform.location - waypoint2.transform.location
    cross_product = vector1.cross(vector2)
    curvature = 2.0 * cross_product.z / max(0.1, vector1.length() * vector2.length() * (vector1 + vector2).length())
    return curvature

# Connect to the Carla server
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

# Retrieve the world object
world = client.get_world()

# Get the blueprint of a vehicle
vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))

# Spawn the vehicle at a random location
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Get the list of waypoints on a specific road
waypoints = world.get_map().get_waypoints(spawn_point.location)

# Set the target speed and acceleration of the vehicle
target_speed = 10.0  # m/s
target_acceleration = 1.0  # m/s^2

# Set the maximum distance to the next waypoint at which to accelerate
max_distance = 10.0  # meters

# Set the maximum angle to the next waypoint at which to start turning
max_angle = math.radians(10.0)  # radians

# Set the steering gain for the curvature
steering_gain = 0.01

# Track the waypoints with the vehicle
for i, waypoint in enumerate(waypoints):
    # Get the current location and velocity of the vehicle
    location = vehicle.get_location()
    velocity = vehicle.get_velocity()

    # Calculate the distance to the next waypoint
    distance_to_waypoint = distance(location, waypoint.transform.location)

    # Calculate the required acceleration to reach the target speed
    required_acceleration = (target_speed - velocity.x) / max(0.1, distance_to_waypoint)

    # Apply the required acceleration up to the target acceleration
    acceleration = min(target_acceleration, max(-target_acceleration, required_acceleration))
    throttle = acceleration / target_acceleration
    brake = 0.0

    # Calculate the vector pointing from the current location of the vehicle to the next waypoint
    waypoint_vector = waypoint.transform.location - location

    # Calculate the orientation of the vehicle and the heading vector
    vehicle_orientation = vehicle.get_transform().rotation.get_forward_vector()
    vehicle_heading = carla.Vector3D(vehicle_orientation.x, vehicle_orientation.y, 0.0)

    # Calculate the angle between the heading vector and the waypoint vector
    angle_to_waypoint = angle_between_vectors(waypoint_vector, vehicle_heading)

    # Calculate the curvature of the road segment between the current waypoint and the next two waypoints
    if i < len(waypoints) - 2:
        curv = curvature(waypoint, waypoints[i+1], waypoints

######Steering Control
import carla
import random
import time
import math

# Function to calculate the distance between two points
def distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

# Function to calculate the angle between two vectors
def angle_between_vectors(vec1, vec2):
    return math.acos(vec1.dot(vec2) / (vec1.length() * vec2.length()))

# Connect to the Carla server
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

# Retrieve the world object
world = client.get_world()

# Get the blueprint of a vehicle
vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))

# Spawn the vehicle at a random location
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Get the list of waypoints on a specific road
waypoints = world.get_map().get_waypoints(spawn_point.location)

# Set the target speed and acceleration of the vehicle
target_speed = 10.0  # m/s
target_acceleration = 1.0  # m/s^2

# Set the maximum distance to the next waypoint at which to accelerate
max_distance = 10.0  # meters

# Set the maximum angle to the next waypoint at which to start turning
max_angle = math.radians(10.0)  # radians

# Track the waypoints with the vehicle
for i, waypoint in enumerate(waypoints):
    # Get the current location and velocity of the vehicle
    location = vehicle.get_location()
    velocity = vehicle.get_velocity()

    # Calculate the distance to the next waypoint
    distance_to_waypoint = distance(location, waypoint.transform.location)

    # Calculate the required acceleration to reach the target speed
    required_acceleration = (target_speed - velocity.x) / max(0.1, distance_to_waypoint)

    # Apply the required acceleration up to the target acceleration
    acceleration = min(target_acceleration, max(-target_acceleration, required_acceleration))
    throttle = acceleration / target_acceleration
    brake = 0.0

    # Calculate the vector pointing from the current location of the vehicle to the next waypoint
    waypoint_vector = waypoint.transform.location - location

    # Calculate the orientation of the vehicle and the heading vector
    vehicle_orientation = vehicle.get_transform().rotation.get_forward_vector()
    vehicle_heading = carla.Vector3D(vehicle_orientation.x, vehicle_orientation.y, 0.0)

    # Calculate the angle between the heading vector and the waypoint vector
    angle_to_waypoint = angle_between_vectors(waypoint_vector, vehicle_heading)

    # Set the steering angle of the vehicle based on the angle to the next waypoint
    if angle_to_waypoint > max_angle:
        sign = (waypoint_vector.y > vehicle_heading.y) - (waypoint_vector.y < vehicle_heading.y)
        steer = sign * 0.3  # adjust the steering angle here as needed
    else:
        steer = 0.0

    # Brake if the distance


#------------------------------------------------
import carla
import random
import time

# Function to calculate the distance between two points
def distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

# Connect to the Carla server
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

# Retrieve the world object
world = client.get_world()

# Get the blueprint of a vehicle
vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))

# Spawn the vehicle at a random location
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Get the list of waypoints on a specific road
waypoints = world.get_map().get_waypoints(spawn_point.location)

# Set the target speed and acceleration of the vehicle
target_speed = 10.0  # m/s
target_acceleration = 1.0  # m/s^2

# Set the maximum distance to the next waypoint at which to accelerate
max_distance = 10.0  # meters

# Track the waypoints with the vehicle
for i, waypoint in enumerate(waypoints):
    # Get the current location and velocity of the vehicle
    location = vehicle.get_location()
    velocity = vehicle.get_velocity()

    # Calculate the distance to the next waypoint
    distance_to_waypoint = distance(location, waypoint.transform.location)

    # Calculate the required acceleration to reach the target speed
    required_acceleration = (target_speed - velocity.x) / max(0.1, distance_to_waypoint)

    # Apply the required acceleration up to the target acceleration
    acceleration = min(target_acceleration, max(-target_acceleration, required_acceleration))
    throttle = acceleration / target_acceleration
    brake = 0.0
    steer = 0.0

    # Brake if the distance to the next waypoint is less than the maximum distance
    if distance_to_waypoint < max_distance:
        brake = abs(acceleration)

    # Set the control inputs of the vehicle
    control = carla.VehicleControl(throttle, brake, steer)
    vehicle.apply_control(control)

    # Set the target location of the vehicle to the current waypoint
    vehicle.set_transform(waypoint.transform)

    # Wait for a short time to simulate the movement of the vehicle
    time.sleep(0.1)

    # Print the current waypoint index and the current location and velocity of the vehicle
    print("Waypoint {}/{} - Location: ({}, {}, {}), Velocity: ({}, {}, {})".format(i+1, len(waypoints),
        location.x, location.y, location.z, velocity.x, velocity.y, velocity.z))

# Destroy the vehicle
vehicle.destroy()

#------------------------------------------------
import carla
import numpy as np

# Connect to the CARLA simulator
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Load the map
world = client.load_world('my_custom_map')

# Get the list of waypoints
map = world.get_map()
waypoints = map.generate_waypoints(distance_between=2.0)

# Create a vehicle actor
spawn_point = waypoints[0].transform
vehicle_bp = world.get_blueprint_library().find('vehicle.audi.a2')
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Define the desired waypoints
desired_waypoints = waypoints[50:100]

# Define the controller parameters
target_speed = 20.0  # m/s
Kp = 1.0
Kd = 0.1

# Main loop
while True:
    # Get the current vehicle position
    vehicle_loc = vehicle.get_location()
    vehicle_yaw = np.deg2rad(vehicle.get_transform().rotation.yaw)

    # Find the closest waypoint to the vehicle
    min_dist = float('inf')
    closest_wp = None
    for wp in desired_waypoints:
        dist = np.linalg.norm(np.array([wp.transform.location.x, wp.transform.location.y]) - np.array([vehicle_loc.x, vehicle_loc.y]))
        if dist < min_dist:
            min_dist = dist
            closest_wp = wp

    # Compute the cross track error
    dx = closest_wp.transform.location.x - vehicle_loc.x
    dy = closest_wp.transform.location.y - vehicle_loc.y
    cte = np.sin(np.arctan2(dy, dx) - vehicle_yaw) * np.linalg.norm([dx, dy])

    # Compute the desired heading
    desired_heading = np.arctan2(dy, dx)

    # Compute the steering angle and throttle
    steering_error = desired_heading - vehicle_yaw
    steering_angle = np.clip(Kp * cte + Kd * steering_error, -1.0, 1.0)
    throttle = np.clip(target_speed - vehicle.get_velocity().length(), 0.0, 1.0)

    # Apply the control signals
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steering_angle))

    # Check if we have reached the final waypoint
    if closest_wp == desired_waypoints[-1]:
        break

# Destroy the actor and cleanup
vehicle.destroy()



#------------------------------------------
import numpy as np
import carla
import math

# Set up CARLA client and connect to server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the world object and the vehicle actor
world = client.get_world()
vehicle = world.get_actors().filter('vehicle.*')[0]

# Get the list of waypoints
waypoints = world.get_map().get_waypoints(vehicle.get_location(), 10.0)

# Initialize variables for distance and time
distance = 0.0
time = 0.0

# Loop through the waypoints and calculate distance and time
for i in range(len(waypoints) - 1):
    start = waypoints[i].transform.location
    end = waypoints[i + 1].transform.location
    segment_distance = math.sqrt((end.x - start.x) ** 2 + (end.y - start.y) ** 2)
    segment_time = (end.timestamp - start.timestamp) * 1e-9
    distance += segment_distance
    time += segment_time

# Calculate the current velocity in meters per second (m/s)
velocity = distance / time

# Print the current velocity
print(f"Current velocity: {velocity:.2f} m/s")







import carla
import numpy as np
import time

class PIDController():
    def __init__(self, Kp, Ki, Kd, Ts):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ts = Ts
        self.integral = 0.0
        self.derivative = 0.0
        self.prev_error = 0.0

    def step(self, error):
        self.integral += error * self.Ts
        self.derivative = (error - self.prev_error) / self.Ts
        self.prev_error = error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * self.derivative
        return output

class VehiclePIDController():
    def __init__(self, vehicle, target_speed):
        self.vehicle = vehicle
        self.target_speed = target_speed
        self.max_throttle = 0.75
        self.max_brake = 0.3
        self.max_steering = 0.8
        self.throttle_controller = PIDController(Kp=1.0, Ki=0.1, Kd=0.1, Ts=0.1)
        self.steering_controller = PIDController(Kp=0.5, Ki=0.1, Kd=0.1, Ts=0.1)

    def get_speed(self):
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        return speed

    def control(self, waypoint):
        target_speed = self.target_speed
        current_speed = self.get_speed()

        throttle = 0.0
        brake = 0.0
        steering = 0.0

        if target_speed > 0.1:
            speed_error = target_speed - current_speed
            throttle = self.throttle_controller.step(speed_error)
            if throttle > self.max_throttle:
                throttle = self.max_throttle
            elif throttle < 0:
                throttle = 0.0
                brake = self.max_brake
            else:
                brake = 0.0

            location = self.vehicle.get_location()
            transform = self.vehicle.get_transform()
            orientation = transform.rotation
            waypoint_location = waypoint.transform.location
            dx = waypoint_location.x - location.x
            dy = waypoint_location.y - location.y

            # convert to radians
            yaw = np.deg2rad(orientation.yaw)
            # calculate the distance and angle between the vehicle and the waypoint
            distance = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)

            # calculate the steering command
            heading_error = yaw - angle
            steering = self.steering_controller.step(heading_error)
            steering = np.clip(steering, -self.max_steering, self.max_steering)

        control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steering)

        return control





Carla_testing
Carla_Sensor
Carla depth sensor
Testing
https://carla.readthedocs.io/en/latest/tuto_G_instance_segmentation_sensor/#:~:text=Instance%20segmentation%20is%20a%20new,class%2C%20like%20for%20example%20vehicles.

PID Controller  Implementation in Python
https://github.com/yan99033/real-time-carla-kalman-filter/blob/main/kalman_filter.py
https://roboticsknowledgebase.com/wiki/simulation/Spawning-and-Controlling-Vehicles-in-CARLA/
https://medium.com/@jaimin-k/longitudinal-lateral-control-for-autonomous-vehicles-carla-simulator-c045918816bd

CArla speed update 
https://github.com/zubair-irshad/classical_controllers_for_self_driving_car
https://zubairirshad.com/portfolio/vehicle-control-for-autonomous-driving-carla-simulator-unreal-engine/


PID Controller
https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/agents/navigation/controller.py
Sensor Reference
https://carla.readthedocs.io/en/latest/ref_sensors/

Papers
https://github.com/Mina2kamel/Reinforcement-learning-for-self-driving-cars-in-CARLA-simulator/blob/main/Steering_Model_for_Autonomous_Driving_System_using_Deep_Reinforcement_Learning.pdf




import carla
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
spawn_point = carla.Transform(carla.Location(x=10, y=10, z=2), carla.Rotation(yaw=0))
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.error = 0
        self.integral = 0
        self.derivative = 0
        self.previous_error = 0
        self.output = 0

    def update(self, measurement):
        self.error = self.setpoint - measurement
        self.integral += self.error
        self.derivative = self.error - self.previous_error
        self.output = self.Kp * self.error + self.Ki * self.integral + self.Kd * self.derivative
        self.previous_error = self.error
        return self.output

pid = PIDController(0.1, 0.001, 0.01, 10.0)
while True:
    # Get the current speed of the vehicle
    current_speed = vehicle.get_velocity()
    current_speed = 3.6 * math.sqrt(current_speed.x**2 + current_speed.y**2 + current_speed.z**2)

    # Update the PID controller and get the control values
    control_signal = pid.update(current_speed)
    throttle = max(min(control_signal, 1.0), 0.0)
    brake = max(min(-control_signal, 1.0), 0.0)
    steering = 0.0

    # Set the control values for the vehicle
    vehicle_control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steering)
    vehicle.apply_control(vehicle_control)

    time.sleep(0.01)

The equation current_speed = math.sqrt(current_speed.x**2 + current_speed.y**2 + current_speed.z**2) calculates the magnitude of the velocity vector of the vehicle in meters per second.

However, in many real-world scenarios, such as in traffic regulations or vehicle control systems, speed is commonly expressed in kilometers per hour (km/h). Therefore, to convert the speed from meters per second to km/h, we can multiply the magnitude of the velocity vector by a conversion factor of 3.6.

So, the equation current_speed = 3.6 * math.sqrt(current_speed.x**2 + current_speed.y**2 + current_speed.z**2) calculates the speed of the vehicle in km/h, which is a more commonly used unit of measurement for vehicle speed.



Machine Learning Coding
import carla
import numpy as np
import tensorflow as tf

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,), name='input'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='tanh', name='output')
])

# Load the trained weights
model.load_weights('model_weights.h5')

# Connect to the Carla simulator
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

# Get the world object
world = client.get_world()

# Get the blueprint of the vehicle you want to control
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

# Spawn the vehicle in the world
spawn_point = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation(yaw=0.0))
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Get the control of the vehicle
vehicle_control = carla.VehicleControl()

# Define the function to get the state of the vehicle
def get_vehicle_state():
    # Get the location and rotation of the vehicle
    location = vehicle.get_location()
    rotation = vehicle.get_transform().rotation
    # Get the velocity of the vehicle
    velocity = vehicle.get_velocity()
    speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
    # Get the distance to the center of the road
    start = location + carla.Location(x=2.0, y=0.0)
    end = location + carla.Location(x=-2.0, y=0.0)
    trace = world.trace_route(start, end)
    if trace:
        center = trace[0][0].location
        distance = location.distance(center)
    else:
        distance = 0.0
    # Return the state as a NumPy array
    return np.array([speed, distance])

# Run the simulation
while True:
    # Get the current state of the vehicle
    state = get_vehicle_state()
    # Use the neural network to predict the control action
    output = model.predict(np.expand_dims(state, axis=0))[0]
    # Set the control action for the vehicle
    vehicle_control.throttle = output[0]
    vehicle_control.steer = output[1]
    vehicle_control.brake = 0.0
    # Apply the control action to the vehicle
    vehicle.apply_control(vehicle_control)
#-----------------------------------------------------------------------------------------
import carla
import numpy as np
import matplotlib.pyplot as plt

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

# Get the world object
world = client.get_world()

# Set the spectator view to the vehicle
spectator = world.get_spectator()
spectator.set_transform(carla.Transform(carla.Location(x=-50, y=-50, z=50), carla.Rotation(yaw=180)))

# Spawn a vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('model3')[0]
spawn_point = carla.Transform(carla.Location(x=40, y=0, z=2), carla.Rotation(yaw=180))
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Define the waypoints
waypoints = []
waypoints.append(carla.Location(x=40, y=0, z=2))
waypoints.append(carla.Location(x=60, y=0, z=2))
waypoints.append(carla.Location(x=80, y=20, z=2))
waypoints.append(carla.Location(x=100, y=40, z=2))
waypoints.append(carla.Location(x=120, y=60, z=2))

# Implement the Stanley controller
k = 0.1
L = 2.5
dt = 0.1
t = 0
x = []
y = []
heading = []
steering = []
throttle = []
brake = []

while True:
    # Get the current location and orientation of the vehicle
    vehicle_location = vehicle.get_location()
    vehicle_orientation = vehicle.get_transform().rotation.yaw

    # Calculate the cross-track error (CTE) and heading error
    closest_waypoint = waypoints[0]
    closest_distance = np.linalg.norm(np.array([closest_waypoint.x, closest_waypoint.y]) - np.array([vehicle_location.x, vehicle_location.y]))
    for waypoint in waypoints:
        distance = np.linalg.norm(np.array([waypoint.x, waypoint.y]) - np.array([vehicle_location.x, vehicle_location.y]))
        if distance < closest_distance:
            closest_distance = distance
            closest_waypoint = waypoint
    delta = closest_waypoint.y - vehicle_location.y
    heading_error = np.deg2rad(closest_waypoint.z) - np.deg2rad(vehicle_orientation)

    # Calculate the steering angle using the Stanley controller
    v = np.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2)
    delta = np.arctan2(k * delta, v) + np.arctan2(k * L * heading_error, v)

    # Set the throttle, brake, and steering based on the output of the Stanley controller
    throttle_value = 0.5
    brake_value = 0.0
    steering_value = delta / np.deg2rad(70)
    vehicle.apply_control(carla.VehicleControl(throttle=throttle_value, brake=brake_value, steer=steering_value))

    # Update the simulation
    world.tick()

    # Save the data for plotting
    x.append(vehicle_location.x)
    y.append(vehicle_location.y)
    heading.append(vehicle_orientation)
    steering.append(delta)
    throttle.append(throttle_value)
    brake.append(brake_value)

    # Check if the vehicle has reached the last waypoint
    if closest_distance < 1:
        break

    t += dt

# Plot the data
fig, (ax1, ax2) = plt.subplots(2, 1


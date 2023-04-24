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


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from mpl_toolkits.mplot3d import Axes3D

EPOCHS = 50
ORIENTATION = [0.0, np.deg2rad(30), 0.0]

# === Rocket State ===
class RocketState:
    def __init__(self, position, velocity, orientation, angular_velocity, winglet_positions):
        self.position = position
        self.velocity = velocity
        self.orientation = orientation
        self.angular_velocity = angular_velocity
        self.winglet_positions = winglet_positions

# === Rocket Simulator ===
class RocketSimulator:
    def __init__(self):
        self.mass = 0.5  # kg
        self.length = 0.5  # m
        self.drag_coefficient = 0.3
        self.reference_area = 0.01  # m^2
        self.air_density = 1.225  # kg/m^3
        self.winglet_effectiveness = 0.1
        self.dt = 0.01
        self.gravity = np.array([0, 0, -9.81])
        self.thrust_curve = self._generate_thrust_curve()
        self.moment_of_inertia = np.array([0.002, 0.002, 0.004])  # kg·m²
        self.time = 0.0

    def _generate_thrust_curve(self):
        times = np.array([0, 0.1, 0.5, 1.0, 1.5, 2.0])
        thrusts = np.array([0, 50, 40, 30, 10, 0])
        return lambda t: np.interp(t, times, thrusts)

    def reset(self):
        self.time = 0.0
        return RocketState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.array(ORIENTATION), # orientation=np.zeros(3),
            angular_velocity=np.zeros(3),
            winglet_positions=np.zeros(4)
        )

    def _rotate_vector(self, vector, angles):
        roll, pitch, yaw = angles
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return R @ vector

    def step(self, state, winglet_commands):
        new_winglet_positions = 0.9 * state.winglet_positions + 0.1 * winglet_commands

        local_thrust = np.array([0, 0, 1]) * self.thrust_curve(self.time)
        thrust = self._rotate_vector(local_thrust, state.orientation)
        drag = -0.5 * self.air_density * self.drag_coefficient * self.reference_area * np.linalg.norm(state.velocity) * state.velocity

        torques = np.zeros(3)
        torques[0] = self.winglet_effectiveness * (new_winglet_positions[0] - new_winglet_positions[2])
        torques[1] = self.winglet_effectiveness * (new_winglet_positions[1] - new_winglet_positions[3])

        acceleration = (thrust + drag) / self.mass + self.gravity
        new_velocity = state.velocity + acceleration * self.dt
        new_position = state.position + new_velocity * self.dt

        angular_acceleration = torques / self.moment_of_inertia
        new_angular_velocity = state.angular_velocity + angular_acceleration * self.dt
        new_orientation = state.orientation + new_angular_velocity * self.dt

        self.time += self.dt

        sensors = {
            'accelerometer': self._rotate_vector(acceleration, new_orientation),
            'gyroscope': new_angular_velocity,
            'altitude': new_position[2],
            'velocity': np.linalg.norm(new_velocity)
        }

        new_state = RocketState(
            position=new_position,
            velocity=new_velocity,
            orientation=new_orientation,
            angular_velocity=new_angular_velocity,
            winglet_positions=new_winglet_positions
        )

        return new_state, sensors

# === Controller ===
class RocketController:
    def __init__(self):
        self.model = self._build_model()
        self.scaler_input = None
        self.scaler_output = None

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(10,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(4, activation='tanh')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_input(self, sensors, current_winglets):
        return np.concatenate([
            sensors['accelerometer'],
            sensors['gyroscope'],
            current_winglets
        ])

    def train(self, episodes=500):
        simulator = RocketSimulator()
        X, y = [], []

        # Collect data (random actions only)
        for episode in range(episodes):
            state = simulator.reset()
            while simulator.time < 3.0:
                _, sensors = simulator.step(state, state.winglet_positions)

                # Use random actions for data collection
                action = np.random.uniform(-1, 1, 4)

                X.append(self.prepare_input(sensors, state.winglet_positions))
                y.append(action)

                state, _ = simulator.step(state, action)

            if episode % 50 == 0:
                print(f"Data collection episode {episode} complete.")

        X = np.array(X)
        y = np.array(y)

        # Initialize and apply scalers
        self.scaler_input = (X.mean(axis=0), X.std(axis=0))
        self.scaler_output = (y.mean(axis=0), y.std(axis=0))

        X = (X - self.scaler_input[0]) / (self.scaler_input[1] + 1e-8)
        y = (y - self.scaler_output[0]) / (self.scaler_output[1] + 1e-8)

        # Train/test split
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Train model
        return self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=EPOCHS, batch_size=32, verbose=1)

# === Main ===
def main():
    controller = RocketController()
    print("Training controller...")
    history = controller.train(episodes=500)

    # Plot training loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training History")
    plt.legend()

    # Test flight
    sim = RocketSimulator()
    state = sim.reset()
    trajectory = []
    positions = []
    winglet_positions_log = []
    time_log = []       
    speed_log = []    

    while sim.time < 3.0:
        _, sensors = sim.step(state, state.winglet_positions)
        x = controller.prepare_input(sensors, state.winglet_positions)
        x_norm = (x - controller.scaler_input[0]) / (controller.scaler_input[1] + 1e-8)
        action_norm = controller.model.predict(x_norm.reshape(1, -1), verbose=0)[0]
        action = action_norm * controller.scaler_output[1] + controller.scaler_output[0]
        action = np.clip(action, -1, 1)
        state, _ = sim.step(state, action)
        trajectory.append(state.position.copy())
        positions.append(state.position.copy())
        winglet_positions_log.append(state.winglet_positions.copy())  # assuming it's a list or np.array
        time_log.append(sim.time)
        speed_log.append(state.velocity.copy())
        

    trajectory = np.array(trajectory)
    plt.subplot(1, 2, 2)
    plt.plot(trajectory[:, 2])
    plt.xlabel("Time step")
    plt.ylabel("Altitude (m)")
    plt.title("Test Flight Altitude")
    plt.tight_layout()
    plt.show()

    speed_log = np.array(speed_log)
    plt.figure(figsize=(12, 5))
    plt.plot(speed_log[:, 2])
    plt.xlabel("Time step")
    plt.ylabel("Speed (m/s)")
    plt.title("Test Flight Speed")
    plt.tight_layout()

    positions = np.array(positions)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Rocket trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Set equal scaling for x and y axes
    max_range = np.max([
        np.max(positions[:, 0]) - np.min(positions[:, 0]),
        np.max(positions[:, 1]) - np.min(positions[:, 1]),
        np.max(positions[:, 2]) - np.min(positions[:, 2])
    ])
    mid_x = (np.max(positions[:, 0]) + np.min(positions[:, 0])) / 2
    mid_y = (np.max(positions[:, 1]) + np.min(positions[:, 1])) / 2
    mid_z = (np.max(positions[:, 2]) + np.min(positions[:, 2])) / 2
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.set_title('3D Rocket Trajectory')
    ax.legend()

    winglet_positions_log = np.array(winglet_positions_log)
    time_log = np.array(time_log)

    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(time_log, winglet_positions_log[:, i], label=f'Winglet {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Winglet Position')
    plt.title('Winglet Positions During Flight')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

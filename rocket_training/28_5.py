import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
import json

@dataclass
class RocketState:
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    orientation: np.ndarray  # [roll, pitch, yaw]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    winglet_positions: np.ndarray  # [w1, w2, w3, w4]

class RocketSimulator:
    def __init__(self):
        self.mass = 0.5  # kg
        self.length = 0.5  # m
        self.drag_coefficient = 0.3
        self.winglet_effectiveness = 0.1
        self.dt = 0.01  # 100Hz simulation
        self.gravity = np.array([0, 0, -9.81])
        self.thrust_curve = self._generate_thrust_curve()
        self.time = 0.0
        
    def _generate_thrust_curve(self):
        # Simple thrust curve for solid rocket motor
        times = np.array([0, 0.1, 0.5, 1.0, 1.5, 2.0])
        thrusts = np.array([0, 50, 40, 30, 10, 0])  # Newtons
        return lambda t: np.interp(t, times, thrusts)
    
    def reset(self) -> RocketState:
        self.time = 0.0
        return RocketState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            winglet_positions=np.array([0.0, 0.0, 0.0, 0.0])
        )
    
    def step(self, state: RocketState, winglet_commands: np.ndarray) -> Tuple[RocketState, dict]:
        # Update winglet positions (with servo dynamics)
        state.winglet_positions = 0.9 * state.winglet_positions + 0.1 * winglet_commands
        
        # Calculate forces
        thrust = self.thrust_curve(self.time) * np.array([0, 0, 1])
        drag = -0.5 * self.drag_coefficient * np.linalg.norm(state.velocity) * state.velocity
        
        # Winglet forces create torques
        torques = np.zeros(3)
        torques[0] = self.winglet_effectiveness * (state.winglet_positions[0] - state.winglet_positions[2])  # Roll
        torques[1] = self.winglet_effectiveness * (state.winglet_positions[1] - state.winglet_positions[3])  # Pitch
        
        # Update dynamics
        acceleration = (thrust + drag) / self.mass + self.gravity
        state.velocity += acceleration * self.dt
        state.position += state.velocity * self.dt
        
        state.angular_velocity += torques * self.dt
        state.orientation += state.angular_velocity * self.dt
        
        self.time += self.dt
        
        # Calculate sensor readings
        sensors = {
            'accelerometer': self._rotate_vector(acceleration, state.orientation),
            'gyroscope': state.angular_velocity,
            'altitude': state.position[2],
            'velocity': np.linalg.norm(state.velocity)
        }
        
        return state, sensors
    
    def _rotate_vector(self, vector: np.ndarray, angles: np.ndarray) -> np.ndarray:
        # Simplified rotation (for more accuracy, use quaternions)
        roll, pitch, yaw = angles
        # Apply rotations
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        return Rx @ Ry @ vector

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
        return model
    
    def prepare_input(self, sensors: dict, current_winglets: np.ndarray) -> np.ndarray:
        # Combine sensor data and current winglet positions
        return np.concatenate([
            sensors['accelerometer'],
            sensors['gyroscope'],
            current_winglets
        ])
    
    def train(self, episodes: int = 1000):
        simulator = RocketSimulator()
        
        # Collect training data
        states_data = []
        actions_data = []
        rewards_data = []
        
        for episode in range(episodes):
            state = simulator.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            
            while simulator.time < 3.0:  # 3 second flights
                # Get sensor readings
                _, sensors = simulator.step(state, state.winglet_positions)
                
                # Random exploration with some guidance
                if episode < episodes // 2:
                    # Early training: random exploration
                    action = np.random.uniform(-1, 1, 4)
                else:
                    # Later training: use model with exploration
                    nn_input = self.prepare_input(sensors, state.winglet_positions)
                    action = self.model.predict(nn_input.reshape(1, -1))[0]
                    action += np.random.normal(0, 0.1, 4)
                    action = np.clip(action, -1, 1)
                
                # Store data
                episode_states.append(self.prepare_input(sensors, state.winglet_positions))
                episode_actions.append(action)
                
                # Calculate reward (maximize altitude, minimize tilt)
                reward = state.position[2] - 10 * (state.orientation[0]**2 + state.orientation[1]**2)
                episode_rewards.append(reward)
                
                # Update state
                state, _ = simulator.step(state, action)
            
            states_data.extend(episode_states)
            actions_data.extend(episode_actions)
            rewards_data.extend(episode_rewards)
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Max altitude = {max([s[2] for s in episode_states]):.2f}m")
        
        # Convert to numpy arrays
        X = np.array(states_data)
        y = np.array(actions_data)
        
        # Normalize data
        self.scaler_input = (X.mean(axis=0), X.std(axis=0))
        self.scaler_output = (y.mean(axis=0), y.std(axis=0))
        
        X_normalized = (X - self.scaler_input[0]) / (self.scaler_input[1] + 1e-8)
        y_normalized = (y - self.scaler_output[0]) / (self.scaler_output[1] + 1e-8)
        
        # Train model
        self.model.compile(optimizer='adam', loss='mse')
        history = self.model.fit(X_normalized, y_normalized, 
                                epochs=50, batch_size=32, 
                                validation_split=0.2, verbose=1)
        
        return history
    
    def export_for_esp32(self, filename: str = "rocket_nn_params.h"):
        # Extract weights and biases
        weights = []
        biases = []
        
        for layer in self.model.layers:
            w, b = layer.get_weights()
            weights.append(w)
            biases.append(b)
        
        # Generate C header file
        with open(filename, 'w') as f:
            f.write("#ifndef ROCKET_NN_PARAMS_H\n")
            f.write("#define ROCKET_NN_PARAMS_H\n\n")
            
            # Network architecture
            f.write("#define INPUT_SIZE 10\n")
            f.write("#define HIDDEN1_SIZE 32\n")
            f.write("#define HIDDEN2_SIZE 16\n")
            f.write("#define OUTPUT_SIZE 4\n\n")
            
            # Scaling parameters
            f.write("// Input normalization parameters\n")
            f.write(f"const float input_mean[INPUT_SIZE] = {{{', '.join(map(str, self.scaler_input[0]))}}};\n")
            f.write(f"const float input_std[INPUT_SIZE] = {{{', '.join(map(str, self.scaler_input[1]))}}};\n\n")
            
            f.write("// Output denormalization parameters\n")
            f.write(f"const float output_mean[OUTPUT_SIZE] = {{{', '.join(map(str, self.scaler_output[0]))}}};\n")
            f.write(f"const float output_std[OUTPUT_SIZE] = {{{', '.join(map(str, self.scaler_output[1]))}}};\n\n")
            
            # Weights and biases
            for i, (w, b) in enumerate(zip(weights, biases)):
                layer_name = f"layer{i+1}"
                
                # Flatten weights and write
                f.write(f"// Layer {i+1} weights\n")
                f.write(f"const float {layer_name}_weights[{w.size}] = {{\n")
                flat_w = w.flatten()
                for j in range(0, len(flat_w), 10):
                    f.write("    " + ", ".join(map(str, flat_w[j:j+10])) + ",\n")
                f.write("};\n\n")
                
                # Write biases
                f.write(f"// Layer {i+1} biases\n")
                f.write(f"const float {layer_name}_bias[{len(b)}] = {{{', '.join(map(str, b))}}};\n\n")
            
            f.write("#endif // ROCKET_NN_PARAMS_H\n")
        
        print(f"Model exported to {filename}")
        
        # Also save model in TensorFlow Lite format for comparison
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open('rocket_model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print("Model also exported as rocket_model.tflite")

# Usage example
if __name__ == "__main__":
    controller = RocketController()
    print("Training rocket controller...")
    history = controller.train(episodes=5)
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    
    plt.subplot(1, 2, 2)
    # Test the trained model
    simulator = RocketSimulator()
    state = simulator.reset()
    positions = []
    
    while simulator.time < 3.0:
        _, sensors = simulator.step(state, state.winglet_positions)
        nn_input = controller.prepare_input(sensors, state.winglet_positions)
        nn_input_norm = (nn_input - controller.scaler_input[0]) / (controller.scaler_input[1] + 1e-8)
        action_norm = controller.model.predict(nn_input_norm.reshape(1, -1))[0]
        action = action_norm * controller.scaler_output[1] + controller.scaler_output[0]
        action = np.clip(action, -1, 1)
        state, _ = simulator.step(state, action)
        positions.append(state.position.copy())
    
    positions = np.array(positions)
    plt.plot(positions[:, 2])
    plt.xlabel('Time steps')
    plt.ylabel('Altitude (m)')
    plt.title('Test Flight Altitude')
    plt.tight_layout()
    plt.show()
    
    # Export for ESP32
    # controller.export_for_esp32()
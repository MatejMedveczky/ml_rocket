"""
Scientific Rocket Neural Network Training Framework - LSTM Version
==================================================================
A framework for training LSTM neural networks to control rockets
with complementary filter for orientation estimation.

This version includes:
- Complementary filter for IMU fusion (gyro + accelerometer)
- LSTM network for temporal sequence learning
- Corrected accelerometer physics model
- Stabilization-focused reward function
- 12 input observations (gyro, accel, orientation, winglets)
- 4 output controls (winglets only)

Dependencies: numpy, matplotlib, pandas (optional for data export)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import time
from collections import deque

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class RocketConfig:
    """Rocket physical parameters"""
    mass: float = 0.3              # kg
    drag_coefficient: float = 0.3
    max_gimbal_angle: float = 15.0  # degrees
    max_gimbal_rate: float = 30.0    # degrees per second
    max_thrust: float = 36.0       # Newtons
    burn_time: float = 2.0         # seconds
    initial_pitch: float = 10.0    # degrees

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    population_size: int = 30
    generations: int = 50
    mutation_rate: float = 0.1
    mutation_strength: float = 0.4
    crossover_rate: float = 0.7
    elite_count: int = 5
    max_episode_steps: int = 3000
    dt: float = 0.01
    save_network: bool = True
    use_lstm: bool = True  # Set to True to use LSTM

@dataclass
class NeuralNetConfig:
    """Neural network architecture"""
    input_size: int = 10  # gyro(3) + accel(3) + orientation(2) + gimbal(2)
    hidden_size: int = 16
    output_size: int = 2   # ← CHANGED: Only 2 outputs (pitch gimbal, roll gimbal)
    
# ==============================================================================
# DATA LOGGER
# ==============================================================================

class DataLogger:
    """Comprehensive data logging for analysis"""
    
    def __init__(self):
        self.generation_data = []
        self.episode_data = []
        self.current_episode = []
        
        # Real-time metrics
        self.fitness_history = []
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.diversity_history = []
        
    def log_step(self, rocket_state: dict, controls: dict, reward: float):
        """Log a single timestep"""
        self.current_episode.append({
            'position': rocket_state['position'].copy(),
            'velocity': rocket_state['velocity'].copy(),
            'orientation': rocket_state['orientation'].copy(),
            'controls': controls.copy(),
            'reward': reward,
            'timestamp': len(self.current_episode) * 0.01
        })
    
    def end_episode(self, fitness: float, rocket_id: int):
        """Finalize episode logging"""
        if self.current_episode:
            self.episode_data.append({
                'rocket_id': rocket_id,
                'fitness': fitness,
                'trajectory': self.current_episode
            })
            self.current_episode = []
    
    def log_generation(self, generation: int, population_fitness: List[float], 
                       best_network_weights=None):
        """Log generation statistics"""
        stats = {
            'generation': generation,
            'best_fitness': max(population_fitness),
            'mean_fitness': np.mean(population_fitness),
            'std_fitness': np.std(population_fitness),
            'min_fitness': min(population_fitness),
            'median_fitness': np.median(population_fitness)
        }
        
        self.generation_data.append(stats)
        self.best_fitness_history.append(stats['best_fitness'])
        self.mean_fitness_history.append(stats['mean_fitness'])
        
        # Calculate diversity (variance in fitness)
        self.diversity_history.append(stats['std_fitness'])
    
    def export_data(self, filename: str):
        """Export data to JSON for external analysis"""
        export_data = {
            'generations': self.generation_data,
            'fitness_history': {
                'best': self.best_fitness_history,
                'mean': self.mean_fitness_history,
                'diversity': self.diversity_history
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

# ==============================================================================
# ROCKET CLASS WITH COMPLEMENTARY FILTER
# ==============================================================================

class Rocket:
    """Rocket with realistic physics simulation and complementary filter"""
    
    def __init__(self, config: RocketConfig = None):
        if config is None:
            config = RocketConfig()
        self.config = config
        
        # State vectors
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.angular_velocity = np.zeros(3)
        
        # Controls
        self.gimbal = np.zeros(2)
        self.gimbal_target = np.zeros(2)
        self.thrust = 0.0
        
        # Sensors (raw)
        self.gyro = np.zeros(3)
        self.accel = np.zeros(3)
        
        # Sensor biases (for drift simulation)
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        
        # Estimated orientation (from complementary filter)
        self.estimated_orientation = np.zeros(3)
        
        # Telemetry
        self.flight_time = 0.0
        self.max_altitude = 0.0
        self.total_distance = 0.0
        self.dt = 0.01
        
    def reset(self):
        """Reset rocket state with small perturbations"""
        
        self.position = np.array([0.0, 0.0, 0.1])
        self.velocity = np.zeros(3)

        self.orientation = np.array([0.0, np.deg2rad(self.config.initial_pitch), 0.0])
        self.angular_velocity = np.zeros(3)
        self.gimbal = np.zeros(2)
        self.thrust = 0.0
        
        # Reset sensor biases with small initial drift
        self.gyro_bias = np.random.randn(3) * 0.005
        self.accel_bias = np.random.randn(3) * 0.05
        
        # Initialize estimated orientation to true orientation (with small error)
        self.estimated_orientation = self.orientation.copy() + np.random.randn(3) * 0.05
        
        self.flight_time = 0.0
        self.max_altitude = 0.0
        self.total_distance = 0.0
        
    def get_rotation_matrix(self) -> np.ndarray:
        """Compute combined rotation matrix"""
        roll, pitch, yaw = self.orientation
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])
        
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                      [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])
        
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
        
        return Rz @ Ry @ Rx
    
    def compute_thrust(self, time: float) -> float:
        """Compute thrust based on burn time"""
        if time < self.config.burn_time:
            return self.config.max_thrust
        return 0.0
    
    def update(self, dt: float):
        """Update rocket physics"""
        
        # NEW: Apply rate limiting to gimbal movement
        max_gimbal_angle = np.deg2rad(self.config.max_gimbal_angle)
        max_gimbal_rate = np.deg2rad(self.config.max_gimbal_rate)  # degrees/s → rad/s
        
        # Calculate desired change in gimbal position
        gimbal_error = self.gimbal_target - self.gimbal
        
        # Limit the rate of change
        max_change_this_step = max_gimbal_rate * dt
        gimbal_change = np.clip(gimbal_error, -max_change_this_step, max_change_this_step)
        
        # Update actual gimbal position
        self.gimbal += gimbal_change
        
        # Ensure gimbal stays within physical limits
        self.gimbal = np.clip(self.gimbal, -max_gimbal_angle, max_gimbal_angle)
        
        # Forces
        forces = np.array([0.0, 0.0, -9.81 * self.config.mass])
        
        # Thrust with vectoring - USE ACTUAL GIMBAL POSITION (not target)
        thrust_magnitude = self.compute_thrust(self.flight_time)
        if thrust_magnitude > 0:
            R = self.get_rotation_matrix()
            
            # Use ACTUAL gimbal position (after rate limiting)
            thrust_direction = np.array([
                self.gimbal[1],   # Roll deflection (x-axis)
                self.gimbal[0],   # Pitch deflection (y-axis)
                1.0               # Main thrust (z-axis)
            ])
            thrust_direction = thrust_direction / np.linalg.norm(thrust_direction)
            
            # Transform to world frame
            thrust_vector = R @ (thrust_direction * thrust_magnitude)
            forces += thrust_vector
        
        # Drag - UNCHANGED
        if np.linalg.norm(self.velocity) > 0.01:
            drag = -self.config.drag_coefficient * self.velocity * np.linalg.norm(self.velocity)
            forces += drag
        
        # REMOVE winglet torques entirely (or comment out)
        # No aerodynamic control torques needed
        torques = np.zeros(3)  # ← SIMPLIFIED: No control torques (thrust does it)
        
        # Update angular velocity - UNCHANGED
        angular_damping = 0.9
        self.angular_velocity = self.angular_velocity * angular_damping + torques * dt
        
        # Update orientation
        self.orientation += self.angular_velocity * dt
        self.orientation[1] = np.clip(self.orientation[1], -np.pi/2, np.pi/2)  # Limit pitch
        
        # Update linear motion
        acceleration = forces / self.config.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Ground collision
        if self.position[2] < 0:
            self.position[2] = 0
            self.velocity[2] = max(0, self.velocity[2])
        
        # Update sensors (includes complementary filter)
        self.update_sensors(acceleration)
        
        # Update telemetry
        self.flight_time += dt
        self.max_altitude = max(self.max_altitude, self.position[2])
        self.total_distance += np.linalg.norm(self.velocity) * dt
        
        self.thrust = thrust_magnitude / self.config.max_thrust  # Normalized
    
    def update_sensors(self, acceleration: np.ndarray):
        """Update IMU sensors with realistic noise and bias drift"""
        R = self.get_rotation_matrix()
        
        # CRITICAL FIX: Accelerometer measures specific force (includes gravity effect)
        # When sitting still, accelerometer reads +9.81 m/s² upward (not zero)
        gravity_world = np.array([0, 0, -9.81])
        
        # Total acceleration in world frame
        total_accel_world = acceleration
        
        # Specific force = what accelerometer actually measures
        # (thrust + drag) / mass, transformed to body frame
        specific_force_world = total_accel_world - gravity_world
        body_specific_force = R.T @ specific_force_world
        
        # Add bias drift (simulates real IMU behavior)
        self.gyro_bias += np.random.randn(3) * 0.0001 * self.dt
        self.accel_bias += np.random.randn(3) * 0.001 * self.dt
        
        # Add noise + bias
        self.gyro = self.angular_velocity + self.gyro_bias + np.random.randn(3) * 0.01
        self.accel = body_specific_force + self.accel_bias + np.random.randn(3) * 0.1
        
        # Run complementary filter (what microcontroller will do)
        self.update_orientation_estimate(self.dt)
    
    def update_orientation_estimate(self, dt: float):
        """
        Complementary filter for orientation estimation
        Fuses gyroscope (high-frequency, drifts) with accelerometer (low-frequency, no drift)
        This is what your microcontroller MUST implement
        """
        # Step 1: Integrate gyroscope for orientation prediction
        gyro_orientation = self.estimated_orientation + self.gyro * dt
        
        # Step 2: Get orientation estimate from accelerometer
        # Only valid when not accelerating (near 1g total acceleration)
        accel_magnitude = np.linalg.norm(self.accel)
        
        # Complementary filter weight (trust gyro more for fast changes)
        alpha = 0.98  # 98% gyro, 2% accel
        
        if 8.0 < accel_magnitude < 11.0:  # Near 1g, can trust accelerometer
            # Calculate roll and pitch from gravity vector
            accel_roll = np.arctan2(self.accel[1], self.accel[2])
            accel_pitch = np.arctan2(-self.accel[0], 
                                     np.sqrt(self.accel[1]**2 + self.accel[2]**2))
            
            # Fuse gyro and accel estimates
            self.estimated_orientation[0] = (alpha * gyro_orientation[0] + 
                                            (1 - alpha) * accel_roll)
            self.estimated_orientation[1] = (alpha * gyro_orientation[1] + 
                                            (1 - alpha) * accel_pitch)
        else:
            # High acceleration - can't trust accelerometer, use only gyro
            self.estimated_orientation[0] = gyro_orientation[0]
            self.estimated_orientation[1] = gyro_orientation[1]
        
        # Yaw cannot be corrected by accelerometer (would need magnetometer)
        self.estimated_orientation[2] = gyro_orientation[2]
        
        # Wrap angles to [-pi, pi]
        self.estimated_orientation = np.arctan2(np.sin(self.estimated_orientation), 
                                                np.cos(self.estimated_orientation))
    
    def get_state_dict(self) -> dict:
        """Get state as dictionary for logging"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'orientation': self.orientation.copy(),
            'estimated_orientation': self.estimated_orientation.copy(),
            'angular_velocity': self.angular_velocity.copy(),
            'gimbal_actual': self.gimbal.copy(),
            'gimbal_target': self.gimbal_target.copy(),
            'altitude': self.position[2],
            'speed': np.linalg.norm(self.velocity)
        }
    
    def test_control_authority(self):
        """Test control at TRUE launch (zero velocity, no perturbations)"""
        print("\n" + "="*60)
        print("REALISTIC CONTROL TEST - TRUE LAUNCH CONDITIONS")
        print("="*60)
        
        # Hard reset with ZERO velocity
        self.position = np.array([0.0, 0.0, 0.1])
        self.velocity = np.array([0.0, 0.0, 0.0])  # ← TRULY zero
        self.orientation = np.array([0.1, 0.0, 0.0])  # 5.7° initial tilt
        self.angular_velocity = np.zeros(3)

        self.gimbal = np.array([-1, 1])  # Max roll control

        initial_tilt = self.orientation[0]
        
        print(f"Initial conditions:")
        print(f"  Airspeed: {np.linalg.norm(self.velocity):.2f} m/s")
        print(f"  Initial tilt: {np.rad2deg(initial_tilt):.2f}°")
        print(f"  Control effectiveness: {min(1.0, np.linalg.norm(self.velocity) / 5.0):.2%}")
        
        # Simulate first 0.1 seconds (critical launch phase)
        print(f"\nFirst 0.1 seconds (CRITICAL PHASE):")
        for i in range(10):
            self.update(0.01)
            if i == 9:  # After 0.1s
                airspeed = np.linalg.norm(self.velocity)
                effectiveness = min(1.0, airspeed / 5.0)
                tilt = np.rad2deg(self.orientation[0])
                print(f"  t=0.1s: airspeed={airspeed:.2f}m/s, effectiveness={effectiveness:.1%}, tilt={tilt:.2f}°")
        
        # Continue to 0.5s
        for i in range(40):
            self.update(0.01)
        
        final_tilt = self.orientation[0]
        airspeed = np.linalg.norm(self.velocity)
        effectiveness = min(1.0, airspeed / 5.0)
        
        print(f"\nAfter 0.5 seconds:")
        print(f"  Airspeed: {airspeed:.2f} m/s")
        print(f"  Control effectiveness: {effectiveness:.1%}")
        print(f"  Initial tilt: {np.rad2deg(initial_tilt):.2f}°")
        print(f"  Final tilt: {np.rad2deg(final_tilt):.2f}°")
        print(f"  Change: {np.rad2deg(final_tilt - initial_tilt):.2f}°")
        
        if abs(final_tilt) > abs(initial_tilt):
            print(f"\n  ❌ FAILED: Tilt INCREASED (controls ineffective at launch)")
        else:
            print(f"\n  ✓ Corrected by {np.rad2deg(initial_tilt - final_tilt):.2f}°")

# ==============================================================================
# LSTM NEURAL NETWORK
# ==============================================================================

class LSTMNetwork:
    """LSTM network for temporal sequence processing"""
    
    def __init__(self, config: NeuralNetConfig = None):
        if config is None:
            config = NeuralNetConfig()
        
        self.config = config
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        
        # LSTM parameters (simplified single-layer LSTM)
        # Gates: input, forget, output, cell
        self.hidden_state = np.zeros(self.hidden_size)
        self.cell_state = np.zeros(self.hidden_size)
        
        # Weights for LSTM gates (input_size + hidden_size -> hidden_size)
        input_dim = self.input_size + self.hidden_size
        
        # Input gate
        self.W_i = np.random.randn(input_dim, self.hidden_size) * 0.1
        self.b_i = np.zeros(self.hidden_size)
        
        # Forget gate
        self.W_f = np.random.randn(input_dim, self.hidden_size) * 0.1
        self.b_f = np.ones(self.hidden_size)  # Bias to 1 (remember by default)
        
        # Output gate
        self.W_o = np.random.randn(input_dim, self.hidden_size) * 0.1
        self.b_o = np.zeros(self.hidden_size)
        
        # Cell candidate
        self.W_c = np.random.randn(input_dim, self.hidden_size) * 0.1
        self.b_c = np.zeros(self.hidden_size)
        
        # Output layer (hidden -> output)
        self.W_out = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b_out = np.zeros(self.output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through LSTM"""
        # Concatenate input with previous hidden state
        combined = np.concatenate([x, self.hidden_state])
        
        # LSTM gates
        input_gate = self.sigmoid(combined @ self.W_i + self.b_i)
        forget_gate = self.sigmoid(combined @ self.W_f + self.b_f)
        output_gate = self.sigmoid(combined @ self.W_o + self.b_o)
        cell_candidate = self.tanh(combined @ self.W_c + self.b_c)
        
        # Update cell state
        self.cell_state = forget_gate * self.cell_state + input_gate * cell_candidate
        
        # Update hidden state
        self.hidden_state = output_gate * self.tanh(self.cell_state)
        
        # Output layer
        output = self.tanh(self.hidden_state @ self.W_out + self.b_out)
        
        return output
    
    def reset_state(self):
        """Reset LSTM hidden states (call at episode start)"""
        self.hidden_state = np.zeros(self.hidden_size)
        self.cell_state = np.zeros(self.hidden_size)
    
    def mutate(self, rate: float, strength: float):
        """Apply mutations to all weights"""
        weights = [self.W_i, self.W_f, self.W_o, self.W_c, self.W_out]
        biases = [self.b_i, self.b_f, self.b_o, self.b_c, self.b_out]
        
        for w in weights:
            mask = np.random.random(w.shape) < rate
            w[mask] += np.random.randn(np.sum(mask)) * strength
        
        for b in biases:
            mask = np.random.random(b.shape) < rate
            b[mask] += np.random.randn(np.sum(mask)) * strength
    
    def crossover(self, other: 'LSTMNetwork') -> 'LSTMNetwork':
        """Create offspring by mixing genes"""
        child = LSTMNetwork(self.config)
        
        # Mix all weight matrices
        weights_pairs = [
            (self.W_i, other.W_i, child.W_i),
            (self.W_f, other.W_f, child.W_f),
            (self.W_o, other.W_o, child.W_o),
            (self.W_c, other.W_c, child.W_c),
            (self.W_out, other.W_out, child.W_out),
        ]
        
        biases_pairs = [
            (self.b_i, other.b_i, child.b_i),
            (self.b_f, other.b_f, child.b_f),
            (self.b_o, other.b_o, child.b_o),
            (self.b_c, other.b_c, child.b_c),
            (self.b_out, other.b_out, child.b_out),
        ]
        
        for w1, w2, wc in weights_pairs:
            mask = np.random.random(w1.shape) > 0.5
            wc[:] = np.where(mask, w1, w2)
        
        for b1, b2, bc in biases_pairs:
            mask = np.random.random(b1.shape) > 0.5
            bc[:] = np.where(mask, b1, b2)
        
        return child
    
    def copy(self) -> 'LSTMNetwork':
        """Deep copy"""
        copy_net = LSTMNetwork(self.config)
        copy_net.W_i = self.W_i.copy()
        copy_net.W_f = self.W_f.copy()
        copy_net.W_o = self.W_o.copy()
        copy_net.W_c = self.W_c.copy()
        copy_net.W_out = self.W_out.copy()
        copy_net.b_i = self.b_i.copy()
        copy_net.b_f = self.b_f.copy()
        copy_net.b_o = self.b_o.copy()
        copy_net.b_c = self.b_c.copy()
        copy_net.b_out = self.b_out.copy()
        return copy_net
    
    def get_weights_dict(self) -> dict:
        """Export weights for microcontroller deployment"""
        return {
            'W_i': self.W_i.tolist(),
            'b_i': self.b_i.tolist(),
            'W_f': self.W_f.tolist(),
            'b_f': self.b_f.tolist(),
            'W_o': self.W_o.tolist(),
            'b_o': self.b_o.tolist(),
            'W_c': self.W_c.tolist(),
            'b_c': self.b_c.tolist(),
            'W_out': self.W_out.tolist(),
            'b_out': self.b_out.tolist()
        }

# ==============================================================================
# ENVIRONMENT
# ==============================================================================

class Environment:
    """Scientific rocket training environment with complementary filter"""

    def __init__(self, rocket_config: RocketConfig = None):
        self.rocket = Rocket(rocket_config)
        self.time_step = 0
        self.max_steps = 3000
        self.dt = 0.01

        self.previous_gimbal = np.zeros(2)

        # Metrics
        self.episode_reward = 0.0
        self.trajectory = []
        
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.rocket.reset()
        self.time_step = 0
        self.previous_gimbal = np.zeros(2)
        self.episode_reward = 0.0
        self.trajectory = []
        
        return self.get_observation()
    
    def get_observation(self) -> np.ndarray:
        """Get sensor observation vector - with filtered orientation"""
        return np.concatenate([
            self.rocket.gyro,                      # 3: angular velocity (rad/s)
            self.rocket.accel,                     # 3: specific force (m/s²)
            self.rocket.estimated_orientation[:2], # 2: roll, pitch (rad) - from filter
            self.previous_gimbal                     # 2: previous gimbal control
        ])  # Total: 12 inputs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one simulation step"""
        # Parse action - network outputs DESIRED gimbal position [-1, 1]
        gimbal_command = action[:2]
        
        # Convert to target angle in radians
        max_gimbal_angle = np.deg2rad(self.rocket.config.max_gimbal_angle)
        self.rocket.gimbal_target = np.clip(gimbal_command, -1, 1) * max_gimbal_angle
        
        # Update physics (gimbal will move towards target with rate limit)
        self.rocket.update(self.dt)
        
        # Store ACTUAL gimbal position for next observation (normalized)
        self.previous_gimbal = self.rocket.gimbal / max_gimbal_angle  # Back to [-1, 1]
        
        # Calculate reward
        reward = self.calculate_reward()
        self.episode_reward += reward
        
        # Store trajectory
        self.trajectory.append(self.rocket.get_state_dict())
        
        done = self.is_done()
        self.time_step += 1
        
        info = {
            'altitude': self.rocket.position[2],
            'speed': np.linalg.norm(self.rocket.velocity),
            'tilt': np.abs(self.rocket.estimated_orientation[:2]).sum(),
            'time': self.time_step * self.dt
        }
        
        return self.get_observation(), reward, done, info
    
    def calculate_reward(self) -> float:
        """Calculate step reward"""
        reward = 0.0
        
        tilt = np.abs(self.rocket.estimated_orientation[:2]).sum()
        reward -= tilt * 10.0
        
        angular_rate = np.abs(self.rocket.gyro[:2]).sum()
        reward -= angular_rate * 2.0
        
        gimbal_effort = np.abs(self.rocket.gimbal).sum()
        reward -= gimbal_effort * 0.1
        
        if tilt < 0.1 and angular_rate < 0.5:
            reward += 5.0
        
        if self.rocket.position[2] <= 0 and self.time_step > 100:
            reward -= 50.0
        
        return reward
    
    def is_done(self) -> bool:
        """Check termination conditions"""
        if self.time_step >= self.max_steps:
            return True
        
        if self.rocket.position[2] <= 0 and self.time_step > 100:
            return True
        
        horizontal_dist = np.linalg.norm(self.rocket.position[:2])
        if horizontal_dist > 50:
            return True
        
        if self.rocket.position[2] > 100:
            return True
        
        return False
    
    def get_fitness(self) -> float:
        """Calculate final fitness score - stabilization focused"""
        fitness = 0.0
        
        # Primary: survival time
        fitness += self.time_step * 1.0
        
        # Penalize average tilt throughout flight
        if len(self.trajectory) > 0:
            avg_tilt = np.mean([np.abs(state['estimated_orientation'][:2]).sum() 
                               for state in self.trajectory])
            fitness -= avg_tilt * 100.0
        
        # Bonus for ending upright
        final_tilt = np.abs(self.rocket.estimated_orientation[:2]).sum()
        if final_tilt < 0.2:
            fitness += 100.0
        
        # Bonus for max altitude achieved (secondary objective)
        fitness += self.rocket.max_altitude * 2.0
        
        return fitness

# ==============================================================================
# TRAINER
# ==============================================================================

class Trainer:
    """Main training orchestrator with LSTM support"""
    
    def __init__(self, config: TrainingConfig = None):
        if config is None:
            config = TrainingConfig()
        
        self.config = config
        self.logger = DataLogger()
        
        # Initialize population
        nn_config = NeuralNetConfig()
        nn_config.use_lstm = config.use_lstm
        

        self.population = [LSTMNetwork(nn_config) for _ in range(config.population_size)]
        
        self.use_lstm = config.use_lstm
        
        # Training state
        self.generation = 0
        self.best_network = None
        self.best_fitness_ever = -float('inf')
        
        # Visualization
        self.setup_visualization()
        
    # Find the Trainer.setup_visualization() method and replace it with this:

    def setup_visualization(self):
        """Setup matplotlib figures for real-time plotting"""
        plt.ion()  # Interactive mode
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 7))  # ← Made wider
        self.fig.suptitle('Rocket LSTM Network Training', fontsize=16)
        
        gs = GridSpec(2, 4, figure=self.fig, hspace=0.3, wspace=0.3)  # ← Changed to 4 columns
        
        # 3D trajectory plot
        self.ax_3d = self.fig.add_subplot(gs[0:2, 0:2], projection='3d')
        self.ax_3d.set_title('Rocket Trajectories')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        
        # Fitness history
        self.ax_fitness = self.fig.add_subplot(gs[0, 2])
        self.ax_fitness.set_title('Fitness Evolution')
        self.ax_fitness.set_xlabel('Generation')
        self.ax_fitness.set_ylabel('Fitness')
        
        # Population distribution
        self.ax_dist = self.fig.add_subplot(gs[1, 2])
        self.ax_dist.set_title('Population Distribution')
        self.ax_dist.set_xlabel('Fitness')
        self.ax_dist.set_ylabel('Count')
        
        # NEW: Gimbal visualization
        self.ax_gimbal = self.fig.add_subplot(gs[0:2, 3])  # ← NEW
        self.ax_gimbal.set_title('Gimbal Position')
        self.ax_gimbal.set_xlim([-1.0, 1.0])
        self.ax_gimbal.set_ylim([-0.2, 2.0])
        self.ax_gimbal.set_aspect('equal')

        plt.show()

    def evaluate_population(self, visualize_best: bool = True):
        """Evaluate entire population"""
        fitnesses = []
        
        # Create environments for each network
        envs = [Environment() for _ in range(self.config.population_size)]
        observations = [env.reset() for env in envs]
        dones = [False] * self.config.population_size
        altitudes = [0.0] * self.config.population_size
        
        # Reset LSTM states
        for network in self.population:
            network.reset_state()
        
        # Run simulation
        while not all(dones):
            for i in range(self.config.population_size):
                if not dones[i]:
                    action = self.population[i].forward(observations[i])
                    obs, reward, done, info = envs[i].step(action)
                    observations[i] = obs
                    dones[i] = done
                    if altitudes[i] < info['altitude']:
                        altitudes[i] = info['altitude']
                    
                    # Reset LSTM state on episode end
                    if done:
                        self.population[i].reset_state()
        
        # Calculate fitness
        fitnesses = [env.get_fitness() for env in envs]
        
        # Visualize best performer
        if visualize_best:
            best_idx = np.argmax(fitnesses)
            self.visualize_trajectory(envs[best_idx])
        
        return fitnesses, envs, altitudes
    
    def evolve_population(self, fitnesses: List[float]):
        """Create next generation using genetic algorithm"""
        # Sort by fitness
        sorted_pairs = sorted(zip(fitnesses, self.population), key=lambda x: x[0], reverse=True)
        sorted_fitness, sorted_pop = zip(*sorted_pairs)
        
        # Track best
        if sorted_fitness[0] > self.best_fitness_ever:
            self.best_fitness_ever = sorted_fitness[0]
            self.best_network = sorted_pop[0].copy()
        
        # Log generation data
        self.logger.log_generation(self.generation, fitnesses)
        
        # Create new population
        new_population = []
        
        # Elitism
        for i in range(self.config.elite_count):
            new_population.append(sorted_pop[i].copy())
        
        # Crossover and mutation
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1_idx = np.random.choice(min(5, len(sorted_pop)))
            parent2_idx = np.random.choice(min(5, len(sorted_pop)))
            
            parent1 = sorted_pop[parent1_idx]
            parent2 = sorted_pop[parent2_idx]
            
            # Crossover
            if np.random.random() < self.config.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            child.mutate(self.config.mutation_rate, self.config.mutation_strength)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    # Find the Trainer.visualize_trajectory() method and replace it with this:

    def visualize_trajectory(self, env: Environment):
        """Update 3D trajectory visualization and gimbal display"""
        if not env.trajectory:
            return
        
        # Extract trajectory data
        positions = np.array([state['position'] for state in env.trajectory])
        
        # Clear and plot 3D trajectory
        self.ax_3d.clear()
        self.ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.7)
        self.ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                        c='g', s=100, marker='o', label='Start')
        self.ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                        c='r', s=100, marker='x', label='End')
        
        self.ax_3d.set_xlim([-20, 20])
        self.ax_3d.set_ylim([-20, 20])
        self.ax_3d.set_zlim([0, 60])
        self.ax_3d.legend()
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        
        # NEW: Gimbal trajectory (2D path)
        self.ax_gimbal.clear()

        if env.trajectory and len(env.trajectory) > 1:
            # Extract gimbal positions over time
            max_angle = np.deg2rad(15.0)
            pitch_history = [state['gimbal_actual'][0] / max_angle for state in env.trajectory]
            roll_history = [state['gimbal_actual'][1] / max_angle for state in env.trajectory]
            
            # Circle boundary
            circle = plt.Circle((0, 0), 1.0, fill=False, color='gray', linewidth=2)
            self.ax_gimbal.add_patch(circle)
            
            # Crosshairs
            self.ax_gimbal.plot([-1, 1], [0, 0], 'gray', alpha=0.3, linewidth=1)
            self.ax_gimbal.plot([0, 0], [-1, 1], 'gray', alpha=0.3, linewidth=1)
            
            # Plot gimbal path
            self.ax_gimbal.plot(roll_history, pitch_history, 'r-', alpha=0.7, linewidth=1.5)
            
            # Current position (end point)
            self.ax_gimbal.scatter(roll_history[-1], pitch_history[-1], s=100, c='red', marker='o', zorder=5)

        self.ax_gimbal.set_xlim([-1.2, 1.2])
        self.ax_gimbal.set_ylim([-1.2, 1.2])
        self.ax_gimbal.set_aspect('equal')
        self.ax_gimbal.set_xlabel('Roll')
        self.ax_gimbal.set_ylabel('Pitch')
        self.ax_gimbal.set_title('Gimbal Path')

    
    def update_plots(self, fitnesses: List[float]):
        """Update all visualization plots"""        
        
        self.ax_fitness.clear()
        self.ax_fitness.plot(self.logger.best_fitness_history, 'r-', label='Best')
        self.ax_fitness.plot(self.logger.mean_fitness_history, 'b-', label='Mean')
        self.ax_fitness.fill_between(range(len(self.logger.mean_fitness_history)),
                                    np.array(self.logger.mean_fitness_history) - np.array(self.logger.diversity_history),
                                    np.array(self.logger.mean_fitness_history) + np.array(self.logger.diversity_history),
                                    alpha=0.3)
        self.ax_fitness.legend()
        self.ax_fitness.set_xlabel('Generation')
        self.ax_fitness.set_ylabel('Fitness')
        self.ax_fitness.grid(True)
        
        # Population distribution
        self.ax_dist.clear()
        self.ax_dist.hist(fitnesses, bins=20, edgecolor='black', alpha=0.7)
        self.ax_dist.axvline(x=np.mean(fitnesses), color='r', linestyle='--', label=f'Mean: {np.mean(fitnesses):.1f}')
        self.ax_dist.axvline(x=max(fitnesses), color='g', linestyle='--', label=f'Best: {max(fitnesses):.1f}')
        self.ax_dist.legend()
        self.ax_dist.set_xlabel('Fitness')
        self.ax_dist.set_ylabel('Count')
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def train(self):
        """Main training loop"""
        print(f"Starting training with {self.config.population_size} rockets for {self.config.generations} generations")
        print(f"Network type: {'LSTM' if self.use_lstm else 'Feedforward'}")
        print("-" * 50)
        
        for gen in range(self.config.generations):
            start_time = time.time()
            
            # Evaluate population
            fitnesses, envs, altitudes = self.evaluate_population(visualize_best=(gen % 5 == 0))

            # Evolve population
            self.evolve_population(fitnesses)

            # Update visualizations
            self.update_plots(fitnesses)
                       
            
            # Print statistics
            elapsed = time.time() - start_time
            print(f"Generation {gen+1}/{self.config.generations} | "
                  f"Best: {max(fitnesses):.2f} | "
                  f"Mean: {np.mean(fitnesses):.2f} | "
                  f"Std: {np.std(fitnesses):.2f} | "
                  f"Time: {elapsed:.2f}s | "
                  f"Max Altitude: {max(altitudes):.2f}m")
        
        print("\nTraining complete!")
        print(f"Best fitness achieved: {self.best_fitness_ever:.2f}")
        
        # Export data
        self.logger.export_data(f"training_data_lstm_{int(time.time())}.json")
        
        return self.best_network

    def test_best_network(self, episodes: int = 5):
        """Test the best network multiple times and collect statistics"""
        if self.best_network is None:
            print("No best network found. Train first!")
            return
        
        print(f"\nTesting best network on {episodes} episodes...")
        
        results = []
        for episode in range(episodes):
            env = Environment()
            obs = env.reset()
            done = False
            
            # Reset LSTM state
            self.best_network.reset_state()
            
            while not done:
                action = self.best_network.forward(obs)
                obs, reward, done, info = env.step(action)
            
            fitness = env.get_fitness()
            results.append({
                'fitness': fitness,
                'max_altitude': env.rocket.max_altitude,
                'flight_time': env.time_step * env.dt,
                'total_distance': env.rocket.total_distance
            })
            
            print(f"Episode {episode+1}: Fitness={fitness:.2f}, "
                  f"Max Alt={env.rocket.max_altitude:.2f}m, "
                  f"Time={env.time_step * env.dt:.2f}s")
        
        # Calculate statistics
        mean_fitness = np.mean([r['fitness'] for r in results])
        std_fitness = np.std([r['fitness'] for r in results])
        print(f"\nTest Results: {mean_fitness:.2f} ± {std_fitness:.2f}")
        
        return results
    
# Add this new class after the Trainer class, before main()



# =================================================================
# ==============================================================================

def main():
    """Main entry point"""
    """
    rocket = Rocket()
    rocket.test_control_authority()
    """
    # Configure training
    training_config = TrainingConfig()
    
    # Create trainer
    trainer = Trainer(training_config)

    print(f"\n{'='*50}")
    print(f"{'='*50}")

    if training_config.save_network:
        best_network = trainer.train()
        
        # Save the best LSTM network weights for use on microcontroller
        network_data = {
            "network_type": "LSTM",
            "input_size": best_network.config.input_size,
            "hidden_size": best_network.config.hidden_size,
            "output_size": best_network.config.output_size,
            "weights": best_network.get_weights_dict(),
            "complementary_filter_alpha": 0.98,  # For microcontroller implementation
            "info": {
                "observation": "gyro(3) + accel(3) + orientation(2) + winglets(4)",
                "output": "winglets(4)",
                "note": "Microcontroller must implement complementary filter for orientation estimation"
            }
        }
        
        # Save as JSON for easy loading on microcontrollers
        with open(f"best_lstm_network_.json", "w") as f:
            json.dump(network_data, f, indent=2)
        
        print(f"LSTM network saved to best_lstm_network.json")
            
    # Test the best network
    trainer.test_best_network(episodes=10)
    
    # Keep plots open
    plt.ioff()
    plt.show()
    
    print("\nTraining complete! Data exported to JSON files for analysis.")
    print("\nNOTE: Your microcontroller must implement:")
    print("  1. Complementary filter (98% gyro, 2% accel)")
    print("  2. LSTM forward pass with hidden state management")
    print("  3. State reset at episode boundaries")
    

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from mpl_toolkits.mplot3d import Axes3D
from datetime import date
import os

RESULTS_DIR = 'results'
EPISODES = 200
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
            orientation=np.array(ORIENTATION),
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

    def calculate_reward(self, state, target_altitude=100.0):
        """
        Simple reward function:
        - Positive reward for gaining altitude
        - Penalty for excessive tilt (instability)
        - Bonus for reaching target altitude
        """
        altitude_reward = state.position[2] * 0.1  # Reward for altitude
        
        # Penalty for tilt (roll and pitch angles)
        tilt_penalty = -(abs(state.orientation[0]) + abs(state.orientation[1])) * 10
        
        # Penalty for lateral drift
        drift_penalty = -(state.position[0]**2 + state.position[1]**2) * 0.01
        
        # Bonus for reaching target altitude
        altitude_bonus = 0
        if state.position[2] >= target_altitude:
            altitude_bonus = 50
        
        # Penalty for crashing (negative altitude)
        crash_penalty = 0
        if state.position[2] < 0:
            crash_penalty = -100
            
        return altitude_reward + tilt_penalty + drift_penalty + altitude_bonus + crash_penalty

# === RL Controller ===
class RLRocketController:
    def __init__(self, learning_rate=0.001):
        self.model = self._build_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
    def _build_model(self):
        """Policy network that outputs action means and log std"""
        inputs = keras.layers.Input(shape=(10,))
        hidden1 = keras.layers.Dense(32, activation='relu')(inputs)
        hidden2 = keras.layers.Dense(16, activation='relu')(hidden1)
        
        # Output means for 4 actions (winglet positions)
        action_means = keras.layers.Dense(4, activation='tanh', name='action_means')(hidden2)
        
        # Output log standard deviations (learnable)
        log_stds = keras.layers.Dense(4, activation='tanh', name='log_stds')(hidden2)
        
        model = keras.Model(inputs=inputs, outputs=[action_means, log_stds])
        return model

    def prepare_input(self, sensors, current_winglets):
        return np.concatenate([
            sensors['accelerometer'],
            sensors['gyroscope'], 
            current_winglets
        ])

    def get_action(self, state_input):
        """Sample action from policy"""
        state_input = tf.expand_dims(state_input, 0)
        action_means, log_stds = self.model(state_input)
        
        # Convert to numpy
        action_means = action_means.numpy()[0]
        log_stds = log_stds.numpy()[0]
        
        # Sample from normal distribution
        stds = np.exp(log_stds * 0.5)  # Scale down std for stability
        actions = np.random.normal(action_means, stds)
        
        # Clip to valid range
        actions = np.clip(actions, -1, 1)
        
        return actions, action_means, log_stds

    def compute_log_prob(self, actions, action_means, log_stds):
        """Compute log probability of actions under current policy"""
        stds = tf.exp(log_stds * 0.5)
        log_prob = -0.5 * tf.reduce_sum(
            tf.square((actions - action_means) / stds) + 
            2 * log_stds * 0.5 + 
            tf.math.log(2 * np.pi), axis=1
        )
        return log_prob

    def train_step(self, states, actions, rewards):
        """REINFORCE training step"""
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        
        # Normalize rewards
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        rewards = (rewards - tf.reduce_mean(rewards)) / (tf.math.reduce_std(rewards) + 1e-8)
        
        with tf.GradientTape() as tape:
            action_means, log_stds = self.model(states)
            log_probs = self.compute_log_prob(actions, action_means, log_stds)
            
            # REINFORCE loss: -log_prob * reward
            loss = -tf.reduce_mean(log_probs * rewards)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss.numpy()

    def train(self, episodes=EPISODES):
        """Train using REINFORCE algorithm"""
        simulator = RocketSimulator()
        episode_rewards = []
        losses = []
        
        for episode in range(episodes):
            # Collect episode data
            states, actions, rewards = [], [], []
            state = simulator.reset()
            episode_reward = 0
            
            while simulator.time < 3.0:
                # Get current observation
                _, sensors = simulator.step(state, state.winglet_positions)
                state_input = self.prepare_input(sensors, state.winglet_positions)
                
                # Get action from policy
                action, action_means, log_stds = self.get_action(state_input)
                
                # Execute action
                new_state, _ = simulator.step(state, action)
                
                # Calculate reward
                reward = simulator.calculate_reward(new_state)
                
                # Store experience
                states.append(state_input)
                actions.append(action)
                rewards.append(reward)
                
                episode_reward += reward
                state = new_state
                
                # Early termination if crashed
                if new_state.position[2] < 0:
                    break
            
            # Train on episode
            if len(states) > 0:
                loss = self.train_step(np.array(states), np.array(actions), np.array(rewards))
                losses.append(loss)
            
            episode_rewards.append(episode_reward)
            
            if episode % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
        
        return episode_rewards, losses

# === Main ===
def main():
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    controller = RLRocketController()
    print("Training RL controller...")
    episode_rewards, losses = controller.train(episodes=EPISODES)

    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Training Progress")
    
    plt.subplot(1, 3, 2)
    # Moving average of rewards
    window_size = 20
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg)
        plt.xlabel("Episode")
        plt.ylabel("Moving Average Reward")
        plt.title(f"Moving Average ({window_size} episodes)")

    plt.subplot(1, 3, 3)
    if losses:
        plt.plot(losses)
        plt.xlabel("Episode")
        plt.ylabel("Policy Loss")
        plt.title("Training Loss")
    
    plt.tight_layout()
    plt.show()

    # Test flight with trained policy
    print("Running test flight...")
    sim = RocketSimulator()
    state = sim.reset()
    trajectory = []
    winglet_positions_log = []
    time_log = []
    rewards_log = []

    while sim.time < 3.0:
        _, sensors = sim.step(state, state.winglet_positions)
        state_input = controller.prepare_input(sensors, state.winglet_positions)
        
        # Use mean action (no noise) for testing
        action_means, _ = controller.model(tf.expand_dims(state_input, 0))
        action = np.clip(action_means.numpy()[0], -1, 1)
        
        state, _ = sim.step(state, action)
        reward = sim.calculate_reward(state)
        
        trajectory.append(state.position.copy())
        winglet_positions_log.append(state.winglet_positions.copy())
        time_log.append(sim.time)
        rewards_log.append(reward)
        
        if state.position[2] < 0:  # Crashed
            break

    # Plot test flight results
    trajectory = np.array(trajectory)
    
    plt.figure(figsize=(15, 10))
    
    # 3D trajectory
    ax1 = plt.subplot(2, 3, 1, projection='3d')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    
    # Altitude vs time
    plt.subplot(2, 3, 2)
    plt.plot(time_log, trajectory[:, 2])
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude vs Time')
    plt.grid(True)
    
    # Winglet positions
    plt.subplot(2, 3, 3)
    winglet_positions_log = np.array(winglet_positions_log)
    for i in range(4):
        plt.plot(time_log, winglet_positions_log[:, i], label=f'Winglet {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Winglet Position')
    plt.title('Winglet Control')
    plt.legend()
    plt.grid(True)
    
    # Lateral position
    plt.subplot(2, 3, 4)
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Lateral Trajectory')
    plt.axis('equal')
    plt.grid(True)
    
    # Rewards over time
    plt.subplot(2, 3, 5)
    plt.plot(time_log, rewards_log)
    plt.xlabel('Time (s)')
    plt.ylabel('Reward')
    plt.title('Rewards During Flight')
    plt.grid(True)
    
    # Final stats
    plt.subplot(2, 3, 6)
    max_altitude = np.max(trajectory[:, 2])
    final_altitude = trajectory[-1, 2]
    total_reward = np.sum(rewards_log)
    
    stats_text = f"""Test Flight Results:
Max Altitude: {max_altitude:.1f} m
Final Altitude: {final_altitude:.1f} m
Flight Time: {time_log[-1]:.1f} s
Total Reward: {total_reward:.1f}"""
    
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    plt.axis('off')
    plt.title('Flight Statistics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'rl_results_{date.today().strftime("%Y%m%d")}.png'), dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
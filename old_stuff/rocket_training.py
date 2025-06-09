from tkinter.tix import MAX
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym
from gym import spaces
import random
import tensorflow_probability as tfp
import os

class RocketEnv(gym.Env):
    """Custom Environment for rocket flight simulation"""
    
    def __init__(self):
        super(RocketEnv, self).__init__()
        
        # Define action and observation space
        # Actions: 4 winglet positions (each -1 to 1, representing angle)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        # Observations: [x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        # Rocket parameters
        self.mass = 0.5  # kg
        self.length = 0.5  # m
        self.diameter = 0.05  # m
        self.thrust_curve = self._generate_thrust_curve()
        self.thrust_time = 0
        self.max_thrust_time = len(self.thrust_curve) - 1
        
        # Simulation parameters
        self.dt = 0.01  # time step in seconds
        self.g = 9.81  # m/s^2
        self.air_density = 1.225  # kg/m^3
        
        # Initial state
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state"""
        # Initial position [x, y, z] in meters - start above ground
        self.position = np.array([0, 0, 0.1], dtype=np.float64)  # 10cm above ground
        
        # Initial velocity [vx, vy, vz] in m/s
        self.velocity = np.array([0, 0, 0], dtype=np.float64)
        
        # Initial orientation [roll, pitch, yaw] in radians
        self.orientation = np.array([0, 0, 0], dtype=np.float64) + np.random.normal(0, 0.01, 3)
        
        # Initial angular velocity [roll_rate, pitch_rate, yaw_rate] in rad/s
        self.angular_velocity = np.array([0, 0, 0], dtype=np.float64)
        
        # Reset thrust counter
        self.thrust_time = 0
        
        # History for plotting
        self.history = {
            'position': [self.position.copy()],
            'velocity': [self.velocity.copy()],
            'orientation': [self.orientation.copy()],
            'actions': []
        }
        
        return self._get_observation()
    
    def _generate_thrust_curve(self):
        """Generate a simple thrust curve for a solid rocket motor"""
        # Simulating an Estes C6-5 motor (rough approximation)
        # Total impulse: 10 N·s, max thrust: 6 N, burn time: 2 seconds
        
        TIME = 200  # 3 seconds at dt=0.01
        MAX_THRUST = 15
        
        # Quick rise to max thrust
        rise = np.linspace(0, MAX_THRUST, 10)
        
        # Plateau at max thrust
        plateau = MAX_THRUST * np.ones(140)
        
        # Decay to zero
        decay = np.linspace(MAX_THRUST, 0, 50)
        
        return np.concatenate([rise, plateau, decay])
    
    def step(self, action):
        """Run one timestep of the environment's dynamics with the given action"""
        # Limit actions to valid range
        action = np.clip(action, -1, 1)
        self.history['actions'].append(action.copy())
        
        # Get current thrust
        if self.thrust_time < self.max_thrust_time:
            thrust = self.thrust_curve[self.thrust_time]
            self.thrust_time += 1
        else:
            thrust = 0
        
        # Calculate forces and moments from winglets
        winglet_forces, winglet_moments = self._calculate_winglet_effects(action)
        
        # Calculate gravity force
        gravity_force = np.array([0, 0, -self.mass * self.g])
        
        # Calculate thrust force (assuming thrust is along the rocket's z-axis)
        # Transform from rocket frame to world frame
        r_matrix = self._rotation_matrix()
        thrust_force = r_matrix @ np.array([0, 0, thrust])
        
        # Calculate aerodynamic drag
        drag_force = self._calculate_drag()
        
        # Total force
        total_force = thrust_force + gravity_force + drag_force + winglet_forces
        
        # Update velocity and position
        self.velocity += (total_force / self.mass) * self.dt
        self.position += self.velocity * self.dt
        
        # Update angular velocity and orientation
        inertia_tensor = self._inertia_tensor()
        self.angular_velocity += np.linalg.inv(inertia_tensor) @ winglet_moments * self.dt
        self.orientation += self.angular_velocity * self.dt
        
        # Normalize angles to [-π, π]
        self.orientation = np.mod(self.orientation + np.pi, 2 * np.pi) - np.pi
        
        # Update history
        self.history['position'].append(self.position.copy())
        self.history['velocity'].append(self.velocity.copy())
        self.history['orientation'].append(self.orientation.copy())
        
        # Check if done (hit ground, out of bounds, or timeout)
        done = (self.position[2] < 0 or  # Hit ground
                np.linalg.norm(self.position[:2]) > 500 or  # Too far horizontally
                self.position[2] > 1000 or  # Too high
                self.thrust_time > self.max_thrust_time + 500)  # Timeout after thrust + 5s
        
        # Calculate reward
        reward = self._calculate_reward()
        
        return self._get_observation(), reward, done, {}
    
    def _calculate_reward(self):
        """Calculate the reward based on rocket stability and trajectory"""
        # Reward for maintaining vertical orientation
        up_vector = self._rotation_matrix() @ np.array([0, 0, 1])
        alignment_reward = np.dot(up_vector, np.array([0, 0, 1]))  # 1 when perfectly aligned
        
        # Penalize angular velocity (scaled by angular velocity magnitude)
        ang_vel_mag = np.linalg.norm(self.angular_velocity)
        stability_penalty = -0.1 * ang_vel_mag
        
        # Reward vertical speed while penalizing horizontal movement
        vertical_reward = 0.5 * self.velocity[2] if self.velocity[2] > 0 else 0
        horizontal_penalty = -0.2 * np.linalg.norm(self.velocity[:2])
        
        # Terminal rewards/penalties
        ground_penalty = -50 if self.position[2] < 0 else 0
        
        return alignment_reward + stability_penalty + vertical_reward + horizontal_penalty + ground_penalty
    
    def _get_observation(self):
        """Convert current state to observation"""
        return np.concatenate([
            self.position,
            self.velocity,
            self.orientation,
            self.angular_velocity
        ])
    
    def _rotation_matrix(self):
        """Create rotation matrix from current orientation"""
        roll, pitch, yaw = self.orientation
        
        # Roll (x-axis rotation)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Pitch (y-axis rotation)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Yaw (z-axis rotation)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix (applying yaw, then pitch, then roll)
        return Rx @ Ry @ Rz
    
    def _inertia_tensor(self):
        """Calculate the inertia tensor of the rocket"""
        # Approximating the rocket as a solid cylinder
        m = self.mass
        l = self.length
        r = self.diameter / 2
        
        # Moment of inertia for a cylinder
        Ixx = Iyy = (1/12) * m * (3*r**2 + l**2)
        Izz = (1/2) * m * r**2
        
        return np.diag([Ixx, Iyy, Izz])
    
    def _calculate_drag(self):
        """Calculate aerodynamic drag force"""
        v_mag = np.linalg.norm(self.velocity)
        if v_mag < 1e-6:
            return np.zeros(3)
        
        # Drag coefficient (simplified)
        Cd = 0.5
        
        # Reference area (frontal area)
        A = np.pi * (self.diameter/2)**2
        
        # Drag magnitude
        drag_mag = 0.5 * self.air_density * v_mag**2 * Cd * A
        
        # Drag direction (opposite to velocity)
        drag_dir = -self.velocity / v_mag
        
        return drag_mag * drag_dir
    
    def _calculate_winglet_effects(self, actions):
        """Calculate forces and moments from winglet deflections"""
        # This is a simplified model - real aerodynamics would be more complex
        
        # Convert actions to winglet angles (in radians, limited to ±15 degrees)
        max_angle = np.radians(15)
        winglet_angles = actions * max_angle
        
        # Simplified force model: forces proportional to dynamic pressure and angle
        q = 0.5 * self.air_density * np.sum(np.square(self.velocity))

        total_force = np.zeros(3)
        total_moment = np.zeros(3)
        
        # Each winglet contributes force based on its angle
        # Assuming winglets are at 90° apart around the rocket
        winglet_positions = [
            np.array([1, 0, 0]),  # winglet 1 pointing +x
            np.array([0, 1, 0]),  # winglet 2 pointing +y
            np.array([-1, 0, 0]), # winglet 3 pointing -x
            np.array([0, -1, 0])  # winglet 4 pointing -y
        ]
        
        for i, (angle, pos) in enumerate(zip(winglet_angles, winglet_positions)):
            # Force perpendicular to rocket body and winglet position
            # Cross product to get direction perpendicular to both rocket axis and winglet position
            force_dir = np.cross(np.array([0, 0, 1]), pos)
            
            # Force magnitude proportional to angle and dynamic pressure
            force_mag = 0.1 * q * angle  # 0.1 is a simplified coefficient
            
            # Calculate force in rocket frame
            force = force_mag * force_dir
            
            # Transform to world frame
            force_world = self._rotation_matrix() @ force
            total_force += force_world
            
            # Calculate moment (torque) - simplified
            # Moment arm: distance from CG to winglet (approx. at 80% of rocket length)
            moment_arm = 0.4 * self.length
            
            # Moment = position × force (cross product)
            moment = np.cross(moment_arm * np.array([0, 0, 1]), force)
            
            # Transform to world frame
            moment_world = self._rotation_matrix() @ moment
            total_moment += moment_world
        
        return total_force, total_moment
    
    def render(self, mode='human'):
        """Render the environment to the screen"""
        # Convert history to numpy arrays for easier plotting
        positions = np.array(self.history['position'])
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        
        # Trajectory plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Rocket Trajectory')
        
        # Set equal aspect ratio
        max_range = max([
            np.max(positions[:, 0]) - np.min(positions[:, 0]),
            np.max(positions[:, 1]) - np.min(positions[:, 1]),
            np.max(positions[:, 2]) - np.min(positions[:, 2])
        ])
        mid_x = (np.max(positions[:, 0]) + np.min(positions[:, 0])) * 0.5
        mid_y = (np.max(positions[:, 1]) + np.min(positions[:, 1])) * 0.5
        mid_z = (np.max(positions[:, 2]) + np.min(positions[:, 2])) * 0.5
        ax1.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax1.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax1.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # Orientation and actions plot
        if len(self.history['actions']) > 0:
            ax2 = fig.add_subplot(122)
            actions = np.array(self.history['actions'])
            orientations = np.array(self.history['orientation'])
            
            times = np.arange(len(actions)) * self.dt
            
            ax2.plot(times, actions[:, 0], 'r-', label='Winglet 1')
            ax2.plot(times, actions[:, 1], 'g-', label='Winglet 2')
            ax2.plot(times, actions[:, 2], 'b-', label='Winglet 3')
            ax2.plot(times, actions[:, 3], 'y-', label='Winglet 4')
            
            # Add orientation lines
            ax2.plot(times, np.rad2deg(orientations[:len(actions), 0]), 'r--', label='Roll (deg)')
            ax2.plot(times, np.rad2deg(orientations[:len(actions), 1]), 'g--', label='Pitch (deg)')
            ax2.plot(times, np.rad2deg(orientations[:len(actions), 2]), 'b--', label='Yaw (deg)')
            
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Values')
            ax2.set_title('Winglet Actions & Orientation')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def close(self):
        plt.close('all')

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.size = 0
        self.position = 0
        
        # For state normalization
        self.state_mean = np.zeros(state_dim, dtype=np.float32)
        self.state_std = np.ones(state_dim, dtype=np.float32)
        self.normalization_steps = 0
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Update normalization statistics
        if self.size < self.capacity:  # Only during buffer filling
            self._update_normalization(state)
            self._update_normalization(next_state)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def _update_normalization(self, state):
        """Update running mean and std for state normalization"""
        self.normalization_steps += 1
        # Incremental update of mean and variance (Welford's algorithm)
        delta = state - self.state_mean
        self.state_mean += delta / self.normalization_steps
        delta2 = state - self.state_mean
        # Ensure argument is non-negative before sqrt
        variance = np.maximum(
            (self.normalization_steps - 1) / self.normalization_steps * 
            self.state_std**2 + delta * delta2 / self.normalization_steps,
            0.0
        )
        self.state_std = np.sqrt(variance)
        # Prevent division by zero
        self.state_std = np.maximum(self.state_std, 1e-6)
    
    def normalize_state(self, state):
        """Normalize state using current statistics"""
        return (state - self.state_mean) / self.state_std
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        states = self.normalize_state(self.states[indices])
        next_states = self.normalize_state(self.next_states[indices])
        
        return (
            states,
            self.actions[indices],
            self.rewards[indices],
            next_states,
            self.dones[indices]
        )

class SAC:
    def __init__(
        self, 
        state_dim=12, 
        action_dim=4,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=100000,
        batch_size=256
    ):
        # Initialize dimensions and hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # discount factor
        self.tau = tau      # soft target update factor
        self.batch_size = batch_size
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)
        
        # Initialize actor network (policy)
        self.actor = self._create_actor_network()
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        
        # Initialize critic networks (Q-functions) and targets
        self.critic_1 = self._create_critic_network()
        self.critic_2 = self._create_critic_network()
        self.target_critic_1 = self._create_critic_network()
        self.target_critic_2 = self._create_critic_network()
        
        # Copy weights to target networks
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())
        
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(critic_lr)
        
        # Initialize temperature parameter alpha (for entropy maximization)
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)
        self.target_entropy = -action_dim  # heuristic
        self.alpha_optimizer = tf.keras.optimizers.Adam(alpha_lr)

    def _create_actor_network(self):
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        
        # Output mean and log_std for each action dimension
        mean = tf.keras.layers.Dense(self.action_dim, activation=None)(x)
        log_std = tf.keras.layers.Dense(self.action_dim, activation=None)(x)
        
        # Constrain log_std to prevent numerical instability
        log_std = tf.keras.layers.Lambda(
            lambda x: tf.clip_by_value(x, -20, 2)
        )(log_std)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=[mean, log_std])
        return model
    
    def _create_critic_network(self):
        # State input
        state_input = tf.keras.layers.Input(shape=(self.state_dim,))
        # Action input
        action_input = tf.keras.layers.Input(shape=(self.action_dim,))
        
        # Concatenate state and action
        concat = tf.keras.layers.Concatenate()([state_input, action_input])
        
        # Hidden layers
        x = tf.keras.layers.Dense(256, activation='relu')(concat)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        
        # Q-value output
        q_value = tf.keras.layers.Dense(1)(x)
        
        # Create model
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=q_value)
        return model
    
    def sample_action(self, state, evaluate=False):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        mean, log_std = self.actor(state)
        
        if evaluate:
            # During evaluation, use the mean action (no exploration)
            return tf.tanh(mean[0])
        
        # During training, sample from the distribution
        std = tf.exp(log_std)
        normal_dist = tfp.distributions.Normal(mean, std)
        z = normal_dist.sample()
        
        # Apply tanh to bound actions to [-1, 1]
        action = tf.tanh(z[0])
        
        # Calculate log probability for training
        log_prob = normal_dist.log_prob(z)
        
        # Apply correction for tanh squashing
        log_prob -= tf.reduce_sum(tf.math.log(1 - tf.square(action) + 1e-6), axis=-1)
        
        return action.numpy(), log_prob.numpy()
    
    def update(self, training_step):
        # Sample batch from replay buffer
        if self.buffer.size < self.batch_size:
            return {}  # Not enough data
            
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Update critic networks
        critic1_loss, critic2_loss = self._update_critics(
            states, actions, rewards, next_states, dones
        )
        
        # Update actor network and temperature parameter less frequently
        if training_step % 2 == 0:
            actor_loss = self._update_actor(states)
            alpha_loss = self._update_alpha(states)
        else:
            actor_loss = None
            alpha_loss = None
        
        # Soft update target networks
        self._update_targets()
        
        return {
            'critic1_loss': critic1_loss,
            'critic2_loss': critic2_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha.numpy()
        }
    
    @tf.function
    def _update_critics(self, states, actions, rewards, next_states, dones):
        # Get next actions and their log probs from current policy
        next_means, next_log_stds = self.actor(next_states)
        next_stds = tf.exp(next_log_stds)
        next_normal = tfp.distributions.Normal(next_means, next_stds)
        next_actions_raw = next_normal.sample()
        next_actions = tf.tanh(next_actions_raw)
        
        # Calculate log probabilities - Fix dimension mismatch
        next_log_probs = next_normal.log_prob(next_actions_raw)
        # Sum across action dimensions to get a single log prob per sample
        next_log_probs = tf.reduce_sum(next_log_probs, axis=1, keepdims=True)
        # Apply tanh squashing correction
        next_log_probs -= tf.reduce_sum(
            tf.math.log(1 - tf.square(next_actions) + 1e-6), 
            axis=1, 
            keepdims=True
        )
        
        # Rest of the function remains the same
        target_q1 = self.target_critic_1([next_states, next_actions])
        target_q2 = self.target_critic_2([next_states, next_actions])
        target_q = tf.minimum(target_q1, target_q2) - self.alpha * next_log_probs
        
        td_targets = rewards + (1 - dones) * self.gamma * target_q
        
        with tf.GradientTape() as tape1:
            q1 = self.critic_1([states, actions])
            critic1_loss = tf.reduce_mean(tf.square(q1 - td_targets))
            
        critic1_gradients = tape1.gradient(critic1_loss, self.critic_1.trainable_variables)
        self.critic_optimizer_1.apply_gradients(
            zip(critic1_gradients, self.critic_1.trainable_variables)
        )
        
        with tf.GradientTape() as tape2:
            q2 = self.critic_2([states, actions])
            critic2_loss = tf.reduce_mean(tf.square(q2 - td_targets))
            
        critic2_gradients = tape2.gradient(critic2_loss, self.critic_2.trainable_variables)
        self.critic_optimizer_2.apply_gradients(
            zip(critic2_gradients, self.critic_2.trainable_variables)
        )
        
        return critic1_loss, critic2_loss
    
    @tf.function
    def _update_actor(self, states):
        with tf.GradientTape() as tape:
            # Get actions from current policy
            means, log_stds = self.actor(states)
            stds = tf.exp(log_stds)
            normal = tfp.distributions.Normal(means, stds)
            actions_raw = normal.sample()
            actions = tf.tanh(actions_raw)
            
            # Calculate log probabilities and apply correction with keepdims=True
            log_probs = normal.log_prob(actions_raw)
            log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)
            correction = tf.reduce_sum(tf.math.log(1 - tf.square(actions) + 1e-6), axis=1, keepdims=True)
            log_probs -= correction
            
            # Calculate Q-values
            q1 = self.critic_1([states, actions])
            q2 = self.critic_2([states, actions])
            q = tf.minimum(q1, q2)
            
            # Actor loss: maximize Q-value while maintaining entropy
            actor_loss = tf.reduce_mean(self.alpha * log_probs - q)
            
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        
        # Clip gradients to avoid exploding gradients
        actor_gradients, _ = tf.clip_by_global_norm(actor_gradients, 1.0)
        
        self.actor_optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables)
        )
        
        return actor_loss
    
    @tf.function
    def _update_alpha(self, states):
        means, log_stds = self.actor(states)
        stds = tf.exp(log_stds)
        normal = tfp.distributions.Normal(means, stds)
        actions_raw = normal.sample()
        actions = tf.tanh(actions_raw)
        
        log_probs = normal.log_prob(actions_raw)
        log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)
        correction = tf.reduce_sum(tf.math.log(1 - tf.square(actions) + 1e-6), axis=1, keepdims=True)
        log_probs -= correction
        
        with tf.GradientTape() as tape:
            alpha_loss = -tf.reduce_mean(
                self.log_alpha * (log_probs + self.target_entropy)
            )
        
        alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_alpha]))
        
        return alpha_loss
    
    def _update_targets(self):
        """Soft update of target critic networks"""
        critic1_weights = self.critic_1.get_weights()
        target_critic1_weights = self.target_critic_1.get_weights()
        
        critic2_weights = self.critic_2.get_weights()
        target_critic2_weights = self.target_critic_2.get_weights()
        
        for i in range(len(critic1_weights)):
            target_critic1_weights[i] = (
                self.tau * critic1_weights[i] + (1 - self.tau) * target_critic1_weights[i]
            )
            target_critic2_weights[i] = (
                self.tau * critic2_weights[i] + (1 - self.tau) * target_critic2_weights[i]
            )
            
        self.target_critic_1.set_weights(target_critic1_weights)
        self.target_critic_2.set_weights(target_critic2_weights)
        
    def save_models(self, path):
        """Save all models to disk"""
        os.makedirs(path, exist_ok=True)
        self.actor.save(f"{path}/actor.keras")
        self.critic_1.save(f"{path}/critic_1.keras")
        self.critic_2.save(f"{path}/critic_2.keras")
        
    def load_models(self, path):
        """Load all models from disk"""
        self.actor = tf.keras.models.load_model(f"{path}/actor")
        self.critic_1 = tf.keras.models.load_model(f"{path}/critic_1")
        self.critic_2 = tf.keras.models.load_model(f"{path}/critic_2")
        self.target_critic_1 = tf.keras.models.load_model(f"{path}/critic_1")
        self.target_critic_2 = tf.keras.models.load_model(f"{path}/critic_2")

def train_sac(episodes=1000, render_every=100, eval_every=20):
    env = RocketEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SAC(state_dim=state_dim, action_dim=action_dim)
    
    # Track metrics
    episode_rewards = []
    evaluation_rewards = []
    best_eval_reward = -np.inf
    training_step = 0
    
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action with exploration
            action, _ = agent.sample_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition in replay buffer
            agent.buffer.add(state, action, reward, next_state, done)
            
            # Update agent
            if agent.buffer.size >= agent.batch_size:
                metrics = agent.update(training_step)
                training_step += 1
            
            state = next_state
            episode_reward += reward
        
        # Log reward
        episode_rewards.append(episode_reward)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
            
            # Also print alpha value and other metrics if available
            if 'alpha' in metrics:
                print(f"Alpha: {metrics['alpha']:.4f}")
        
        # Evaluate agent performance without exploration
        if episode % eval_every == 0:
            eval_reward = evaluate_agent(env, agent)
            evaluation_rewards.append(eval_reward)
            print(f"Evaluation at episode {episode}: {eval_reward:.2f}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save_models("best_model")
                print(f"New best model saved with reward {eval_reward:.2f}")
        
        # Render occasionally
        if episode % render_every == 0:
            print(f"Rendering episode {episode}")
            render_agent(env, agent)
    
    # Final evaluation and saving
    agent.save_models("final_model")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, episodes, eval_every), evaluation_rewards)
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    return agent

def evaluate_agent(env, agent, n_episodes=5):
    """Evaluate agent performance without exploration"""
    total_rewards = []
    
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.sample_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
        
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards)

def render_agent(env, agent):
    """Render an episode with the current agent policy"""
    state = env.reset()
    done = False
    
    while not done:
        action = agent.sample_action(state, evaluate=True)
        next_state, _, done, _ = env.step(action)
        state = next_state
    
    env.render()

# Main function to run the training process
def main():
    # Train the model
    print("Training neural network for rocket control...")
    agent = train_sac(episodes=500, render_every=501, eval_every=20)
    
    # Test the trained model
    print("Testing trained model...")
    env = RocketEnv()
    obs = env.reset()
    done = False
    
    while not done:
        action = agent.sample_action(obs, evaluate=True)
        obs, reward, done, _ = env.step(action)
    
    env.render()
    env.close()

if __name__ == "__main__":
    main()
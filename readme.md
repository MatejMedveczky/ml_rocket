# Neural Network Rocket Control via Particle Swarm Optimization

A framework for training neural network controllers for gimbal-stabilized model rockets using evolutionary optimization. This project explores whether evolutionary algorithms can discover effective control policies for thrust vector control systems, an alternative approach to classical control theory that trades analytical derivation for computational search.

## Motivation

Traditional model rocket stabilization relies on carefully tuned PID controllers or model-based approaches requiring precise system identification. But what if we could skip the mathematics and let evolution discover the control law directly? This project investigates that question using Particle Swarm Optimization to train neural networks that stabilize vertical flight using only IMU sensor feedback.

## Technical Approach

### Physics Simulation

The core simulator implements 6-DOF rigid body dynamics with realistic aerodynamic forces, thrust curves from TSP D-12 rocket motors (sourced from OpenRocket), and sensor noise modeling. The physics were validated against OpenRocket simulations where they match closely, giving confidence that the learned controllers could transfer to hardware.

### Neural Network Architecture

The controller is a compact feedforward network (8→8→5→2 neurons) designed to run on embedded flight computers. It processes gyroscope and accelerometer data to generate gimbal deflection commands, with tanh output activation ensuring physically realizable control signals within the ±7.5° mechanical limits.

### Particle Swarm Optimization

Rather than gradient-based reinforcement learning, I use PSO to evolve network weights directly. A swarm of 30 particles (each representing complete network parameters) explores the 115-dimensional parameter space, with particles sharing information about successful control strategies. Over 50 generations of simulated flights, the swarm converges toward networks that maximize altitude while maintaining vertical stability.

The fitness function rewards altitude gain, penalizes tilt and angular rates, and includes shaped rewards for stable vertical flight.

```python
fitness = altitude_gain × 15.0 
        + survival_time × 0.05
        + stability_bonus(tilt < 0.2 rad)
        - horizontal_drift × 2.0
        - angular_rate_penalty
```

## Results

The trained networks demonstrate robust vertical flight control, actively correcting initial perturbations and maintaining stability throughout motor burn. Maximum achieved altitudes match OpenRocket predictions, and the controller generalizes well to varied initial conditions during testing.

The network weights export to JSON format for deployment on ESP32-based flight computers, maintaining the inference simplicity needed for real-time embedded control.

## Repository Structure

```bash
├── rocket_training/
│   └── rocket_pso_gimbal.py       # Training framework and physics simulation
├── rocket_models/
│   ├── best_gimbal_network.json   # Trained network weights
├── rocket_render.png              # CAD model visualization
└── openrocket_sim.ork             # Physics validation
```

## Future Directions

The natural next step is hardware validation on actual flight tests. The CAD model shows the mechanical design for the gimbal mount system—translating these trained control policies from simulation to physical hardware remains the critical validation step.

[Rocket CAD Model](rocket_body_v35.png)

Beyond flight testing, interesting questions remain: Can extended Kalman filtering improve state estimation from noisy IMU data? Could model predictive control leverage this learned dynamics model for predictive stabilization? How robust are these controllers to wind disturbances and model uncertainties?

These questions sit at the intersection of control theory, machine learning, and aerospace engineering—exactly where the interesting problems live.

---

*Note: No actual rockets were harmed in the development process.*
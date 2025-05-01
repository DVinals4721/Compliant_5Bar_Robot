# Compliant 5-Bar Robot Simulation

A Python package for modeling and simulating a compliant 5-bar linkage robot using the Direct Stiffness Method.

## Project Overview

This repository implements a simulation of a compliant 5-bar linkage robot mechanism. Unlike traditional rigid 5-bar linkages with four links and two motors, this project models a simplified version with only two flexible links mounted on motors and connected to each by a pin joint on the other ends. The flexible links bend in response to motor angles and torques, replicating the workspace and behavior of a traditional 5-bar mechanism through compliant deformation rather than rigid connections.

## Key Features

* **3D Frame Solver** using the Direct Stiffness Method, including:
   * 3D beam element formulation
   * Geometric nonlinearity consideration
   * Local and global stiffness matrix assembly
   * Boundary condition application
   * Solver for displacements and reactions
* **Pseudo-Rigid-Body Model (PRBM)** implementation based on "Handbook of Compliant Mechanisms" by Larry L. Howell, Spencer P. Magleby, and Brian M. Olsen.
* **5-Bar Mechanism Kinematics** calculated using equations from "A Method for Optimal Kinematic Design of Five-bar Planar Parallel Manipulators" by Tien Dung Le, Hee-Jun Kang, and Quang Vinh Doan.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/DVinals4721/Compliant_5Bar_Robot.git
   cd Compliant_5Bar_Robot
   ```

2. Set up a Conda environment:
   ```bash
   conda create --name robotenv python=3.11
   conda activate robotenv
   ```
   Note: You can also use mamba if you prefer.

3. Verify Python version:
   ```bash
   python --version
   ```
   Ensure it shows version 3.11 or later.

4. Update pip and essential tools:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

5. Install the package in editable mode:
   ```bash
   pip install -e .
   ```
   Make sure you're in the correct directory (Compliant_5Bar_Robot) when running this command.

6. Install pytest and pytest-cov for testing:
   ```bash
   pip install pytest pytest-cov
   ```

7. Run specific tests:
   ```bash
   pytest tests/test_frame_solver.py
   ```

## Usage Example

After installation, explore the functionality through our example script:

### Compliant Robot Simulation

```bash
python examples/compliant_robot_example.py
```

This script demonstrates how to:
* Set up a compliant 5-bar mechanism
* Apply motor angles and torques
* Solve for the resulting deformation
* Visualize the compliant robot's behavior

## Package Structure

* `src/compliant_robot/`: Contains the main implementation:
   * Frame solver using Direct Stiffness Method
   * Pseudo-Rigid-Body Model for compliant mechanism simulation
   * Kinematic analysis tools for 5-bar linkages
* `tests/`: Contains unit tests for the solver and models
* `examples/`: Contains example scripts demonstrating the usage

## Theoretical Background

This project combines structural mechanics with compliant mechanism theory to model flexible linkages. The approach uses:

1. **Pseudo-Rigid-Body Model**: Approximates flexible members as rigid links connected by torsional springs, allowing efficient simulation of large deflections.

2. **Direct Stiffness Method**: Analyzes the deformation of the compliant links under various loads and boundary conditions.

3. **5-Bar Mechanism Kinematics**: Provides the mathematical framework for calculating workspace and motion patterns.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
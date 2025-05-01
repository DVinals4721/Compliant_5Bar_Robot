"""
Five-Bar Robot Simulation Demonstration
======================================

This script demonstrates the key features of the FiveBarRobot class,
providing a comprehensive showcase of its capabilities including:
- Robot initialization with custom parameters
- Workspace exploration
- Inverse kinematics
- Position analysis
- Visualization and animation
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random
# Import the FiveBarRobot class - update path as needed
from compliant_robot import FiveBarRobot

def main():
    # Create robot with custom parameters
    robot = FiveBarRobot(
        # Material properties
        youngs_modulus=2.0e11,  # Steel
        poisson_ratio=0.3,
        yield_strength=8.0e8,

        # Beam dimensions
        beam_width=0.5,
        beam_height=0.01,

        # Geometry
        arm_length=10,
        motor_distance=1.0,
        compliant_ratio=0.85,
        stiffness_factor=2.65,

        # Limits
        max_deformation_angle=64.3,  # in degrees
        max_force_ratio=5.0,

        # Visualization
        arm_color='darkblue',
        workspace_color='lightgreen',
        marker_color='orange',
        spline1_color_range=[(0, 0.5, 1), (1, 0, 0)],  # Light blue to red
        spline2_color_range=[(0, 0.5, 1), (1, 0, 0)],  # Light blue to red

        # Display options
        show_angles=True,
        show_points=True
    )

    # Example 1: Find accessible coordinates
    # Lower resolution for faster results; increase for more accuracy
    resolution = 0.05
    robot.find_accessible_coordinates(resolution=resolution)
    print(robot.accessible_coords)
    # Example 2: Demonstrate the inverse kinematics function for a specific point
    test_point = (0.0, 9.5)  # Example point to test
    print("\nTesting inverse kinematics for point:", test_point)
    result = robot.calculate_inverse_kinematics(*test_point)
    if result:
        shoulder1, shoulder2, mm1, mm2 = result
        print(f"Shoulder angles: ({math.degrees(shoulder1):.2f}°, {math.degrees(shoulder2):.2f}°)")
        print(f"Motor torques: ({mm1:.2f} Nm, {mm2:.2f} Nm)")

        # Print detailed information about this point
        robot.print_coordinates_info(*test_point)

        # Visualize this specific point
        robot.visualize_point(*test_point)
    else:
        print("Point is not reachable")

    # Example 3: Animate all accessible coordinates
    if len(robot.accessible_coords) > 0:
        print(f"\nStarting animation with {len(robot.accessible_coords)} points...")
        # Use the new method to visit all coordinates (in random order)
        robot.animate_all_coordinates(delay=0.1, random_order=True)
    else:
        print("No accessible coordinates found. Check robot parameters.")


if __name__ == "__main__":
    main()
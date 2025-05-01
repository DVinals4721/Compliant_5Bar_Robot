import pytest
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os
import sys
from compliant_robot import FiveBarRobot
import matplotlib
matplotlib.use('Agg')

# Fixture for creating a robot instance for testing
@pytest.fixture
def robot():
    # Create a robot with default parameters
    return FiveBarRobot()


# Fixture for creating a robot with custom parameters
@pytest.fixture
def custom_robot():
    return FiveBarRobot(
        beam_width=0.6,
        beam_height=0.015,
        youngs_modulus=1.8e11,
        poisson_ratio=0.25,
        yield_strength=7.0e8,
        arm_length=8,
        motor_distance=1.5,
        max_deformation_angle=50,
        max_force_ratio=3.0
    )


# Test initialization and parameter setting
def test_initialization(robot):
    """Test that the robot initializes with correct default parameters"""
    assert robot.b == 0.5
    assert robot.h == 0.01
    assert robot.E == 2.0e11
    assert robot.nu == 0.3
    assert robot.Y == 8.0e8
    assert robot.l == 10
    assert robot.l0 == 1.0
    assert robot.ye == 0.85
    assert robot.K0 == 2.65
    assert robot.max_deformation_angle == 64.3
    assert robot.max_force_ratio == 5.0

    # Check derived parameters
    assert robot.A == robot.b * robot.h
    assert abs(robot.Iz - (robot.b * (robot.h**3) / 12)) < 1e-10
    assert abs(robot.K - robot.ye * robot.K0 * robot.E * robot.Iz / robot.l) < 1e-10
    assert abs(robot.l1 - (robot.l - robot.ye * robot.l)) < 1e-10
    assert abs(robot.l2 - (robot.ye * robot.l)) < 1e-10


# Test custom parameter initialization
def test_custom_initialization(custom_robot):
    """Test that custom parameters are correctly set"""
    assert custom_robot.b == 0.6
    assert custom_robot.h == 0.015
    assert custom_robot.E == 1.8e11
    assert custom_robot.nu == 0.25
    assert custom_robot.Y == 7.0e8
    assert custom_robot.l == 8
    assert custom_robot.l0 == 1.5
    assert custom_robot.max_deformation_angle == 50
    assert custom_robot.max_force_ratio == 3.0


# Test the reach check function
def test_in_reach(robot):
    """Test the in_reach method for different points"""
    # Points that should be in reach
    assert robot.in_reach(0, 8) == True
    assert robot.in_reach(0.5, 5) == True

    # Points that should be out of reach
    assert robot.in_reach(0, 12) == False  # Too far
    assert robot.in_reach(15, 15) == False  # Way too far


# Test angle calculations
def test_calc_all_angles(robot):
    """Test angle calculations for valid and invalid positions"""
    # Test a valid position
    angles = robot.calc_all_angles(0, 9.5)
    assert angles is not None
    assert len(angles) == 6

    # Verify the calculated angles are reasonable
    shoulder1, shoulder2, elbow1, elbow2, def_angle1, def_angle2 = angles
    assert 0 <= shoulder1 <= math.pi
    assert 0 <= shoulder2 <= math.pi
    assert 0 <= elbow1 <= math.pi
    assert 0 <= elbow2 <= math.pi

    # Test an invalid position (should return None)
    angles = robot.calc_all_angles(0, 20)
    assert angles is None


# Test statics calculations
def test_calculate_statics(robot):
    """Test static force calculations"""
    # First, get valid angles
    angles = robot.calc_all_angles(0, 8)
    assert angles is not None

    # Calculate statics
    statics = robot.calculate_statics(angles)
    assert statics is not None
    assert len(statics) == 4

    # Check that torques are reasonable
    mm1, mm2, fy, fx = statics
    assert isinstance(mm1, float)
    assert isinstance(mm2, float)
    assert isinstance(fy, float)
    assert isinstance(fx, float)


# Test stress calculations
def test_calculate_stress(robot):
    """Test stress calculations for a valid position"""
    # First, get valid angles and statics
    angles = robot.calc_all_angles(0, 8)
    assert angles is not None

    statics = robot.calculate_statics(angles)
    assert statics is not None

    # Calculate stresses
    max_stress1, max_stress2, n1, n2 = robot.calculate_stress(angles, statics)
    assert max_stress1 is not None
    assert max_stress2 is not None
    assert n1 is not None
    assert n2 is not None

    # Check that stresses are below yield strength
    assert max_stress1 < robot.Y
    assert max_stress2 < robot.Y


# Test constraint checks
def test_constraint_checks(robot):
    """Test angle and force ratio constraint checks"""
    # Test a position with valid angles
    angles = robot.calc_all_angles(0, 9.5)
    statics = robot.calculate_statics(angles)
    _, _, n1, n2 = robot.calculate_stress(angles, statics)

    # Verify constraints
    assert robot.in_angle(angles)
    assert robot.in_n(n1, n2)

    # Create invalid angles (exceeding limits)
    invalid_angles = list(angles)
    invalid_angles[4] = 1.5  # Set def_angle1 to a large value
    invalid_angles = tuple(invalid_angles)

    # Test with invalid angles
    assert not robot.in_angle(invalid_angles)


# Test spline creation
def test_create_smooth_spline(robot):
    """Test the spline creation function"""
    # Define test spline parameters
    spline_params = {
        'start_point': (-1, 0),
        'end_point': (1, 2),
        'start_angle': math.pi/4,
        'end_angle': math.pi/3,
        'curve_length': 4,
        'C': 0.5,
        'label': 'Test Spline'
    }

    # Create spline
    x_points, y_points = robot.create_smooth_spline(spline_params)

    # Verify output
    assert len(x_points) == 100  # Default number of points
    assert len(y_points) == 100
    assert x_points[0] == spline_params['start_point'][0]
    assert y_points[0] == spline_params['start_point'][1]
    assert abs(x_points[-1] - spline_params['end_point'][0]) < 1e-10
    assert abs(y_points[-1] - spline_params['end_point'][1]) < 1e-10


# Test color interpolation
def test_interpolate_color(robot):
    """Test the color interpolation function"""
    # Test with c = 0 (should return first color)
    color_range = [(0, 0, 1), (1, 0, 0)]  # Blue to Red
    color = robot.interpolate_color(0, color_range)
    assert color == (0, 0, 1)

    # Test with c = 1 (should return second color)
    color = robot.interpolate_color(1, color_range)
    assert color == (1, 0, 0)

    # Test with c = 0.5 (should return middle color)
    color = robot.interpolate_color(0.5, color_range)
    assert color == (0.5, 0, 0.5)

    # Test with c outside range (should clamp)
    color = robot.interpolate_color(2, color_range)
    assert color == (1, 0, 0)
    color = robot.interpolate_color(-1, color_range)
    assert color == (0, 0, 1)


# Test inverse kinematics
def test_calculate_inverse_kinematics(robot):
    """Test the inverse kinematics function"""
    # Test a valid point
    result = robot.calculate_inverse_kinematics(0, 9.5)
    assert result is not None
    assert len(result) == 4

    shoulder1, shoulder2, mm1, mm2 = result
    assert 0 <= shoulder1 <= math.pi
    assert 0 <= shoulder2 <= math.pi
    assert isinstance(mm1, float)
    assert isinstance(mm2, float)

    # Test an invalid point
    result = robot.calculate_inverse_kinematics(0, 20)
    assert result is None


# Test workspace exploration
def test_find_accessible_coordinates(robot):
    """Test the workspace exploration function"""
    # Use a coarse resolution for quick testing
    robot.find_accessible_coordinates(resolution=0.5)

    # Verify that at least some accessible coordinates were found
    assert len(robot.accessible_coords) > 0

    # Verify structure of accessible coordinates
    x, y, angles, statics, max_stress1, max_stress2, n1, n2 = robot.accessible_coords[0]
    assert isinstance(x, float)
    assert isinstance(y, float)
    assert len(angles) == 6
    assert len(statics) == 4
    assert isinstance(max_stress1, float)
    assert isinstance(max_stress2, float)
    assert isinstance(n1, float)
    assert isinstance(n2, float)


# Test visualization setup
def test_setup_animation(robot):
    """Test the animation setup function"""
    # Find some accessible coordinates first
    if not robot.accessible_coords:
        robot.find_accessible_coordinates(resolution=0.5)

    # Setup animation
    fig, ax = robot.setup_animation()

    # Verify figure and axes
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert robot.fig is fig
    assert robot.ax is ax

    # Clean up
    plt.close(fig)


# Test coordinate analysis functionality
def test_print_coordinates_info(robot, capsys):
    """Test the coordinate information printing function"""
    # Test with a valid point
    robot.print_coordinates_info(0, 8)
    captured = capsys.readouterr()
    assert "===== Coordinate Analysis:" in captured.out
    assert "STATUS:" in captured.out

    # Test with an invalid point
    robot.print_coordinates_info(0, 20)
    captured = capsys.readouterr()
    assert "===== Coordinate Analysis:" in captured.out
    assert "STATUS: OUT OF REACH" in captured.out


# Test the main visualization methods (simple smoke tests)
def test_visualization_methods(robot, monkeypatch):
    """Test visualization methods (smoke test only)"""
    # Mock plt.show to prevent actual windows from appearing
    monkeypatch.setattr(plt, 'show', lambda: None)
    monkeypatch.setattr(plt, 'pause', lambda x: None)

    # Make sure we have accessible coordinates
    if not robot.accessible_coords:
        robot.find_accessible_coordinates(resolution=0.5)

    # Test visualizing a specific point
    if robot.calculate_inverse_kinematics(0, 8) is not None:
        robot.visualize_point(0, 8)

    # For more complex methods like animate_all_coordinates,
    # we'll just do the minimal testing to avoid long test runs
    robot.animate_all_coordinates(delay=0.1, random_order=True)


# Run all the tests when executed directly
if __name__ == "__main__":
    pytest.main(["-v", __file__])
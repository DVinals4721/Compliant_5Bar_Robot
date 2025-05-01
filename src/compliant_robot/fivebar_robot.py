import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

class FiveBarRobot:
    def __init__(self, **kwargs):
        """
        Initialize the robot with customizable parameters

        Parameters:
        - beam_width (b): width of the beam (default: 0.5 m)
        - beam_height (h): height of the beam (default: 0.01 m)
        - youngs_modulus (E): Young's modulus of the material (default: 2.0e11 Pa for steel)
        - poisson_ratio (nu): Poisson's ratio of the material (default: 0.3 for steel)
        - yield_strength (Y): Yield strength of the material (default: 8.0e8 Pa for steel)
        - arm_length (l): Length of each arm (default: 10 m)
        - motor_distance (l0): Distance between motors (default: 1.0 m)
        - compliant_ratio (ye): Ratio of compliant segment length to total length (default: 0.85)
        - stiffness_factor (K0): Stiffness factor for the compliant mechanism (default: 2.65)
        - max_deformation_angle: Maximum deformation angle in degrees (default: 64.3)
        - max_force_ratio: Maximum allowed force ratio (default: 5.0)
        """
        # Steel flat beam Properties
        self.b = kwargs.get('beam_width', 0.5)
        self.h = kwargs.get('beam_height', 0.01)
        self.E = kwargs.get('youngs_modulus', 2.0e11)  # Steel default
        self.nu = kwargs.get('poisson_ratio', 0.3)  # Steel default
        self.Y = kwargs.get('yield_strength', 8.0e8)  # Steel default
        self.A = self.b * self.h
        self.Iy = self.h * (self.b**3) / 12
        self.Iz = self.b * (self.h**3) / 12
        self.l = kwargs.get('arm_length', 10)
        self.c = self.h/2

        self.l0 = kwargs.get('motor_distance', 1.0)  # Length between origin and the two motors

        # Cantilever Beam with a force at the free end compliant mechanism model parameters
        self.ye = kwargs.get('compliant_ratio', 0.85)
        self.K0 = kwargs.get('stiffness_factor', 2.65)

        # Limits
        self.max_deformation_angle = kwargs.get('max_deformation_angle', 64.3)  # in degrees
        self.max_force_ratio = kwargs.get('max_force_ratio', 5.0)

        # Colors
        self.arm_color = kwargs.get('arm_color', 'black')
        self.spline1_color_range = kwargs.get('spline1_color_range', [(0, 0, 1), (1, 0, 0)])  # Blue to Red
        self.spline2_color_range = kwargs.get('spline2_color_range', [(0, 0, 1), (1, 0, 0)])  # Blue to Red
        self.workspace_color = kwargs.get('workspace_color', 'yellow')
        self.marker_color = kwargs.get('marker_color', 'green')

        # Display parameters
        self.show_angles = kwargs.get('show_angles', True)
        self.show_points = kwargs.get('show_points', True)

        # Derived parameters
        self.K = self.ye * self.K0 * self.E * self.Iz / self.l
        self.l1 = self.l - self.ye * self.l
        self.l2 = self.ye * self.l

        self.y_limit = (self.l**2 - self.l0**2)**0.5 - 0.0001

        # Store accessible coordinates
        self.accessible_coords = []

        # Figure for animation
        self.fig = None
        self.ax = None

        # Print robot configuration
        self._print_config()

    def _print_config(self):
        """Print the current robot configuration"""
        print("=== Five-Bar Robot Configuration ===")
        print(f"Material Properties:")
        print(f"  - Young's Modulus: {self.E:.2e} Pa")
        print(f"  - Poisson's Ratio: {self.nu}")
        print(f"  - Yield Strength: {self.Y:.2e} Pa")
        print(f"Beam Dimensions:")
        print(f"  - Width: {self.b} m")
        print(f"  - Height: {self.h} m")
        print(f"Geometry:")
        print(f"  - Arm Length: {self.l} m")
        print(f"  - Motor Distance: {self.l0} m")
        print(f"  - Compliant Segment Ratio: {self.ye}")
        print(f"  - Fixed Segment Length: {self.l1} m")
        print(f"  - Compliant Segment Length: {self.l2} m")
        print(f"Limits:")
        print(f"  - Max Deformation Angle: {self.max_deformation_angle}°")
        print(f"  - Max Force Ratio: {self.max_force_ratio}")
        print("===================================")

    def calc_all_angles(self, x, y):
        """Calculate all angles for the five-bar mechanism"""
        try:
            # Angle from left shoulder to end effector
            beta1 = math.atan2(y, (self.l0 + x))

            # Angle from right shoulder to end effector
            beta2 = math.atan2(y, (self.l0 - x))

            # Alpha angle pre-calculations
            alpha1_calc = (self.l1**2 + ((self.l0 + x)**2 + y**2) - self.l2**2) / (2 * self.l1 * math.sqrt((self.l0 + x)**2 + y**2))
            alpha2_calc = (self.l1**2 + ((self.l0 - x)**2 + y**2) - self.l2**2) / (2 * self.l1 * math.sqrt((self.l0 - x)**2 + y**2))

            # If calculations > 1 or < -1, will fail acos function
            if alpha1_calc > 1 or alpha1_calc < -1 or alpha2_calc > 1 or alpha2_calc < -1:
                return None  # Unreachable coordinates

            # Angle of left shoulder - beta1 and right shoulder - beta2
            try:
                alpha1 = math.acos(alpha1_calc)
            except (ValueError, TypeError):
                return None  # Error in acos calculation

            try:
                alpha2 = math.acos(alpha2_calc)
            except (ValueError, TypeError):
                return None  # Error in acos calculation

            # Angles of left and right shoulders
            shoulder1 = beta1 + alpha1
            shoulder2 = math.pi - beta2 - alpha2

            # Handle potential acos errors for elbow angles
            try:
                elbow1_calc = (x + self.l0 - self.l1 * np.cos(shoulder1)) / self.l2
                if elbow1_calc > 1 or elbow1_calc < -1:
                    return None
                elbow1 = math.acos(elbow1_calc)

                elbow2_calc = (x - self.l0 - self.l1 * np.cos(shoulder2)) / self.l2
                if elbow2_calc > 1 or elbow2_calc < -1:
                    return None
                elbow2 = math.acos(elbow2_calc)
            except (ValueError, TypeError):
                return None  # Error in elbow angle calculation

            def_angle1 = elbow1 - shoulder1
            def_angle2 = elbow2 - shoulder2

            return [shoulder1, shoulder2, elbow1, elbow2, def_angle1, def_angle2]

        except Exception as e:
            # Any other error during calculation
            return None

    def calculate_statics(self, angles):
        """Calculate static forces based on the angles"""
        try:
            s1, s2, e1, e2, d1, d2 = angles
            a = np.array([
                [1, 0, self.l1 * np.cos(s1), -self.l1 * np.sin(s1)],
                [0, 0, self.l2 * np.cos(e1), -self.l2 * np.sin(e1)],
                [0, 0, -self.l2 * np.cos(e2), self.l2 * np.sin(e2)],
                [0, 1, -self.l1 * np.cos(s2), self.l1 * np.sin(s2)],
            ])
            ms1 = self.K * d1
            ms2 = self.K * d2
            b = np.array([-ms1, ms1, ms2, -ms2])

            # Check for ill-conditioned matrix
            if np.linalg.cond(a) > 1e15:
                return None

            try:
                x = np.linalg.solve(a, b)
                # Check for invalid results
                if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                    return None
                return x
            except np.linalg.LinAlgError:
                return None
        except Exception as e:
            # Any other error
            return None

    def calculate_stress(self, angles, statics):
        """Calculate stress in the mechanism"""
        try:
            s1, s2, e1, e2, d1, d2 = angles
            _, _, fy, fx = statics

            rot1 = np.array([
                [np.cos(-s1), -np.sin(-s1)],
                [np.sin(-s1), np.cos(-s1)]
            ])

            rot2 = np.array([
                [np.cos(-s2), -np.sin(-s2)],
                [np.sin(-s2), np.cos(-s2)]
            ])

            f1_new = np.dot(rot1, np.array([fx, fy]))
            f2_new = np.dot(rot2, np.array([fx, fy]))

            F1 = np.linalg.norm(f1_new)
            F2 = np.linalg.norm(f2_new)

            # Handle division by zero
            try:
                n1 = f1_new[0] / f1_new[1] if abs(f1_new[1]) > 1e-10 else float('inf')
                n2 = f2_new[0] / f2_new[1] if abs(f2_new[1]) > 1e-10 else float('inf')
            except:
                return None, None, float('inf'), float('inf')

            if np.isinf(n1) or np.isinf(n2) or np.isnan(n1) or np.isnan(n2):
                return None, None, n1, n2

            try:
                P1 = F1 / (1 + n1**2)**0.5
                P2 = F2 / (1 + n2**2)**0.5
            except (ValueError, ZeroDivisionError):
                return None, None, n1, n2

            a1 = self.l * (1 - self.ye * (1 - np.cos(d1)))
            a2 = self.l * (1 - self.ye * (1 - np.cos(d2)))

            b1 = self.ye * self.l * np.sin(d1)
            b2 = self.ye * self.l * np.sin(d2)

            max_stress1 = P1 * (a1 + n1 * b1) * self.c / self.Iz + n1 * P1 / self.A
            max_stress2 = P2 * (a2 + n2 * b2) * self.c / self.Iz + n2 * P2 / self.A

            # Check for invalid stress values
            if (np.isnan(max_stress1) or np.isinf(max_stress1) or
                np.isnan(max_stress2) or np.isinf(max_stress2)):
                return None, None, n1, n2

            return max_stress1, max_stress2, n1, n2

        except Exception as e:
            # Any other error during calculation
            return None, None, None, None

    def in_reach(self, x, y):
        """Check if the point (x,y) is within reach of both arms"""
        r1 = abs(np.linalg.norm(np.array([x, y]) - np.array([-self.l0, 0])))
        r2 = abs(np.linalg.norm(np.array([x, y]) - np.array([self.l0, 0])))

        return r1 < self.l and r2 < self.l

    def in_angle(self, angles):
        """Check if the deformation angles are within limits"""
        if angles is None:
            return False

        shoulder1, shoulder2, elbow1, elbow2, def_angle1, def_angle2 = angles
        max_angle_rad = self.max_deformation_angle * math.pi/180
        return abs(def_angle1) < max_angle_rad and abs(def_angle2) < max_angle_rad

    def in_n(self, n1, n2):
        """Check if the force ratios are within limits"""
        if n1 is None or n2 is None or np.isinf(n1) or np.isinf(n2) or np.isnan(n1) or np.isnan(n2):
            return False
        return abs(n1) < self.max_force_ratio and abs(n2) < self.max_force_ratio

    def create_smooth_spline(self, spline_params, num_points=100):
        """Creates a smooth spline curve with the given constraints"""
        # Extract parameters
        start_point = spline_params['start_point']
        end_point = spline_params['end_point']
        start_angle = spline_params['start_angle']
        end_angle = spline_params['end_angle']
        curve_length = spline_params['curve_length']

        # Calculate control points for cubic Bezier curve
        straight_distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)

        # Scale factor for control points
        scale_factor = (curve_length / straight_distance - 1) * 0.5 + 0.3

        # Create control points based on angles
        control_point1 = (
            start_point[0] + scale_factor * straight_distance * np.cos(start_angle),
            start_point[1] + scale_factor * straight_distance * np.sin(start_angle)
        )

        control_point2 = (
            end_point[0] - scale_factor * straight_distance * np.cos(end_angle),
            end_point[1] - scale_factor * straight_distance * np.sin(end_angle)
        )

        # Create a parametric Bezier curve
        def cubic_bezier(t, p0, p1, p2, p3):
            return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3

        # Generate points along the curve
        t_values = np.linspace(0, 1, num_points)
        x_points = cubic_bezier(t_values,
                            start_point[0],
                            control_point1[0],
                            control_point2[0],
                            end_point[0])
        y_points = cubic_bezier(t_values,
                            start_point[1],
                            control_point1[1],
                            control_point2[1],
                            end_point[1])

        return x_points, y_points

    def interpolate_color(self, c, color_range):
        """Interpolate between two colors based on value c (0 to 1)"""
        c = max(0, min(1, c))  # Clamp between 0 and 1
        color1, color2 = color_range
        r = color1[0] + c * (color2[0] - color1[0])
        g = color1[1] + c * (color2[1] - color1[1])
        b = color1[2] + c * (color2[2] - color1[2])
        return (r, g, b)

    def find_accessible_coordinates(self, resolution=0.01):
        """Find all accessible coordinates for the robot"""
        print("Finding accessible coordinates...")

        x_vector = np.linspace(-self.l0, self.l0, int((2 * self.l0) / resolution) + 1)
        y_vector = np.linspace(1, self.y_limit, int(self.y_limit / resolution) + 1)

        x_grid, y_grid = np.meshgrid(x_vector, y_vector)
        points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

        self.accessible_coords = []
        total_points = len(points)
        processed = 0
        valid_points = 0

        print(f"Testing {total_points} potential coordinates...")

        for x, y in points:
            processed += 1
            if processed % 1000 == 0:
                print(f"Processed {processed}/{total_points} points. Found {valid_points} valid coordinates.")

            try:
                # Check if the point is within physical reach
                if not self.in_reach(x, y):
                    continue

                # Calculate all angles
                angles = self.calc_all_angles(x, y)
                if angles is None:
                    continue

                # Calculate static forces
                statics = self.calculate_statics(angles)
                if statics is None:
                    continue

                # Calculate stresses
                max_stress1, max_stress2, n1, n2 = self.calculate_stress(angles, statics)
                if max_stress1 is None or max_stress2 is None:
                    continue

                # Check all constraints
                checks = [
                    self.in_angle(angles),
                    self.in_n(n1, n2)
                ]

                if all(checks):
                    self.accessible_coords.append((x, y, angles, statics, max_stress1, max_stress2, n1, n2))
                    valid_points += 1

            except Exception as e:
                # Skip any point that causes an exception
                continue

        print(f"Found {len(self.accessible_coords)} accessible coordinates out of {total_points} tested points")

        # If no accessible coordinates were found, warn the user
        if len(self.accessible_coords) == 0:
            print("WARNING: No accessible coordinates found! Try adjusting the robot parameters or increasing the search resolution.")

        return self.accessible_coords

    def plot_accessible_coordinates(self):
        """Plot all accessible coordinates"""
        if not self.accessible_coords:
            print("No accessible coordinates found. Run find_accessible_coordinates() first.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract x and y from accessible coordinates
        x_coords = [coord[0] for coord in self.accessible_coords]
        y_coords = [coord[1] for coord in self.accessible_coords]

        ax.scatter(x_coords, y_coords, s=10, alpha=0.5, c=self.workspace_color)

        # Plot robot base points
        ax.scatter([-self.l0, self.l0], [0, 0], color='red', s=100, marker='s')

        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title('Accessible Workspace for the Five-Bar Robot')
        ax.grid(True)
        ax.axis('equal')

        plt.tight_layout()
        plt.show()

    def plot_arms(self, shoulder1, shoulder2, efx, efy):
        """Plot the robot arms"""
        # Passive joints (x, y) location
        p1 = (-self.l0 + self.l1 * math.cos(shoulder1), self.l1 * math.sin(shoulder1))
        p2 = (self.l0 + self.l1 * math.cos(shoulder2), self.l1 * math.sin(shoulder2))

        # Left arm
        self.ax.plot([-self.l0, p1[0], efx], [0, p1[1], efy], color=self.arm_color, linestyle='-',
                    marker='o', markersize=8, linewidth=2.5, markerfacecolor=self.marker_color)
        if self.show_angles:
            self.ax.text(-self.l0-0.5, 0-0.5, f"{math.degrees(shoulder1):.1f}°", fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7))
        if self.show_points:
            self.ax.text(p1[0]+0.3, p1[1]+0.3, f"({p1[0]:.1f}, {p1[1]:.1f})", fontsize=8)

        # Right arm
        self.ax.plot([self.l0, p2[0], efx], [0, p2[1], efy], color=self.arm_color, linestyle='-',
                    marker='o', markersize=8, linewidth=2.5, markerfacecolor=self.marker_color)
        if self.show_angles:
            self.ax.text(self.l0+0.5, 0-0.5, f"{math.degrees(shoulder2):.1f}°", fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7))
        if self.show_points:
            self.ax.text(p2[0]+0.3, p2[1]+0.3, f"({p2[0]:.1f}, {p2[1]:.1f})", fontsize=8)

        # End effector
        self.ax.plot(efx, efy, marker='o', markersize=10, color='red')
        self.ax.text(efx+0.3, efy+0.3, f"({efx:.2f}, {efy:.2f})", fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))

    def plot_splines(self, x, y, angles, max_stress1, max_stress2):
        """Plot the splines for a given end-effector position"""
        shoulder1, shoulder2, elbow1, elbow2, def_angle1, def_angle2 = angles

        # Define splines as dictionaries with C values
        spline1 = {
            'start_point': (-self.l0, 0),
            'end_point': (x, y),
            'start_angle': shoulder1,
            'end_angle': 1.24 * def_angle1 + shoulder1,
            'curve_length': self.l,
            'C': 1 - abs((self.Y - max_stress1) / self.Y),  # 0 to 1 based on stress
            'label': 'Left Path'
        }

        spline2 = {
            'start_point': (self.l0, 0),
            'end_point': (x, y),
            'start_angle': shoulder2,
            'end_angle': 1.24 * def_angle2 + shoulder2,
            'curve_length': self.l,
            'C': 1 - abs((self.Y - max_stress2) / self.Y),  # 0 to 1 based on stress
            'label': 'Right Path'
        }

        # Create spline curves
        x_points1, y_points1 = self.create_smooth_spline(spline1)
        x_points2, y_points2 = self.create_smooth_spline(spline2)

        # Calculate colors based on stress level
        color1 = self.interpolate_color(spline1['C'], self.spline1_color_range)
        color2 = self.interpolate_color(spline2['C'], self.spline2_color_range)

        # Plot the curves
        self.ax.plot(x_points1, y_points1, color=color1, linewidth=3, label=spline1['label'], alpha=0.7)
        self.ax.plot(x_points2, y_points2, color=color2, linewidth=3, label=spline2['label'], alpha=0.7)

        # Plot the arms
        self.plot_arms(shoulder1, shoulder2, x, y)

    def setup_animation(self):
        """Setup plot for animation"""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(-self.l*1.2, self.l*1.2)
        self.ax.set_ylim(-1, self.y_limit*1.2)
        self.ax.set_title('Five-Bar Parallel Robot Animation with Accessible Workspace')
        self.ax.grid(True)
        self.ax.set_aspect('equal')

        # Plot robot base points
        self.ax.scatter([-self.l0, self.l0], [0, 0], color='red', s=100, marker='s')

        # Plot all accessible coordinates as a background
        if self.accessible_coords:
            x_coords = [coord[0] for coord in self.accessible_coords]
            y_coords = [coord[1] for coord in self.accessible_coords]
            self.ax.scatter(x_coords, y_coords, s=5, alpha=0.2, c=self.workspace_color, label='Accessible Workspace')

        return self.fig, self.ax

    def animate_all_coordinates(self, delay=0.5, random_order=True):
        """
        Animate the robot moving to all accessible coordinates

        Parameters:
        - delay: Time delay between frames in seconds
        - random_order: If True, visit points in random order; if False, visit in sequence
        """
        if not self.accessible_coords:
            print("No accessible coordinates found. Run find_accessible_coordinates() first.")
            return

        self.setup_animation()

        # Add a legend for the workspace
        self.ax.legend(loc='upper right')

        # Create an index array for all points
        num_points = len(self.accessible_coords)
        indices = list(range(num_points))

        # Randomize the order if requested
        if random_order:
            random.shuffle(indices)

        # Create a color bar for stress visualization

        # Create a custom colormap for stress visualization
        colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]  # Blue -> Green -> Red
        cmap_name = 'stress_cmap'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

        # Add a colorbar to show stress levels
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=self.ax)
        cbar.set_label('Stress Level (Normalized)')

        # Display total number of points
        print(f"Animating movement through all {num_points} accessible coordinates")

        # Track timing for estimated completion
        start_time = time.time()

        for i, idx in enumerate(indices):
            if i > 0:  # Don't clear on the first frame to keep the workspace plot
                # Clear only the splines and robot, but keep the workspace points
                for artist in self.ax.lines + self.ax.texts:
                    artist.remove()

            # Re-plot base points
            self.ax.scatter([-self.l0, self.l0], [0, 0], color='red', s=100, marker='s')

            x, y, angles, statics, max_stress1, max_stress2, n1, n2 = self.accessible_coords[idx]

            # Update the title to show current position and progress
            progress = (i + 1) / num_points * 100
            self.ax.set_title(f'Five-Bar Compliant Robot Workspace Animation - Position: ({x:.2f}, {y:.2f}) - Progress: {progress:.1f}%')

            # Plot the splines and arms
            self.plot_splines(x, y, angles, max_stress1, max_stress2)

            # Add current position info
            stress_ratio1 = max_stress1 / self.Y
            stress_ratio2 = max_stress2 / self.Y
            info_text = f"Max Stress Ratio - Left: {stress_ratio1:.2f}, Right: {stress_ratio2:.2f}"
            #self.ax.text(0, -0.5, info_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

            # Add force ratio info
            force_text = f"Force Ratio - Left: {n1:.2f}, Right: {n2:.2f}"
            #self.ax.text(0, -0.8, force_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

            # Add deformation angle info
            deform_text = f"Deform Angle - Left: {math.degrees(angles[4]):.1f}°, Right: {math.degrees(angles[5]):.1f}°"
            #self.ax.text(0, -1.1, deform_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

            # Estimate time remaining
            if i > 0:
                elapsed = time.time() - start_time
                points_per_sec = (i + 1) / elapsed
                remaining_points = num_points - (i + 1)
                est_remaining = remaining_points / points_per_sec

                time_text = f"Time Remaining: {est_remaining:.1f} seconds"
                self.ax.text(0, -5, time_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

            plt.pause(delay)

        # Add a final message
        plt.title("Animation Complete - All Points Visited")
        plt.show()
    def calculate_inverse_kinematics(self, x, y):
            """
            Calculate the inverse kinematics for the given end-effector position.
            Returns the shoulder angles and required motor torques.

            Parameters:
            - x, y: Coordinates of the end-effector

            Returns:
            - tuple: (shoulder1, shoulder2, mm1, mm2) if valid, None if not reachable
            """
            # First, check if the point is within reach
            if not self.in_reach(x, y):
                print(f"Point ({x}, {y}) is out of reach.")
                return None

            # Calculate all angles
            angles = self.calc_all_angles(x, y)
            if angles is None:
                print(f"Point ({x}, {y}) has no valid angle solution.")
                return None

            # Calculate static forces
            statics = self.calculate_statics(angles)
            if statics is None:
                print(f"Point ({x}, {y}) has no valid static solution.")
                return None

            # Unpack angles and statics
            shoulder1, shoulder2, _, _, _, _ = angles
            mm1, mm2, _, _ = statics

            # Verify angles and forces are within limits
            max_stress1, max_stress2, n1, n2 = self.calculate_stress(angles, statics)
            if max_stress1 is None or max_stress2 is None:
                print(f"Point ({x}, {y}) has invalid stress values.")
                return None

            # Check all constraints
            if not self.in_angle(angles):
                print(f"Point ({x}, {y}) exceeds deformation angle limits.")
                return None

            if not self.in_n(n1, n2):
                print(f"Point ({x}, {y}) exceeds force ratio limits.")
                return None

            return (shoulder1, shoulder2, mm1, mm2)

    def print_coordinates_info(self, x, y):
        """
        Print detailed information about a specific coordinate point.
        Shows whether the point is reachable and all relevant values.

        Parameters:
        - x, y: Coordinates to check
        """
        print(f"\n===== Coordinate Analysis: ({x:.2f}, {y:.2f}) =====")

        # First, check if the point is within reach
        if not self.in_reach(x, y):
            print("STATUS: OUT OF REACH")
            print(f"Distance from left motor: {abs(np.linalg.norm(np.array([x, y]) - np.array([-self.l0, 0]))):.2f}")
            print(f"Distance from right motor: {abs(np.linalg.norm(np.array([x, y]) - np.array([self.l0, 0]))):.2f}")
            print(f"Maximum reach: {self.l:.2f}")
            return

        # Calculate all angles
        angles = self.calc_all_angles(x, y)
        if angles is None:
            print("STATUS: NO VALID ANGLE SOLUTION")
            return

        # Calculate static forces
        statics = self.calculate_statics(angles)
        if statics is None:
            print("STATUS: NO VALID STATIC SOLUTION")
            return

        # Calculate stresses
        max_stress1, max_stress2, n1, n2 = self.calculate_stress(angles, statics)
        if max_stress1 is None or max_stress2 is None:
            print("STATUS: INVALID STRESS VALUES")
            return

        # Unpack angles and statics
        shoulder1, shoulder2, elbow1, elbow2, def_angle1, def_angle2 = angles
        mm1, mm2, fy, fx = statics

        # Check constraints
        angle_valid = self.in_angle(angles)
        force_valid = self.in_n(n1, n2)

        if angle_valid and force_valid:
            print("STATUS: VALID AND REACHABLE ✓")
        else:
            print("STATUS: GEOMETRICALLY REACHABLE BUT CONSTRAINTS VIOLATED ✗")

        # Print detailed information
        print("\nAngles (degrees):")
        print(f"  Left Shoulder: {math.degrees(shoulder1):.2f}°")
        print(f"  Right Shoulder: {math.degrees(shoulder2):.2f}°")
        print(f"  Left Elbow: {math.degrees(elbow1):.2f}°")
        print(f"  Right Elbow: {math.degrees(elbow2):.2f}°")
        print(f"  Left Deformation: {math.degrees(def_angle1):.2f}° (Limit: ±{self.max_deformation_angle:.1f}°) {'✓' if abs(math.degrees(def_angle1)) < self.max_deformation_angle else '✗'}")
        print(f"  Right Deformation: {math.degrees(def_angle2):.2f}° (Limit: ±{self.max_deformation_angle:.1f}°) {'✓' if abs(math.degrees(def_angle2)) < self.max_deformation_angle else '✗'}")

        print("\nForces:")
        print(f"  End-Effector Force: ({fx:.2f}, {fy:.2f}) N")
        print(f"  Left Force Ratio (n1): {n1:.2f} (Limit: ±{self.max_force_ratio:.1f}) {'✓' if abs(n1) < self.max_force_ratio else '✗'}")
        print(f"  Right Force Ratio (n2): {n2:.2f} (Limit: ±{self.max_force_ratio:.1f}) {'✓' if abs(n2) < self.max_force_ratio else '✗'}")

        print("\nStresses:")
        print(f"  Left Max Stress: {max_stress1:.2e} Pa (Yield: {self.Y:.2e} Pa) {'✓' if max_stress1 < self.Y else '✗'}")
        print(f"  Right Max Stress: {max_stress2:.2e} Pa (Yield: {self.Y:.2e} Pa) {'✓' if max_stress2 < self.Y else '✗'}")

        print("\nMotor Torques:")
        print(f"  Left Motor Torque (mm1): {mm1:.2f} Nm")
        print(f"  Right Motor Torque (mm2): {mm2:.2f} Nm")
        print("="*50)

    def visualize_point(self, x, y):
        """
        Visualize a specific point with all the robot details

        Parameters:
        - x, y: Coordinates to visualize
        """
        # Calculate inverse kinematics
        result = self.calculate_inverse_kinematics(x, y)
        if result is None:
            print(f"Cannot visualize point ({x}, {y}) - not reachable")
            return

        shoulder1, shoulder2, mm1, mm2 = result

        # Setup a plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(-self.l*1.2, self.l*1.2)
        ax.set_ylim(-1, self.y_limit*1.2)
        ax.set_title(f'Five-Bar Robot at Position ({x:.2f}, {y:.2f})')
        ax.grid(True)
        ax.set_aspect('equal')

        # Plot robot base points
        ax.scatter([-self.l0, self.l0], [0, 0], color='red', s=100, marker='s')

        # We need to recalculate all angles for visualization
        angles = self.calc_all_angles(x, y)
        statics = self.calculate_statics(angles)
        max_stress1, max_stress2, n1, n2 = self.calculate_stress(angles, statics)

        # Store the current axes for the plot_splines method
        self.ax = ax

        # Plot the splines and arms
        self.plot_splines(x, y, angles, max_stress1, max_stress2)

        # Add annotations
        deform_text = f"Deform Angles: Left={math.degrees(angles[4]):.1f}°, Right={math.degrees(angles[5]):.1f}°"
        torque_text = f"Motor Torques: Left={mm1:.2f} Nm, Right={mm2:.2f} Nm"
        force_text = f"Force Ratios: Left={n1:.2f}, Right={n2:.2f}"

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, deform_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        ax.text(0.05, 0.9, torque_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        ax.text(0.05, 0.85, force_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.show()


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
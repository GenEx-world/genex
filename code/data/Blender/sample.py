import bpy
import random
import math
import mathutils

# Global parameters
MAX_START_ATTEMPTS = 100  # Maximum attempts to find a valid start location
MAX_DIRECTION_ATTEMPTS = 10  # Maximum attempts to find a valid direction
DEFAULT_STEP_SIZE = 20  # Default distance for path length
LOCATION_BOUNDS = {  # Space boundaries for sampling locations
    'min_x': -3.51543,
    'max_x': 3.3263,
    'min_y': -3.61569,
    'max_y': 3.40674,
    'z': 1.2  # Fixed height
}
RAY_MAX_DISTANCE = 10.0  # Maximum distance for upward ray casting

def is_location_valid(location):
    """
    Check if a location is valid by ensuring there is no object within specified distance above.
    
    Args:
        location (tuple): (x, y, z) coordinates to check
        
    Returns:
        bool: True if location is valid, False otherwise
    """
    ray_origin = location
    ray_direction = (0, 0, 1)  # Cast ray upwards
    
    # Perform ray casting and return True if no hit is found
    hit, _, _, _, _, _ = bpy.context.scene.ray_cast(
        bpy.context.view_layer.depsgraph, ray_origin, ray_direction, distance=RAY_MAX_DISTANCE
    )
    
    return not hit


def is_path_clear(start_location, direction, distance):
    """
    Check if a path is clear by casting multiple rays along the path.
    Tests a vertical block (±0.2 units) and horizontal block (±0.1 units).
    
    Args:
        start_location (tuple): Starting point (x, y, z)
        direction (tuple): Direction vector (x, y)
        distance (float): Distance to check
        
    Returns:
        tuple: (is_clear, end_location) - Boolean and end point coordinates
    """
    # Calculate the end location
    end_location = (
        start_location[0] + direction[0] * distance,
        start_location[1] + direction[1] * distance,
        start_location[2]
    )
    
    # Define sampling steps for grid
    vertical_step = 0.1
    horizontal_step = 0.1

    # Create a grid of ray origins and check each point
    for dz in [-0.2, 0.0, 0.2]:  # Vertical test points
        for dx in [-0.1, 0.0, 0.1]:  # Horizontal X offsets
            for dy in [-0.1, 0.0, 0.1]:  # Horizontal Y offsets
                ray_origin = (
                    start_location[0] + dx,
                    start_location[1] + dy,
                    start_location[2] + dz
                )
                
                ray_direction = (direction[0], direction[1], 0)  # Horizontal direction
                
                # Check for a hit with this ray
                hit, _, _, _, _, _ = bpy.context.scene.ray_cast(
                    bpy.context.view_layer.depsgraph, ray_origin, ray_direction, distance=distance
                )
                
                if hit:
                    return False, end_location  # Path blocked

    return True, end_location  # All rays clear


def sample_valid_location(max_attempts=MAX_START_ATTEMPTS):
    """
    Sample a valid location by randomly testing points.
    
    Args:
        max_attempts (int): Maximum sampling attempts
        
    Returns:
        tuple: (location, trial_count) - Valid location and number of attempts,
               or (None, trial_count) if no valid location found
    """
    trial_count = 0
    
    while trial_count < max_attempts:
        trial_count += 1
        
        # Generate random coordinates within bounds
        x = random.uniform(LOCATION_BOUNDS['min_x'], LOCATION_BOUNDS['max_x'])
        y = random.uniform(LOCATION_BOUNDS['min_y'], LOCATION_BOUNDS['max_y'])
        z = LOCATION_BOUNDS['z']  # Fixed height
        location = (x, y, z)
        
        if is_location_valid(location):
            return location, trial_count
            
    return None, trial_count


def find_valid_path(start_location, max_attempts=MAX_DIRECTION_ATTEMPTS, step_size=DEFAULT_STEP_SIZE):
    """
    Find a valid path from the start location.
    
    Args:
        start_location (tuple): Starting point coordinates
        max_attempts (int): Maximum attempts to try different directions
        step_size (float): Length of the path to check
        
    Returns:
        tuple: (direction, end_position, trial_count, angle) or (None, None, trial_count, None)
    """
    trial_count = 0
    
    for _ in range(max_attempts):
        trial_count += 1
        
        # Generate random direction
        angle = random.uniform(0, 2 * math.pi)
        direction = (math.cos(angle), math.sin(angle))
        
        # Check if path is clear
        is_clear, final_position = is_path_clear(start_location, direction, step_size)
        
        if is_clear:
            return (direction[0], direction[1], 0), final_position, trial_count, angle
            
    return None, None, trial_count, None


def get_valid_start_end_and_direction(max_start_attempts=MAX_START_ATTEMPTS, 
                                      max_direction_attempts=MAX_DIRECTION_ATTEMPTS, 
                                      step_size=DEFAULT_STEP_SIZE):
    """
    Find a valid camera path with start/end locations and direction.
    
    Args:
        max_start_attempts (int): Maximum attempts for start location
        max_direction_attempts (int): Maximum attempts for direction
        step_size (float): Distance between start and end points
        
    Returns:
        tuple: (start_location, end_location, direction, total_start_trials,
                total_direction_trials, angle)
    """
    total_start_trials = 0
    total_direction_trials = 0
    
    while total_start_trials < max_start_attempts:
        # Sample a valid location
        valid_location, start_trials = sample_valid_location(max_start_attempts)
        total_start_trials += start_trials
        
        if valid_location is None:
            continue
        
        # Try to find a valid path
        direction, final_position, direction_trials, angle = find_valid_path(
            valid_location, max_direction_attempts, step_size
        )
        total_direction_trials += direction_trials
        
        if direction is not None and final_position is not None:
            return (valid_location, final_position, direction, 
                    total_start_trials, total_direction_trials, angle)
        else:
            print("No valid path found, resampling start location.")
    
    return None, None, None, total_start_trials, total_direction_trials, None
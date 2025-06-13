import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def generate_vectors_precise(n, k, max_distance=None, learning_rate=0.005, iterations=10000, max_error=1e-5):
    while True:

        # Generate random integer magnitudes that sum to n
        magnitudes = np.ones(k, dtype=int)

        if max_distance == None:
            max_distance = n // 2

        # Calculate remaining distance to distribute
        remaining_distance = n - k

        # Randomly distribute the remaining distance with the cap of max_distance
        while remaining_distance > 0:
            index = np.random.randint(0, k)
            if magnitudes[index] < max_distance:
                magnitudes[index] += 1
                remaining_distance -= 1

        if k == 2:
            assert n % 2 == 0, 'n must be divisible by 2 when k equals to 2 (move forward and directly backward)'
            magnitudes[0] = n // 2
            magnitudes[1] = magnitudes[0]

        # Normalize magnitudes
        normalized_magnitudes = magnitudes / n
        
        # Initialize random angles
        angles = np.random.uniform(0, 2 * np.pi, k)

        # Iteratively adjust angles to minimize resultant vector
        for iteration in range(iterations):
            x_sum = np.sum(normalized_magnitudes * np.cos(angles))
            y_sum = np.sum(normalized_magnitudes * np.sin(angles))

            if np.isclose(x_sum, 0, atol=max_error) and np.isclose(y_sum, 0, atol=max_error):
                break

            # Compute gradients
            grad_x = -normalized_magnitudes * np.sin(angles)
            grad_y = normalized_magnitudes * np.cos(angles)

            # Update angles with scaled learning rate
            angles -= learning_rate * n * (x_sum * grad_x + y_sum * grad_y)

        # Check error again after the loop ends
        x_sum = np.sum(normalized_magnitudes * np.cos(angles))
        y_sum = np.sum(normalized_magnitudes * np.sin(angles))

        if np.isclose(x_sum, 0, atol=max_error) and np.isclose(y_sum, 0, atol=max_error):
            print('Magnitudes:', magnitudes)
            print('Angles:', [np.degrees(angle) for angle in angles])
            return magnitudes, angles
        else:
            print('Retrying')

def construct_path(n, k, max_distance=None, max_error=1e-5):

    if k > n:
        k = n

    magnitudes, angles = generate_vectors_precise(n, k, max_distance, max_error=max_error)
    
    path = []
    current_angle = 0.0  # Start with angle 0
    current_position = np.array([0.0, 0.0])  # Allow position to be float

    for i in range(k):
        # Calculate the angle to turn to reach the new angle
        turn_angle = np.degrees(angles[i]) - current_angle
        # Normalize the angle to be positive between 0 and 360
        turn_angle = turn_angle % 360

        # Update current angle to the new angle
        new_angle = np.degrees(angles[i]) % 360
        current_angle = new_angle
        
        # Calculate new position
        move_distance = magnitudes[i]
        delta_x = move_distance * np.cos(angles[i])
        delta_y = move_distance * np.sin(angles[i])
        new_position = current_position + np.array([delta_x, delta_y])

        # Record the step
        path.append({
            'turn_angle': turn_angle,
            'move_distance': move_distance,
            'new_position': (round(new_position[0], 2), round(new_position[1], 2)),
            'new_angle': new_angle
        })

        # Update current position
        current_position = new_position

    # Calculate final angle correction if needed
    corrective_turn_angle = -current_angle % 360
    path.append({
        'turn_angle': corrective_turn_angle,
        'move_distance': 0,
        'new_position': (round(new_position[0], 2), round(new_position[1], 2)),
        'new_angle': (current_angle + corrective_turn_angle) % 360
    })

    return path


def draw_path(path, save_path=None):
    fig, ax = plt.subplots()
    current_position = np.array([0.0, 0.0])

    # Draw origin point
    ax.plot(0, 0, 'ko', label='Origin', markersize=8)  # Black dot at origin

    # Get the number of steps to define color map
    num_steps = len(path)
    colors = cm.viridis(np.linspace(0, 1, num_steps))  # Use a colormap for arrows

    # Plot each step as an arrow
    for i, step in enumerate(path):
        new_position = np.array(step['new_position'])
        vector = new_position - current_position
        ax.arrow(
            current_position[0], current_position[1], 
            vector[0], vector[1], 
            head_width=0.2, head_length=0.4, fc=colors[i], ec=colors[i],
            length_includes_head=True
        )
        current_position = new_position

    # Set equal scaling and plot grid
    ax.set_aspect('equal', 'box')
    ax.grid(True)

    # Calculate the plot limits based on the maximum distance from the origin
    max_distance = np.max(np.sqrt(np.sum((np.array([step['new_position'] for step in path]))**2, axis=1)))
    limit = max_distance + 1  # Add a margin
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    # Set labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('2D Trajectory with Vectors')
    plt.legend()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # Show plot
    plt.show()
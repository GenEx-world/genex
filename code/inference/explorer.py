from PIL import Image
import numpy as np
from math import pi, atan2, hypot
import torch
from diffusers import UNetSpatioTemporalConditionModel, StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif
import imageio
import cv2
from tqdm import tqdm

class Explorer:
    def __init__(self):
        self.generations = []

    def set_current_image(self, image):
        self.generations.append([image])

    def get_current_image(self):
        return self.generations[-1][-1]

    def clear_movements(self):
        self.generations = []

    def get_all_frames(self):
        flattened_frames = [frame for movement in self.generations for frame in movement]
        return flattened_frames

    def get_pipeline(self, unet_path, svd_path, num_frames=25, fps=7, progress_bar=True, image_width=1024, image_height=512, model_width=1024, model_height=576):
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            unet_path,
            subfolder="unet",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, 
        )
        print("[Explorer] UNet model loaded.")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            svd_path,
            unet=unet,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        pipe.set_progress_bar_config(disable=not progress_bar)
        pipe.to("cuda:0")
        print("[Explorer] GenEx Video Diffusion pipeline initialized.")
        self.pipe = pipe

        self.image_width = image_width
        self.image_height = image_height
        self.model_width = model_width
        self.model_height = model_height
        self.num_frames = num_frames
        self.fps = fps

        return pipe

    def move_forward(self, image=None, num_frames=25, num_steps=None, width=None, height=None, num_inference_steps=30, noise_aug_strength=0.02):
        if not image:
            image = self.generations[-1][-1]

        model_width = self.model_width if self.model_width else width
        model_height = self.model_height if self.model_height else width

        width = image.size[0] if not width else width
        height = image.size[1] if not height else height

        image = image.resize((model_width, model_height), Image.BICUBIC).convert('RGB')

        num_frames = num_steps + 1 if num_steps else num_frames

        generator = torch.manual_seed(-1)
        with torch.inference_mode():
            frames = self.pipe(image,
                        num_frames=self.num_frames,
                        width=model_width,
                        height=model_height,
                        decode_chunk_size=8, generator=generator, motion_bucket_id=127, fps=self.fps, num_inference_steps=num_inference_steps, noise_aug_strength=noise_aug_strength).frames[0]
        print(f'Generated {num_frames} frames successfully')

        frames = frames[:num_frames]
        frames = [frame.resize((width, height), Image.BICUBIC) for frame in frames]

        self.generations.append(frames[1:])

        return frames

    def save_video(self, save_path, fps=10):
        # Create a writer object
        writer = imageio.get_writer(save_path, fps=fps)

        frames = [frame.resize((self.image_width, self.image_height), Image.BICUBIC) for movement in self.generations for frame in movement]

        if len(frames) == 0:
            print("[Explorer] No movements to export. Please generate frames first.")
            return

        # Add images to the video
        for frame in tqdm(frames, desc="Processing frames for video export"):
            # Convert the PIL image to a numpy array
            frame = np.array(frame)
            writer.append_data(frame)

        # Close the writer to finalize the video
        writer.close()

        print(f"[Explorer] Video successfully saved as {save_path}")

    def save_gif(self, save_path, fps=10):
        # Calculate the duration of each frame in the GIF
        duration = int(1000 / fps)  # duration in milliseconds per frame

        frames = [frame.resize((self.image_width, self.image_height), Image.BICUBIC) for movement in self.generations for frame in movement]

        if len(frames) == 0:
            print("[Explorer] No movements to export. Please generate frames first.")
            return

        # Convert frames to PIL images and save as GIF with duration
        frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)

        print(f"[Explorer] GIF successfully saved as {save_path}")

    def navigate_path(self, path, start_image, num_inference_steps=50):
        """
        Example Trajectory: 
        {'turn_angle': 62.44414980499492, 'move_distance': 5, 'new_position': (2.31, 4.43), 'new_angle': 62.44414980499492}
        {'turn_angle': 261.78416914120317, 'move_distance': 7, 'new_position': (7.99, 0.34), 'new_angle': 324.2283189461981}
        {'turn_angle': 218.2137484705168, 'move_distance': 8, 'new_position': (-0.0, 0.0), 'new_angle': 182.4420674167149}
        {'turn_angle': 177.5579325832851, 'move_distance': 0, 'new_position': (-0.0, 0.0), 'new_angle': 0.0}
        """

        generations = []
        
        current_image = start_image
        generations.append([current_image])

        while len(path) > 0:
            step = path.pop(0)

            turn_angle = step['turn_angle']
            move_distance = step['move_distance']

            if move_distance > self.num_frames:
                move_distance = self.num_frames - 1
                path.insert(0, {'turn_angle': 0, 'move_distance': step['move_distance'] - move_distance, 'new_position': step['new_position'], 'new_angle': step['new_angle']})

            if turn_angle != 0:
                current_image = self.rotate_panorama(current_image, turn_angle)

            if move_distance != 0:
                movement = self.move_forward(current_image, num_steps=move_distance, num_inference_steps=num_inference_steps)
                generations.append(movement)
                current_image = movement[-1]
            else:
                generations.append([current_image])
        
        self.generations = generations
        print(f"[Explorer] Path navigation completed with {len(generations)} movement segments.")
        return generations

    def rotate_panorama(self, panorama_image=None, rotation_degrees=0, scale_factor=5):
        """
        Rotate an equirectangular panorama image along the z-axis using spherical coordinates.

        Parameters:
        panorama_image (Image): The input equirectangular panorama image.
        rotation_degrees (float): The amount to rotate the image in degrees. Positive values rotate to the right.
        scale_factor (int): The factor by which to scale the image for higher-quality processing.

        Returns:
        Image: The rotated equirectangular panorama image.
        """

        if not panorama_image:
            panorama_image = self.get_current_image()

        # Scale up the image for processing
        original_size = panorama_image.size
        scaled_size = (original_size[0] * scale_factor, original_size[1] * scale_factor)
        panorama_image = panorama_image.resize(scaled_size, Image.LANCZOS)

        # Convert the input image to a numpy array
        panorama_array = np.array(panorama_image)
        height, width, _ = panorama_array.shape

        # Convert rotation degrees to radians (no negation for clockwise rotation)
        rotation_radians = np.deg2rad(rotation_degrees)

        # Create a meshgrid for the pixel coordinates
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        xv, yv = np.meshgrid(x, y)

        # Calculate the spherical coordinates (longitude and latitude)
        longitude = (xv / width) * 2 * pi
        latitude = (yv / height) * pi - (pi / 2)

        # Adjust the longitude by the rotation amount for clockwise rotation
        rotated_longitude = (longitude + rotation_radians) % (2 * pi)

        # Convert spherical coordinates back to image coordinates
        uf = rotated_longitude / (2 * pi) * width
        vf = (latitude + (pi / 2)) / pi * height

        # Ensure indices are within bounds
        ui = np.clip(uf, 0, width - 1).astype(int)
        vi = np.clip(vf, 0, height - 1).astype(int)

        # Create the rotated image using the new coordinates
        rotated_array = panorama_array[vi, ui]

        # Convert the numpy array back to an image
        rotated_image = Image.fromarray(rotated_array)

        # Resize back to the original size for output
        rotated_image = rotated_image.resize(original_size, Image.LANCZOS)

        self.generations.append([rotated_image])
        print(f"[Explorer] Rotated panorama by {rotation_degrees} degrees.")

        return rotated_image

    def convert_panorama_to_cubemap(self, panorama_image=None, interpolation=True, scale_factor=2):
        """
        Convert an equirectangular panorama image to a cube map with optional scaling.

        Parameters:
        panorama_image (Image): The input equirectangular panorama image.
        interpolation (bool): Whether to use bilinear interpolation for sampling.
        scale_factor (int): Factor by which to scale the input panorama.

        Returns:
        tuple: A tuple containing the cube map image and a dictionary of individual cube faces.
        """

        if not panorama_image:
            panorama_image = self.get_current_image()

        # Scale up the input panorama
        original_size = panorama_image.size
        scaled_size = (original_size[0] * scale_factor, original_size[1] * scale_factor)
        panorama_image = panorama_image.resize(scaled_size, Image.LANCZOS)

        # Assert that the panorama image has the correct aspect ratio
        panorama_size = panorama_image.size
        assert panorama_size[0] == 2 * panorama_size[1], "Panorama width must be twice the height."

        # Create an output image with appropriate dimensions for the cube map
        cubemap = Image.new("RGB", (panorama_size[0], int(panorama_size[0] * 3 / 4)), "black")

        input_pixels = np.array(panorama_image)
        cubemap_pixels = np.zeros((cubemap.size[1], cubemap.size[0], 3), dtype=np.uint8)  # Initialize with black
        edge_length = panorama_size[0] // 4  # Length of each edge in pixels

        # Create coordinate grids
        i, j = np.meshgrid(np.arange(cubemap.size[0]), np.arange(cubemap.size[1]), indexing='xy')

        # Assign face indices
        face_index = i // edge_length
        face_index[j < edge_length] = 4  # 'top'
        face_index[j >= 2 * edge_length] = 5  # 'bottom'

        def pixel_to_xyz(i, j, face):
            """
            Convert pixel coordinates of the output image to 3D coordinates based on the cube face index.
            
            Parameters:
            i (int): The x-coordinate of the pixel.
            j (int): The y-coordinate of the pixel.
            face (int): The face of the cube (0-5).

            Returns:
            tuple: A tuple (x, y, z) representing the 3D coordinates.
            """
            a = 2.0 * i / edge_length
            b = 2.0 * j / edge_length

            x = np.zeros_like(a, dtype=float)
            y = np.zeros_like(a, dtype=float)
            z = np.zeros_like(a, dtype=float)

            # Assign 3D coordinates based on the face index
            mask = face == 0  # back
            x[mask], y[mask], z[mask] = -1.0, 1.0 - a[mask], 3.0 - b[mask]

            mask = face == 1  # left
            x[mask], y[mask], z[mask] = a[mask] - 3.0, -1.0, 3.0 - b[mask]

            mask = face == 2  # front
            x[mask], y[mask], z[mask] = 1.0, a[mask] - 5.0, 3.0 - b[mask]

            mask = face == 3  # right
            x[mask], y[mask], z[mask] = 7.0 - a[mask], 1.0, 3.0 - b[mask]

            mask = face == 4  # top
            x[mask], y[mask], z[mask] = b[mask] - 1.0, a[mask] - 5.0, 1.0

            mask = face == 5  # bottom
            x[mask], y[mask], z[mask] = 5.0 - b[mask], a[mask] - 5.0, -1.0

            return x, y, z

        # Convert pixel coordinates to 3D coordinates
        x, y, z = pixel_to_xyz(i, j, face_index)
        theta = np.arctan2(y, x)  # Angle in the xy-plane
        r = np.hypot(x, y)  # Distance from origin in the xy-plane
        phi = np.arctan2(z, r)  # Angle from the z-axis

        # Source image coordinates
        uf = 2.0 * edge_length * (theta + pi) / pi
        vf = 2.0 * edge_length * (pi / 2 - phi) / pi

        if interpolation:
            # Bilinear interpolation
            ui = np.floor(uf).astype(int)
            vi = np.floor(vf).astype(int)
            u2 = ui + 1
            v2 = vi + 1
            mu = uf - ui
            nu = vf - vi

            # Ensure indices are within bounds
            ui = np.clip(ui, 0, panorama_size[0] - 1)
            vi = np.clip(vi, 0, panorama_size[1] - 1)
            u2 = np.clip(u2, 0, panorama_size[0] - 1)
            v2 = np.clip(v2, 0, panorama_size[1] - 1)

            # Pixel values of the four corners
            A = input_pixels[vi, ui]
            B = input_pixels[vi, u2]
            C = input_pixels[v2, ui]
            D = input_pixels[v2, u2]

            # Interpolate the RGB values
            R = A[:, :, 0] * (1 - mu) * (1 - nu) + B[:, :, 0] * mu * (1 - nu) + C[:, :, 0] * (1 - mu) * nu + D[:, :, 0] * mu * nu
            G = A[:, :, 1] * (1 - mu) * (1 - nu) + B[:, :, 1] * mu * (1 - nu) + C[:, :, 1] * (1 - mu) * nu + D[:, :, 1] * mu * nu
            B = A[:, :, 2] * (1 - mu) * (1 - nu) + B[:, :, 2] * mu * (1 - nu) + C[:, :, 2] * (1 - mu) * nu + D[:, :, 2] * mu * nu

            interp_pixels = np.stack((R, G, B), axis=-1).astype(np.uint8)

            # Ensure all pure black pixels in the nearest-neighbor result are directly used in the final output
            cubemap_pixels = interp_pixels
        else:
            # Nearest-neighbor sampling
            ui = np.round(uf).astype(int)
            vi = np.round(vf).astype(int)

            valid = (ui >= 0) & (ui < panorama_size[0]) & (vi >= 0) & (vi < panorama_size[1])
            cubemap_pixels[(face_index >= 0) & valid] = input_pixels[vi[(face_index >= 0) & valid], ui[(face_index >= 0) & valid]]

        # First row: set empty spaces to black
        cubemap_pixels[0:edge_length, 0:edge_length] = [0, 0, 0]
        cubemap_pixels[0:edge_length, edge_length:2 * edge_length] = [0, 0, 0]
        cubemap_pixels[0:edge_length, 3 * edge_length:] = [0, 0, 0]

        # Third row: set empty spaces to black
        cubemap_pixels[2 * edge_length:3 * edge_length, 0:edge_length] = [0, 0, 0]
        cubemap_pixels[2 * edge_length:3 * edge_length, edge_length:2 * edge_length] = [0, 0, 0]
        cubemap_pixels[2 * edge_length:3 * edge_length, 3 * edge_length:] = [0, 0, 0]

        # Convert the numpy array back to an image
        cubemap = Image.fromarray(cubemap_pixels)
        print("[Explorer] Panorama successfully converted to cube map.")

        def extract_individual_faces():
            """
            Extract the individual cube faces from the cube map image.

            Returns:
            dict: A dictionary of individual cube faces with their names as keys.
            """
            # Define the names and coordinates for each face
            face_names = ['right', 'left', 'top', 'bottom', 'front', 'back']
            face_coordinates = [
                (edge_length * 3, edge_length, edge_length * 4, edge_length * 2),  # right
                (edge_length, edge_length, edge_length * 2, edge_length * 2),      # left
                (2 * edge_length, 0, 3 * edge_length, edge_length),                # top
                (2 * edge_length, 2 * edge_length, 3 * edge_length, 3 * edge_length),  # bottom
                (2 * edge_length, edge_length, 3 * edge_length, edge_length * 2),  # front
                (0, edge_length, edge_length, edge_length * 2)                     # back
            ]

            # Extract each face as an individual image and store it in a dictionary
            faces = {}
            for face_name, (x1, y1, x2, y2) in zip(face_names, face_coordinates):
                face_img = cubemap.crop((x1, y1, x2, y2))
                faces[face_name] = face_img
            return faces

        # Extract individual cube faces
        cubes = extract_individual_faces()
        print("[Explorer] Individual cube faces extracted successfully.")

        # Resize the cubemap back to its unscaled size
        unscaled_cubemap_size = (original_size[0], int(original_size[0] * 3 / 4))
        cubemap = cubemap.resize(unscaled_cubemap_size, Image.LANCZOS)

        return cubemap, cubes

    def precompute_rotation_matrix(self, rx, ry, rz):
        """
        Precompute a rotation matrix given rotation angles around x, y, and z axes.

        Parameters:
        rx, ry, rz: Rotation angles in degrees.

        Returns:
        numpy.ndarray: The resulting 3x3 rotation matrix.
        """
        # Convert degrees to radians
        rx = np.deg2rad(rx)
        ry = np.deg2rad(ry)
        rz = np.deg2rad(rz)
        
        # Rotation matrices for x, y, and z axes
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        R = Rz @ Ry @ Rx
        return R

    def cubemap_to_equirectangular(self, cubemap_faces, output_width, output_height, scale_factor=2):
        """
        Convert cube map images to a panorama image with optional scaling.

        Parameters:
        cubemap_faces (dict): Dictionary containing images of the cube map faces.
        output_width (int): Width of the output equirectangular image.
        output_height (int): Height of the output equirectangular image.
        scale_factor (int): Factor by which to scale the output equirectangular image for processing.

        Returns:
        Image: The equirectangular panorama image.
        """
        # Scale up the target resolution for processing
        scaled_output_width = output_width * scale_factor
        scaled_output_height = output_height * scale_factor

        # Precompute rotation matrix
        rx, ry, rz = 90, -90, 180  # Rotation parameters
        R = self.precompute_rotation_matrix(rx, ry, rz)

        # Create meshgrid for pixel coordinates
        x = np.linspace(0, scaled_output_width - 1, scaled_output_width)
        y = np.linspace(0, scaled_output_height - 1, scaled_output_height)
        xv, yv = np.meshgrid(x, y)

        # Convert equirectangular coordinates to spherical coordinates
        theta = (xv / scaled_output_width) * 2 * np.pi - np.pi
        phi = (yv / scaled_output_height) * np.pi - (np.pi / 2)

        # Convert spherical coordinates to Cartesian coordinates
        xs = np.cos(phi) * np.cos(theta)
        ys = np.cos(phi) * np.sin(theta)
        zs = np.sin(phi)

        def apply_rotation(x, y, z):
            """
            Apply a rotation matrix to 3D coordinates.

            Parameters:
            x, y, z: 3D coordinates as numpy arrays.

            Returns:
            numpy.ndarray: Rotated 3D coordinates.
            """
            return R @ np.array([x, y, z])

        # Apply precomputed rotation using the apply_rotation function
        xs, ys, zs = apply_rotation(xs.flatten(), ys.flatten(), zs.flatten())
        xs = xs.reshape((scaled_output_height, scaled_output_width))
        ys = ys.reshape((scaled_output_height, scaled_output_width))
        zs = zs.reshape((scaled_output_height, scaled_output_width))

        # Determine which face of the cubemap this point maps to
        abs_x, abs_y, abs_z = np.abs(xs), np.abs(ys), np.abs(zs)
        face_indices = np.argmax(np.stack([abs_x, abs_y, abs_z], axis=-1), axis=-1)

        equirectangular_pixels = np.zeros((scaled_output_height, scaled_output_width, 3), dtype=np.uint8)

        for face_name, face_image in cubemap_faces.items():
            face_image = np.array(face_image)
            if face_name == 'right':
                mask = (face_indices == 0) & (xs > 0)
                u = (-zs[mask] / abs_x[mask] + 1) / 2
                v = (ys[mask] / abs_x[mask] + 1) / 2
            elif face_name == 'left':
                mask = (face_indices == 0) & (xs < 0)
                u = (zs[mask] / abs_x[mask] + 1) / 2
                v = (ys[mask] / abs_x[mask] + 1) / 2
            elif face_name == 'bottom':
                mask = (face_indices == 1) & (ys > 0)
                u = (xs[mask] / abs_y[mask] + 1) / 2
                v = (-zs[mask] / abs_y[mask] + 1) / 2
            elif face_name == 'top':
                mask = (face_indices == 1) & (ys < 0)
                u = (xs[mask] / abs_y[mask] + 1) / 2
                v = (zs[mask] / abs_y[mask] + 1) / 2
            elif face_name == 'front':
                mask = (face_indices == 2) & (zs > 0)
                u = (xs[mask] / abs_z[mask] + 1) / 2
                v = (ys[mask] / abs_z[mask] + 1) / 2
            elif face_name == 'back':
                mask = (face_indices == 2) & (zs < 0)
                u = (-xs[mask] / abs_z[mask] + 1) / 2
                v = (ys[mask] / abs_z[mask] + 1) / 2

            # Convert the face u, v coordinates to pixel coordinates
            face_height, face_width, _ = face_image.shape
            u_pixel = np.clip((u * face_width).astype(int), 0, face_width - 1)
            v_pixel = np.clip((v * face_height).astype(int), 0, face_height - 1)

            # Ensure mask is correctly shaped and boolean
            mask = mask.astype(bool)

            # Create boolean indices for equirectangular assignment
            masked_yv = yv[mask]
            masked_xv = xv[mask]

            # Ensure the index arrays are integer type
            masked_yv = masked_yv.astype(int)
            masked_xv = masked_xv.astype(int)

            # Get the color from the cubemap face and set it in the equirectangular image
            equirectangular_pixels[masked_yv, masked_xv] = face_image[v_pixel, u_pixel]

        # Convert the numpy array back to an image
        equirectangular_image = Image.fromarray(equirectangular_pixels)

        # Resize back to the desired output size
        if scale_factor > 1:
            equirectangular_image = equirectangular_image.resize((output_width, output_height), Image.LANCZOS)

        print("[Explorer] Cube map successfully converted to equirectangular panorama.")

        return equirectangular_image

    def extract_normal_view(self, panorama, fov_h, fov_v, center_yaw, center_pitch, center_roll, output_width, output_height):
        """
        Extract a rectilinear view from an equirectangular panorama with proper rotation order.
        """
        if isinstance(panorama, Image.Image):
            panorama = np.array(panorama)

        pano_height, pano_width = panorama.shape[:2]

        # Convert angles to radians
        yaw, pitch, roll = np.deg2rad([center_yaw, center_pitch, center_roll])
        fov_h_rad = np.deg2rad(fov_h)
        fov_v_rad = np.deg2rad(fov_v)

        # Create rotation matrices with correct order: roll -> pitch -> yaw
        # Roll (around viewing direction / z-axis)
        Rroll = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])

        # Pitch (around x-axis)
        Rpitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])

        # Yaw (around y-axis)
        Ryaw = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        # Combined rotation matrix: apply roll first, then pitch, then yaw
        R = Ryaw @ Rpitch @ Rroll

        # Create viewing grid
        x = np.linspace(-np.tan(fov_h_rad / 2), np.tan(fov_h_rad / 2), output_width)
        y = np.linspace(np.tan(fov_v_rad / 2), -np.tan(fov_v_rad / 2), output_height)
        x_grid, y_grid = np.meshgrid(x, y)

        # Create normalized direction vectors
        z = np.ones_like(x_grid)
        vectors = np.stack([x_grid, y_grid, z], axis=-1)
        norms = np.sqrt(np.sum(vectors**2, axis=2, keepdims=True))
        vectors = vectors / norms

        # Reshape for matrix multiplication
        vectors = vectors.reshape(-1, 3)

        # Apply rotation
        rotated_vectors = vectors @ R.T

        # Convert to spherical coordinates
        x_rot, y_rot, z_rot = rotated_vectors[:, 0], rotated_vectors[:, 1], rotated_vectors[:, 2]

        # Calculate spherical coordinates
        lon = np.arctan2(x_rot, z_rot)
        lat = np.arcsin(np.clip(y_rot, -1, 1))

        # Convert to panorama pixel coordinates
        px = ((lon + np.pi) / (2 * np.pi)) * pano_width
        py = ((np.pi/2 - lat) / np.pi) * pano_height

        # Reshape for remap
        px = px.reshape(output_height, output_width).astype(np.float32)
        py = py.reshape(output_height, output_width).astype(np.float32)

        # Ensure pixel indices are within bounds
        px = np.clip(px, 0, pano_width - 1)
        py = np.clip(py, 0, pano_height - 1)

        # Use OpenCV remap to sample pixels
        output_image = cv2.remap(panorama, px, py, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        print(f"[Explorer] Normal view extracted with FOV: {fov_h}x{fov_v}, rotation: {center_yaw},{center_pitch},{center_roll}.")

        return Image.fromarray(output_image)

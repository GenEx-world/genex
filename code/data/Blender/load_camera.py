import bpy
import math
import time
import os

# Global parameters
CAMERA_COLLECTION_NAME = "cameras"
DEFAULT_FRAME_START = 1
DEFAULT_FRAME_END = 30
CAMERA_POSITIONS = ['Front', 'Left', 'Back', 'Right', 'Bottom', 'Top']
CAMERA_ROTATIONS = {
    'Front': (90, 0, 0),
    'Left': (90, 0, 90),
    'Back': (90, 0, 180),
    'Right': (90, 0, 270),
    'Bottom': (0, 0, 0),
    'Top': (180, 0, 0),
}

def initialize_camera(renderer_type):
    """
    Initialize cameras for rendering based on renderer type.
    
    Args:
        renderer_type (str): 'EEVEE' or 'CYCLES'
        
    Returns:
        dict: Dictionary of camera objects by position name
    """
    original_collection = bpy.context.view_layer.active_layer_collection
    cameras = {}
    
    # Create or get the camera collection
    if CAMERA_COLLECTION_NAME not in bpy.data.collections:
        new_collection = bpy.data.collections.new(CAMERA_COLLECTION_NAME)
        bpy.context.scene.collection.children.link(new_collection)
        print(f"Collection '{CAMERA_COLLECTION_NAME}' created.")
    else:
        new_collection = bpy.data.collections[CAMERA_COLLECTION_NAME]
        print(f"Collection '{CAMERA_COLLECTION_NAME}' already exists.")
    
    # Create cameras based on renderer type
    if renderer_type == 'EEVEE':
        # Create perspective cameras for multiple views
        for position in CAMERA_POSITIONS:
            camera_name = f'{position}Camera'
            camera_data = bpy.data.cameras.new(name=camera_name)
            camera_data.type = 'PERSP'
            camera_data.angle = math.radians(90)  # 90 degree FOV
            camera_object = bpy.data.objects.new(camera_name, camera_data)
            new_collection.objects.link(camera_object)
            cameras[position] = camera_object
            print(f"Camera '{camera_name}' created for {position} view.")
            
    elif renderer_type == 'CYCLES':
        # Create panoramic camera (only need one)
        position = 'Front'
        camera_name = f'{position}Camera'
        camera_data = bpy.data.cameras.new(name=camera_name)
        camera_data.type = 'PANO'
        camera_data.panorama_type = 'EQUIRECTANGULAR'
        camera_object = bpy.data.objects.new(camera_name, camera_data)
        new_collection.objects.link(camera_object)
        cameras[position] = camera_object
        print(f"Panoramic camera '{camera_name}' created.")
        
    else:
        raise ValueError('Renderer must be EEVEE OR CYCLES')

    # Restore original collection
    bpy.context.view_layer.active_layer_collection = original_collection
        
    return cameras


def set_render(renderer_type, frame_start=DEFAULT_FRAME_START, frame_end=DEFAULT_FRAME_END):
    """
    Configure render settings based on renderer type.
    
    Args:
        renderer_type (str): 'EEVEE' or 'CYCLES'
        frame_start (int): First frame of animation
        frame_end (int): Last frame of animation
    """
    # Set frame range
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end

    if renderer_type == 'CYCLES':
        # Configure Cycles renderer (GPU accelerated)
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        
        # Set up CUDA compute devices
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            device.use = True
        
        # Quality settings
        bpy.context.scene.cycles.samples = 512
        bpy.context.scene.cycles.use_adaptive_sampling = True
        bpy.context.scene.cycles.adaptive_threshold = 0.05
        bpy.context.scene.cycles.max_bounces = 8
        bpy.context.scene.cycles.min_bounces = 3
        bpy.context.scene.cycles.caustics_reflective = True
        bpy.context.scene.cycles.caustics_refractive = True
        
        # Resolution (16:9 aspect ratio)
        bpy.context.scene.render.resolution_x = 1024
        bpy.context.scene.render.resolution_y = 576
        bpy.context.scene.render.image_settings.color_mode = 'RGB'
        
    else:
        # Configure Eevee renderer (faster)
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.eevee.taa_render_samples = 16
        
        # Output settings
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.resolution_x = 500
        bpy.context.scene.render.resolution_y = 500


def set_camera(cameras, start, end, initial_rotation):
    """
    Set up camera animation from start to end position.
    
    Args:
        cameras (dict): Dictionary of camera objects
        start (tuple): Start position (x, y, z)
        end (tuple): End position (x, y, z)
        initial_rotation (tuple): Additional rotation to apply (x, y, z) in degrees
    """
    for face, camera_object in cameras.items():    
        # Apply base rotation for the camera face, plus the additional rotation
        rotation = tuple(map(sum, zip(CAMERA_ROTATIONS[face], initial_rotation)))
        camera_object.rotation_euler = tuple(map(math.radians, rotation))

        # Set start position and keyframe
        camera_object.location = start
        camera_object.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_start)

        # Set end position and keyframe
        camera_object.location = end
        camera_object.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_end)

        # Set keyframe interpolation to linear
        if camera_object.animation_data and camera_object.animation_data.action:
            for fcurve in camera_object.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = 'LINEAR'


def render_video(renderer_type, cameras, output_path):
    """
    Render frames using configured cameras.
    
    Args:
        renderer_type (str): 'EEVEE' or 'CYCLES'
        cameras (dict): Dictionary of camera objects
        output_path (str): Path to save rendered frames
    """
    # Start timing the render
    start_time = time.time()
    
    # Render each frame
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(frame)

        if renderer_type == 'CYCLES':
            # For Cycles, render one panoramic view per frame
            for face, camera in cameras.items():
                bpy.context.scene.camera = camera
                bpy.context.scene.render.filepath = os.path.join(output_path, f'video_frame{frame-1}.png')
                bpy.ops.render.render(write_still=True)
        else:
            # For Eevee, render multiple camera perspectives per frame
            frame_path = os.path.join(output_path, f'frame{str(frame)}')
            os.makedirs(frame_path, exist_ok=True)
            
            for face, camera in cameras.items():
                bpy.context.scene.camera = camera
                bpy.context.scene.render.filepath = os.path.join(frame_path, f'{face.lower()}.png')
                bpy.ops.render.render(write_still=True)
    
    # Report render time
    end_time = time.time()
    render_duration = end_time - start_time
    
    print(f"Rendering completed and saved to '{output_path}'.")
    print(f"Render time: {render_duration:.2f} seconds")
    
    # Clean up unused data
    bpy.ops.outliner.orphans_purge()
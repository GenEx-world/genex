import numpy as np
from diffusers import FluxFillPipeline
from PIL import Image

class GenExWorldInitializerPipeline(FluxFillPipeline):
    def precompute_rotation_matrix(self, rx, ry, rz):
        rx = np.deg2rad(rx)
        ry = np.deg2rad(ry)
        rz = np.deg2rad(rz)
        
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
        
        R = Rz @ Ry @ Rx
        return R
    
    def cubemap_to_equirectangular(self, cubemap_faces, output_width, output_height, scale_factor=2):
        scaled_output_width = output_width * scale_factor
        scaled_output_height = output_height * scale_factor
        
        rx, ry, rz = 90, -90, 180  
        R = self.precompute_rotation_matrix(rx, ry, rz)
        
        x = np.linspace(0, scaled_output_width - 1, scaled_output_width)
        y = np.linspace(0, scaled_output_height - 1, scaled_output_height)
        xv, yv = np.meshgrid(x, y)
        
        theta = (xv / scaled_output_width) * 2 * np.pi - np.pi
        phi = (yv / scaled_output_height) * np.pi - (np.pi / 2)
        
        xs = np.cos(phi) * np.cos(theta)
        ys = np.cos(phi) * np.sin(theta)
        zs = np.sin(phi)
        
        def apply_rotation(x, y, z):
            return R @ np.array([x, y, z])
        
        xs, ys, zs = apply_rotation(xs.flatten(), ys.flatten(), zs.flatten())
        xs = xs.reshape((scaled_output_height, scaled_output_width))
        ys = ys.reshape((scaled_output_height, scaled_output_width))
        zs = zs.reshape((scaled_output_height, scaled_output_width))
        
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
        
            face_height, face_width, _ = face_image.shape
            u_pixel = np.clip((u * face_width).astype(int), 0, face_width - 1)
            v_pixel = np.clip((v * face_height).astype(int), 0, face_height - 1)
        
            mask = mask.astype(bool)
        
            masked_yv = yv[mask]
            masked_xv = xv[mask]
        
            masked_yv = masked_yv.astype(int)
            masked_xv = masked_xv.astype(int)
        
            equirectangular_pixels[masked_yv, masked_xv] = face_image[v_pixel, u_pixel]
        
        equirectangular_image = Image.fromarray(equirectangular_pixels)
        
        if scale_factor > 1:
            equirectangular_image = equirectangular_image.resize((output_width, output_height), Image.LANCZOS)
        
        return equirectangular_image
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        side = min(w, h)
        left = (w - side) // 2
        top  = (h - side) // 2
        img = image.crop((left, top, left + side, top + side))
        front = img.resize((512, 512))
    
        cubes = {}
        cubes['front'] = front
        cubes['back'] = Image.new("RGB", (512, 512), (255, 255, 255))
        cubes['left'] = Image.new("RGB", (512, 512), (255, 255, 255))
        cubes['right'] = Image.new("RGB", (512, 512), (255, 255, 255))
        cubes['top'] = Image.new("RGB", (512, 512), (255, 255, 255))
        cubes['bottom'] = Image.new("RGB", (512, 512), (255, 255, 255))
    
        input_panorama = self.cubemap_to_equirectangular(cubes, 2048, 1024, scale_factor=2)
    
        return front, input_panorama
            
        
    def preprocess_mask(self) -> Image.Image:
        mask = Image.open("pano_mask.png").convert("L")
        return mask.resize((2048, 1024))

    def create_mask(self) -> Image.Image:
        cubes = {}
        cubes['front'] = Image.new("RGB", (512, 512), (0, 0, 0))
        cubes['back'] = Image.new("RGB", (512, 512), (255, 255, 255))
        cubes['left'] = Image.new("RGB", (512, 512), (255, 255, 255))
        cubes['right'] = Image.new("RGB", (512, 512), (255, 255, 255))
        cubes['top'] = Image.new("RGB", (512, 512), (255, 255, 255))
        cubes['bottom'] = Image.new("RGB", (512, 512), (255, 255, 255))
        
        mask = self.cubemap_to_equirectangular(cubes, 2048, 1024, scale_factor=1)

        mask = mask.convert("L")
    
        return mask
    
    
    def __call__(
        self,
        image: Image.Image,
        prompt: str = None,
        guidance_scale: float = 3.5,
    ):
        front, img   = self.preprocess_image(image)
        # mask  = self.preprocess_mask()
        mask  = self.create_mask()

        
        if prompt:
            prompt = 'GenEx Panoramic World Initialization, ' + prompt
        else:
            prompt = 'GenEx Panoramic World Initialization'
    
        return front, super().__call__(
            prompt=prompt,
            image=img,
            mask_image=mask,
            guidance_scale=guidance_scale,
            width=2048,
            height=1024,
        )
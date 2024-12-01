import cv2
import numpy as np
from PIL import Image

def create_circular_mask(size, radius_reduction):
    mask = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    radius = int((size // 2) - radius_reduction) 
    if radius < 1:
        radius = 1  
    cv2.circle(mask, (center, center), radius, (255), thickness=-1)
    return mask

def draw_black_circle(image, center, radius):
    cv2.circle(image, center, radius, (0, 0, 0, 255), thickness=-1)

def draw_small_white_circle(image, center, radius):
    cv2.circle(image, center, radius, (255, 255, 255, 255), thickness=-1)

def create_mosaic(target_image_path, eye_image_path, output_image_path, eye_size, radius_reduction):
    target_image = Image.open(target_image_path).convert('RGBA')
    target_width, target_height = target_image.size

    eye_image = cv2.imread(eye_image_path, cv2.IMREAD_UNCHANGED)

    if eye_image is None:
        raise FileNotFoundError(f"Unable to load eye image from path: {eye_image_path}")
        
    circular_mask = create_circular_mask(eye_size, radius_reduction)

    eye_image_resized = cv2.resize(eye_image, (eye_size, eye_size), interpolation=cv2.INTER_AREA)

    mosaic_image = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 0))

    black_circle_radius = eye_size // 6  
    small_white_circle_radius = black_circle_radius // 2 
    center = (eye_size // 2, eye_size // 2)
    small_circle_center = (center[0] - black_circle_radius + small_white_circle_radius,
                           center[1] - black_circle_radius + small_white_circle_radius)

    step_size = eye_size * 1.5 

    for y in range(0, target_height, int(step_size)):
        for x in range(0, target_width, int(step_size)):

            target_region = target_image.crop((x, y, x + eye_size, y + eye_size))
            target_region_array = np.array(target_region)


            masked_eye_image = cv2.bitwise_and(eye_image_resized, eye_image_resized, mask=circular_mask)

 
            if eye_image_resized.shape[2] == 3:
                masked_eye_image = np.dstack([masked_eye_image, circular_mask])
            else:
                masked_eye_image = np.dstack([masked_eye_image[:, :, :3], circular_mask])


            masked_eye_image_pil = Image.fromarray(cv2.cvtColor(masked_eye_image, cv2.COLOR_BGRA2RGBA))

    
            for i in range(eye_size):
                for j in range(eye_size):
                    if circular_mask[i, j] == 255:  
                        masked_eye_image_pil.putpixel((i, j), tuple(target_region_array[j, i]))


            masked_eye_image_array = np.array(masked_eye_image_pil)

            draw_black_circle(masked_eye_image_array, center, black_circle_radius)

            draw_small_white_circle(masked_eye_image_array, small_circle_center, small_white_circle_radius)

            masked_eye_image_pil = Image.fromarray(masked_eye_image_array)

            mosaic_image.paste(masked_eye_image_pil, (x, y), masked_eye_image_pil)

    mosaic_image.save(output_image_path, quality=100)

create_mosaic("./target_image.jpg", './eye_image.png', './mosaic_output.png', 30, -1)

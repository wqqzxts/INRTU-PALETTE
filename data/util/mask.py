# Copyright (c) OpenMMLab. All rights reserved.
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw
import random
import os


def random_cropping_bbox(img_shape=(256,256), mask_mode='onedirection'):
    h, w = img_shape
    if mask_mode == 'onedirection':
        _type = np.random.randint(0, 4)
        if _type == 0:
            top, left, height, width = 0, 0, h, w//2
        elif _type == 1:
            top, left, height, width = 0, 0, h//2, w
        elif _type == 2:
            top, left, height, width = h//2, 0, h//2, w
        elif _type == 3:
            top, left, height, width = 0, w//2, h, w//2
    else:
        target_area = (h*w)//2
        width = np.random.randint(target_area//h, w)
        height = target_area//width
        if h==height:
            top = 0
        else:
            top = np.random.randint(0, h-height)
        if w==width:
            left = 0
        else:
            left = np.random.randint(0, w-width)
    return (top, left, height, width)

def random_bbox(img_shape=(256,256), max_bbox_shape=(128, 128), max_bbox_delta=40, min_margin=20):
    """Generate a random bbox for the mask on a given image.

    In our implementation, the max value cannot be obtained since we use
    `np.random.randint`. And this may be different with other standard scripts
    in the community.

    Args:
        img_shape (tuple[int]): The size of a image, in the form of (h, w).
        max_bbox_shape (int | tuple[int]): Maximum shape of the mask box,
            in the form of (h, w). If it is an integer, the mask box will be
            square.
        max_bbox_delta (int | tuple[int]): Maximum delta of the mask box,
            in the form of (delta_h, delta_w). If it is an integer, delta_h
            and delta_w will be the same. Mask shape will be randomly sampled
            from the range of `max_bbox_shape - max_bbox_delta` and
            `max_bbox_shape`. Default: (40, 40).
        min_margin (int | tuple[int]): The minimum margin size from the
            edges of mask box to the image boarder, in the form of
            (margin_h, margin_w). If it is an integer, margin_h and margin_w
            will be the same. Default: (20, 20).

    Returns:
        tuple[int]: The generated box, (top, left, h, w).
    """
    if not isinstance(max_bbox_shape, tuple):
        max_bbox_shape = (max_bbox_shape, max_bbox_shape)
    if not isinstance(max_bbox_delta, tuple):
        max_bbox_delta = (max_bbox_delta, max_bbox_delta)
    if not isinstance(min_margin, tuple):
        min_margin = (min_margin, min_margin)
        
    img_h, img_w = img_shape[:2]
    max_mask_h, max_mask_w = max_bbox_shape
    max_delta_h, max_delta_w = max_bbox_delta
    margin_h, margin_w = min_margin

    if max_mask_h > img_h or max_mask_w > img_w:
        raise ValueError(f'mask shape {max_bbox_shape} should be smaller than '
                         f'image shape {img_shape}')
    if (max_delta_h // 2 * 2 >= max_mask_h
            or max_delta_w // 2 * 2 >= max_mask_w):
        raise ValueError(f'mask delta {max_bbox_delta} should be smaller than'
                         f'mask shape {max_bbox_shape}')
    if img_h - max_mask_h < 2 * margin_h or img_w - max_mask_w < 2 * margin_w:
        raise ValueError(f'Margin {min_margin} cannot be satisfied for img'
                         f'shape {img_shape} and mask shape {max_bbox_shape}')

    # get the max value of (top, left)
    max_top = img_h - margin_h - max_mask_h
    max_left = img_w - margin_w - max_mask_w
    # randomly select a (top, left)
    top = np.random.randint(margin_h, max_top)
    left = np.random.randint(margin_w, max_left)
    # randomly shrink the shape of mask box according to `max_bbox_delta`
    # the center of box is fixed
    delta_top = np.random.randint(0, max_delta_h // 2 + 1)
    delta_left = np.random.randint(0, max_delta_w // 2 + 1)
    top = top + delta_top
    left = left + delta_left
    h = max_mask_h - delta_top
    w = max_mask_w - delta_left
    return (top, left, h, w)


def bbox2mask(img_shape, bbox, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 1

    return mask


def brush_stroke_mask(img_shape,
                      num_vertices=(4, 12),
                      mean_angle=2 * math.pi / 5,
                      angle_range=2 * math.pi / 15,
                      brush_width=(12, 40),
                      max_loops=4,
                      dtype='uint8'):
    """Generate free-form mask.

    The method of generating free-form mask is in the following paper:
    Free-Form Image Inpainting with Gated Convolution.

    When you set the config of this type of mask. You may note the usage of
    `np.random.randint` and the range of `np.random.randint` is [left, right).

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    TODO: Rewrite the implementation of this function.

    Args:
        img_shape (tuple[int]): Size of the image.
        num_vertices (int | tuple[int]): Min and max number of vertices. If
            only give an integer, we will fix the number of vertices.
            Default: (4, 12).
        mean_angle (float): Mean value of the angle in each vertex. The angle
            is measured in radians. Default: 2 * math.pi / 5.
        angle_range (float): Range of the random angle.
            Default: 2 * math.pi / 15.
        brush_width (int | tuple[int]): (min_width, max_width). If only give
            an integer, we will fix the width of brush. Default: (12, 40).
        max_loops (int): The max number of for loops of drawing strokes.
        dtype (str): Indicate the data type of returned masks.
            Default: 'uint8'.

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    img_h, img_w = img_shape[:2]
    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        raise TypeError('The type of num_vertices should be int'
                        f'or tuple[int], but got type: {num_vertices}')

    if isinstance(brush_width, tuple):
        min_width, max_width = brush_width
    elif isinstance(brush_width, int):
        min_width, max_width = brush_width, brush_width + 1
    else:
        raise TypeError('The type of brush_width should be int'
                        f'or tuple[int], but got type: {brush_width}')

    average_radius = math.sqrt(img_h * img_h + img_w * img_w) / 8
    mask = Image.new('L', (img_w, img_h), 0)

    loop_num = np.random.randint(1, max_loops)
    num_vertex_list = np.random.randint(
        min_num_vertices, max_num_vertices, size=loop_num)
    angle_min_list = np.random.uniform(0, angle_range, size=loop_num)
    angle_max_list = np.random.uniform(0, angle_range, size=loop_num)

    for loop_n in range(loop_num):
        num_vertex = num_vertex_list[loop_n]
        angle_min = mean_angle - angle_min_list[loop_n]
        angle_max = mean_angle + angle_max_list[loop_n]
        angles = []
        vertex = []

        # set random angle on each vertex
        angles = np.random.uniform(angle_min, angle_max, size=num_vertex)
        reverse_mask = (np.arange(num_vertex, dtype=np.float32) % 2) == 0
        angles[reverse_mask] = 2 * math.pi - angles[reverse_mask]

        h, w = mask.size

        # set random vertices
        vertex.append((np.random.randint(0, w), np.random.randint(0, h)))
        r_list = np.random.normal(
            loc=average_radius, scale=average_radius // 2, size=num_vertex)
        for i in range(num_vertex):
            r = np.clip(r_list[i], 0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))
        # draw brush strokes according to the vertex and angle list
        draw = ImageDraw.Draw(mask)
        width = np.random.randint(min_width, max_width)
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2, v[1] - width // 2,
                          v[0] + width // 2, v[1] + width // 2),
                         fill=1)
    # randomly flip the mask
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.array(mask).astype(dtype=getattr(np, dtype))
    mask = mask[:, :, None]
    return mask


def random_irregular_mask(img_shape,
                          num_vertices=(4, 8),
                          max_angle=4,
                          length_range=(10, 100),
                          brush_width=(10, 40),
                          dtype='uint8'):
    """Generate random irregular masks.

    This is a modified version of free-form mask implemented in
    'brush_stroke_mask'.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    TODO: Rewrite the implementation of this function.

    Args:
        img_shape (tuple[int]): Size of the image.
        num_vertices (int | tuple[int]): Min and max number of vertices. If
            only give an integer, we will fix the number of vertices.
            Default: (4, 8).
        max_angle (float): Max value of angle at each vertex. Default 4.0.
        length_range (int | tuple[int]): (min_length, max_length). If only give
            an integer, we will fix the length of brush. Default: (10, 100).
        brush_width (int | tuple[int]): (min_width, max_width). If only give
            an integer, we will fix the width of brush. Default: (10, 40).
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    h, w = img_shape[:2]

    mask = np.zeros((h, w), dtype=dtype)
    if isinstance(length_range, int):
        min_length, max_length = length_range, length_range + 1
    elif isinstance(length_range, tuple):
        min_length, max_length = length_range
    else:
        raise TypeError('The type of length_range should be int'
                        f'or tuple[int], but got type: {length_range}')
    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        raise TypeError('The type of num_vertices should be int'
                        f'or tuple[int], but got type: {num_vertices}')

    if isinstance(brush_width, int):
        min_brush_width, max_brush_width = brush_width, brush_width + 1
    elif isinstance(brush_width, tuple):
        min_brush_width, max_brush_width = brush_width
    else:
        raise TypeError('The type of brush_width should be int'
                        f'or tuple[int], but got type: {brush_width}')

    num_v = np.random.randint(min_num_vertices, max_num_vertices)

    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        # from the start point, randomly setlect n \in [1, 6] directions.
        direction_num = np.random.randint(1, 6)
        angle_list = np.random.randint(0, max_angle, size=direction_num)
        length_list = np.random.randint(
            min_length, max_length, size=direction_num)
        brush_width_list = np.random.randint(
            min_brush_width, max_brush_width, size=direction_num)
        for direct_n in range(direction_num):
            angle = 0.01 + angle_list[direct_n]
            if i % 2 == 0:
                angle = 2 * math.pi - angle
            length = length_list[direct_n]
            brush_w = brush_width_list[direct_n]
            # compute end point according to the random angle
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1, brush_w)
            start_x, start_y = end_x, end_y
    mask = np.expand_dims(mask, axis=2)

    return mask


def get_irregular_mask(img_shape, area_ratio_range=(0.15, 0.5), **kwargs):
    """Get irregular mask with the constraints in mask ratio

    Args:
        img_shape (tuple[int]): Size of the image.
        area_ratio_range (tuple(float)): Contain the minimum and maximum area
        ratio. Default: (0.15, 0.5).

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    mask = random_irregular_mask(img_shape, **kwargs)
    min_ratio, max_ratio = area_ratio_range

    while not min_ratio < (np.sum(mask) /
                           (img_shape[0] * img_shape[1])) < max_ratio:
        mask = random_irregular_mask(img_shape, **kwargs)

    return mask


def create_mask_from_image(img_shape):
    """
    Creates a mask array from an image by randomly selecting a mask file from a directory.
    The mask is inverted (black becomes white and white becomes black) after conversion to grayscale.
    
    Args:
        img_shape (tuple): Shape of the target image (height, width, channels). Not used directly but could be for validation.
    
    Returns:
        numpy.ndarray: Binary mask of shape (height, width, 1), where 1 indicates missing pixels.
    """
    mask_dir = "/home/jovyan/cloud/Palette/INRTU-PALETTE/data/util/masks"
    
    # Get list of all mask files
    mask_files = [f for f in os.listdir(mask_dir) if f.startswith('MASK_') and f.endswith('.png')]
    
    if not mask_files:
        raise ValueError("No mask files found in the directory")
    
    # Randomly select a mask file
    selected_mask = random.choice(mask_files)
    image_path = os.path.join(mask_dir, selected_mask)
    
    # Load the image
    image = Image.open(image_path)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Invert the image (black becomes white and white becomes black)
    image = Image.eval(image, lambda x: 255 - x)
    
    # Threshold the image to create a binary mask
    mask = np.array(image) > 0
    mask = mask.astype(np.uint8)
    
    # Add the channel dimension
    mask = mask[:, :, np.newaxis]
    
    return mask


def get_mask_from_image_white_pixels(image_size, image_path):
    """
    Advanced white pixel detection for IPFZ maps.
    Specifically targets pure white pixels (255, 255, 255).
    """    
    original_img = Image.open(image_path).convert('RGB')
    
    # FIX: Correct the resize dimensions - PIL uses (width, height)
    original_img = original_img.resize((image_size[1], image_size[0]))  # This is correct: (width, height)
    
    img_array = np.array(original_img)    
    
    # Method 1: Exact white pixel detection (255, 255, 255)
    exact_white_mask = np.all(img_array == 255, axis=2).astype(np.uint8)
    
    # Method 2: Near-white detection with very tight tolerance
    white_reference = np.array([255, 255, 255])
    color_distance = np.sqrt(np.sum((img_array.astype(float) - white_reference) ** 2, axis=2))
    near_white_mask = (color_distance < 5).astype(np.uint8)
    
    # Combine both methods
    mask = exact_white_mask | near_white_mask
    
    # Debug output
    white_pixel_count = np.sum(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    white_percentage = (white_pixel_count / total_pixels) * 100
    
    print(f"White pixels detected: {white_pixel_count}/{total_pixels} ({white_percentage:.2f}%) in {os.path.basename(image_path)}")
    print(f"Image shape: {img_array.shape}, Mask shape: {mask.shape}")
    
    # Clean up the mask with morphological operations
    if np.any(mask):
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Fallback if no white pixels found
    if not np.any(mask):
        print(f"Warning: No white pixels found in {os.path.basename(image_path)}, using fallback mask")
        mask = bbox2mask(image_size, random_bbox(max_bbox_shape=(32, 32)))
    
    # FIX: Ensure mask has the same spatial dimensions as the image
    # Add channel dimension
    mask = mask[:, :, np.newaxis]
    
    # FIX: Double-check the dimensions match the expected image_size
    if mask.shape[0] != image_size[0] or mask.shape[1] != image_size[1]:
        print(f"Warning: Mask shape {mask.shape} doesn't match image_size {image_size}")
        # Resize mask to match exactly
        mask_resized = cv2.resize(mask, (image_size[1], image_size[0]))
        mask = mask_resized[:, :, np.newaxis]
    
    return mask
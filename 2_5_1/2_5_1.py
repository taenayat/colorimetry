import numpy as np
from skimage import io, img_as_float
import matplotlib.pyplot as plt
from os import listdir

def srgb_to_xyz(srgb):
    srgb = np.clip(srgb, 0, 1)
    a = 0.055
    threshold = 0.04045
    linear_rgb = np.where(srgb <= threshold, srgb / 12.92, ((srgb + a) / (1 + a)) ** 2.4)
    
    M_sRGB_to_XYZ = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])
    xyz = np.dot(linear_rgb, M_sRGB_to_XYZ.T)
    return xyz

def xyz_to_srgb(xyz):
    # Matrix to convert XYZ to linear RGB
    M_XYZ_to_sRGB = np.array([
        [ 3.2410, -1.5374, -0.4986],
        [-0.9692,  1.8760,  0.0416],
        [ 0.0556, -0.2040,  1.0570]
    ])
    # Convert XYZ to linear RGB
    linear_rgb = np.dot(xyz, M_XYZ_to_sRGB.T)
    
    # Apply the gamma correction to convert linear RGB to sRGB
    a = 0.055
    threshold = 0.0031308
    srgb = np.where(linear_rgb <= threshold, 12.92 * linear_rgb, (1 + a) * (linear_rgb ** (1 / 2.4)) - a)
    
    # Ensure sRGB values are in the range [0, 1]
    srgb = np.clip(srgb, 0, 1)
    
    return srgb

# D65 white point
D65_XYZ = np.array([95.047, 100.00, 108.883])/100

def von_kries_transform(source_white, target_white):
    cone_response_matrix = np.array([
        [0.40024, 0.7076, -0.08081],   # L cone response
        [-0.2263, 1.16532, 0.0457],    # M cone response
        [0, 0, 0.91822]                # S cone response
    ])
    
    inv_cone_response_matrix = np.linalg.inv(cone_response_matrix)
    source_cones = np.dot(cone_response_matrix, source_white)
    target_cones = np.dot(cone_response_matrix, target_white)
    scaling_factors = target_cones / source_cones
    scaling_matrix = np.diag(scaling_factors)
    transformation_matrix = np.dot(np.dot(inv_cone_response_matrix, scaling_matrix), cone_response_matrix)
    return transformation_matrix


def cat02_transform(source_white, target_white):
    cat02_matrix = np.array([
        [ 0.7328,  0.4296, -0.1624],
        [-0.7036,  1.6975,  0.0061],
        [ 0.0030,  0.0136,  0.9834]
    ])
    inv_cat02_matrix = np.linalg.inv(cat02_matrix)
    source_cones = np.dot(cat02_matrix, source_white)
    target_cones = np.dot(cat02_matrix, target_white)
    scaling_factors = target_cones / source_cones
    scaling_matrix = np.diag(scaling_factors)
    transformation_matrix = np.dot(np.dot(inv_cat02_matrix, scaling_matrix), cat02_matrix)
    return transformation_matrix


def method_1_white_point(image):
    xyz_to_lms_matrix = np.array([
        [ 0.4002, 0.7075, -0.0808],
        [-0.2263, 1.1653,  0.0457],
        [ 0.0000, 0.0000,  0.9182]
    ])
    image_xyz = srgb_to_xyz(image)
    image_lms = np.dot(image_xyz, xyz_to_lms_matrix.T)
    L, M, S = image_lms[:, :, 0], image_lms[:, :, 1], image_lms[:, :, 2]
    achromatic_channel = 2 * L + M + (1 / 20) * S
    brightest_index = np.unravel_index(np.argmax(achromatic_channel, axis=None), achromatic_channel.shape)
    brightest_xyz = image_xyz[brightest_index]
    return brightest_xyz

def method_2_white_point(white_pixel):
    return srgb_to_xyz(white_pixel.reshape(1, 1, 3)).reshape(3)
    # return rgb2xyz(white_pixel.reshape(1, 1, 3)).reshape(3)

def method_3_white_point(image):
    avg_rgb = np.mean(image, axis=(0, 1))
    avg_xyz = srgb_to_xyz(avg_rgb.reshape(1, 1, 3)).reshape(3)
    x, y = avg_xyz[0] / np.sum(avg_xyz), avg_xyz[1] / np.sum(avg_xyz)
    return np.array([x / y, 1, (1 - x - y) / y])
    # return np.array([x / y * 100, 100, (1 - x - y) / y * 100])

def apply_white_balance(image, method, white_point=None):
    if method == 1:
        wp_source = method_1_white_point(image)
    elif method == 2 and white_point is not None:
        wp_source = method_2_white_point(white_point)
    elif method == 3:
        wp_source = method_3_white_point(image)
    else:
        raise ValueError("Invalid method or missing white point for method 2")

    return wp_source

# Chromatic adaptation
def adapt_image(image, transform):
    xyz_img = srgb_to_xyz(image) 
    adapted_img = np.dot(xyz_img.reshape(-1, 3), transform.T).reshape(xyz_img.shape)
    adapted_img = np.clip(adapted_img, 0, None)  # Ensure no negative values
    return xyz_to_srgb(adapted_img) 

def white_balance(image, method, transform_method, white_pixel=None):
    wp_source = apply_white_balance(image, method, white_pixel)
    if transform_method == "von_kries":
        transform = von_kries_transform(wp_source, D65_XYZ)
    elif transform_method == "cat02":
        transform = cat02_transform(wp_source, D65_XYZ)
    else:
        raise ValueError("Invalid transform method")
    return adapt_image(image, transform)


def display_images(axs, row_idx, original_image, method_1_vk, method_1_cat02, method_2_vk, method_2_cat02, method_3_vk, method_3_cat02):
    
    titles = ['Original', 'Method 1 VK', 'Method 1 CAT02', 'Method 2 VK', 'Method 2 CAT02', 'Method 3 VK', 'Method 3 CAT02']
    images = [original_image, method_1_vk, method_1_cat02, method_2_vk, method_2_cat02, method_3_vk, method_3_cat02]
    
    # for ax, img, title in zip(axs, images, titles):
    for col, (img, title) in enumerate(zip(images, titles)):
        axs[row_idx, col].imshow(img)
        if row_idx == 0:
            axs[row_idx, col].set_title(title)
        axs[row_idx, col].axis('off')
    


folder_dir = "2_5_1/"
image_paths = []
for image in listdir(folder_dir):
    if (image.startswith('Lab2_')):
        image_paths.append(folder_dir+image)

n_image = len(image_paths)
white_point_list = [[182,193,176], [253,172,78], [190,192,176], [234, 224, 48], 
                    [93,226,0], [247,111,0], [64,0,191], [240,143,92], [47,165,175]]
fig, axs = plt.subplots(n_image, 7, figsize=(20, 10 * n_image))
for row_idx, (path, white_point) in enumerate(zip(image_paths, white_point_list)):
    image = img_as_float(io.imread(path))
    image = image[:,:,:3]
    print(path, image.shape)
    white_point = np.array(white_point)/255
    method1_vk = white_balance(image, 1, "von_kries")
    method1_cat = white_balance(image, 1, "cat02")
    method2_vk = white_balance(image, 2, "von_kries", white_point)
    method2_cat = white_balance(image, 2, "cat02", white_point)
    method3_vk = white_balance(image, 3, "von_kries")
    method3_cat = white_balance(image, 3, "cat02")
    display_images(axs, row_idx, image, method1_vk, method1_cat, method2_vk, method2_cat, method3_vk, method3_cat)
plt.show()


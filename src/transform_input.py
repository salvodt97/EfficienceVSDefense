import cv2, numpy as np

def transform_image(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    translation_matrix = np.float32([[1, 0, 2], [0, 1, 2]])
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[0], image.shape[1]))
    rotation_matrix = cv2.getRotationMatrix2D((image.shape[0] / 2, image.shape[1] / 2), 45, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[0], image.shape[1]))
    flipped_image = cv2.flip(image, 1)
    scaling_matrix = np.float32([[0.8, 0, 0], [0, 0.8, 0]])
    scaled_image = cv2.warpAffine(image, scaling_matrix, (image.shape[0], image.shape[1]))
    contrast_image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    
    return blurred_image, translated_image, rotated_image, flipped_image, scaled_image, contrast_image
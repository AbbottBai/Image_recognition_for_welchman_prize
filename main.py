import cv2
import numpy as np

image = cv2.imread('selfie.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized_gray = cv2.resize(gray, (250, 250))
# Casts the original image into float, so that overflow does not occur due to negative and decimal calculations
# being carried out on integers.
resized_gray = resized_gray.astype(np.float32)


# This is also used in normal distribution!
def standardization(img):
    img_size = img.shape
    sum = 0
    squared_sum = 0
    number_of_pixels = img.shape[0] * img.shape[1]
    # Define an empty 2D matrix with the same shape as the original, unstandardized greyscale image
    std_img = np.zeros_like(img)

    for i in range (img_size[0]):
        for j in range (img_size[1]):
            sum += img[i][j]
            squared_sum += np.square(img[i][j])

    mean = sum / number_of_pixels
    std_deviation = np.sqrt((squared_sum / number_of_pixels) - np.square(mean))

    for i in range (img_size[0]):
        for j in range (img_size[1]):
            # This is where the actual normalisation occurs
            std_img[i][j] = (img[i][j] - mean) / std_deviation

    return std_img


def five_by_five(input_matrix):
    size = input_matrix.shape
    if size[0] != size[1]:
        print("Matrix is not a square, 5x5 function cannot be executed")

    else:
        num_squares = size[0] // 5
        output_matrix = [[[] for _ in range(num_squares)] for _ in range(num_squares)]
        for i in range (num_squares): #Vertical
            for j in range(num_squares): #Horizontal
                for k in range(5): #Individual vertical levels
                    for l in range(5): #Individual values
                        output_matrix[i][j].append(input_matrix[i * 5 + k][j * 5 + l])
        output_matrix = np.array(output_matrix)
        return output_matrix


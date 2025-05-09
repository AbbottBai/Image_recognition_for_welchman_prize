import cv2
import numpy as np
import os

number_of_photos = 5000

def take_photos():
    print("Please enter a directory name to add photos into")
    name = input(">")
    output_dir = name
    os.makedirs(output_dir, exist_ok=True)

    # Open the webcam (0 = default camera)
    cap = cv2.VideoCapture(0)

    # Take 5000 photos
    for i in range(number_of_photos):
        ret, frame = cap.read()  # Capture a frame
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_gray = cv2.resize(gray, (250, 250))
            resized_gray = resized_gray.astype(np.float32)
            # Casts the original image into float, so that overflow does not occur due to negative and decimal calculations
            # being carried out on integers.

            filename = os.path.join(output_dir, f"photo_{i}.png")
            cv2.imwrite(filename, resized_gray)  # Save the frame to file
            print(f"Saved {filename}")
            cv2.imshow('Captured Image', frame)  # Optional: show the photo
            #cv2.waitKey(1)  # Wait 1 ms between captures
        else:
            print("Failed to capture image")

    cap.release()
    cv2.destroyAllWindows()


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


# Pass in all images instead of just one image at a time
# Pass in the whole picture, not just a 5x5 area
# Self.x and self.y will be 3D numpy arrays. They will represent all images.
# W will be a 2D array, so will B. They will be local to the image.
# REMEMBER, this is ONE LAYER FOR ALL PHOTOS
class logistic_regression():
    def __init__(self, x, y, prev_d):
        self.x = x
        self.y = y
        self.x_shape = x.shape
        self.w = np.ones(self.x_shape[1]) # One W for each (5x5) area of an image
        self.b = np.ones(self.x_shape[1])
        self.sigmoid = np.zeros(self.x_shape[1]) # For one whole image
        self.all_sigmoid = np.zeros(self.x_shape[0])
        self.cost = np.ones(self.x_shape[1])
        self.prev_d = prev_d # 1D array local to the image
        self.alpha = 0.0001

    def forward_prop(self):
        # Calculate the total sigmoid of one image, and then concatenate sigmoid of all of the images at the end.
        for i in range (self.x_shape[0]): # Loop through all images
            for j in range(self.x_shape[1]): # For all areas of one image
                self.sigmoid[j] = 1 / (1 + np.e ** (-(np.dot(self.x[i][j], self.w[j]) + self.b[j])))

            self.all_sigmoid[i] = self.sigmoid

        return self.all_sigmoid


    def back_prop(self):
        pass # Find the derivative of the cost function of the final baymax function first.
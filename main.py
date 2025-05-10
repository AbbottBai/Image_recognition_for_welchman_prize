import cv2
import numpy as np
import os

number_of_photos = 2000
number_of_people = 4

def take_photos():
    for a in range(number_of_people):
        inp = input("Please hit enter once a new person is in front of the camera")
        output_dir = str(a)
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

                filename = os.path.join(output_dir, f"image{i}.png")
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
    def __init__(self, x):
        self.x = x
        self.x_shape = x.shape
        self.w = np.ones(self.x_shape[1]) # One W for each (5x5) area of an image
        self.b = np.ones(self.x_shape[1])
        self.sigmoid = np.zeros(self.x_shape[1]) # For one whole image
        self.all_sigmoid = np.zeros((self.x_shape[0], self.x_shape[1])) # 2D array containing sigmoid of all areas of all images
        self.cost = np.ones(self.x_shape[1])
        self.alpha = 0.0001

    def forward_prop(self):
        # Calculate the total sigmoid of one image, and then concatenate sigmoid of all of the images at the end.
        for i in range (self.x_shape[0]): # Loop through all images
            for j in range(self.x_shape[1]): # For all areas of one image
                self.sigmoid[j] = 1 / (1 + np.e ** (-(np.dot(self.x[i][j], self.w[j]) + self.b[j])))

            self.all_sigmoid[i] = self.sigmoid

        return self.all_sigmoid


    def back_prop(self, prev_d):
        # Prev_d is a 1D array for all areas of the image
        self.forward_prop()
        current_d = []
        for i in range(self.x_shape[0]): # For all images
            for j in range(self.x_shape[1]): # For each area of an image
                total_x = 0
                for k in range(self.x_shape[2]): # Find the average value that the current w changes the final result.
                    total_x += self.x[i][j][k]

                self.w[j] -= self.alpha * prev_d[j] * self.sigmoid[i][j] * (1 - self.sigmoid[i][j]) * (total_x/self.x_shape[2])
                self.b[j] -= self.alpha * prev_d[j] * self.sigmoid[i][j] * (1 - self.sigmoid[i][j])
                # Iteration to create new 5x5 areas with repeating current_d
                # Current_d will be a 2D matrix, local to the image
                for z in range(25):
                    current_d.append([prev_d[j] * self.sigmoid[i][j] * (1 - self.sigmoid[i][j])])

        return current_d


class softmax():
    def __init__(self, x, y):
        self.x = x
        self.y = y # Integer value for index of each face/a face, starting from ZERO!!!
        self.x_shape = x.shape
        self.w = np.ones(self.x_shape[1])
        self.b = np.ones(self.x_shape[1])
        self.softmax = np.zeros(self.x_shape[1])  # For one whole image
        self.all_softmax = np.zeros((self.x_shape[0], self.x_shape[1]))  # 2D array containing sigmoid of all areas of all images
        self.alpha = 0.0001

    def forward_prop(self):
        func = np.zeros((self.x_shape[0], self.x_shape[1])) # 2D array containing all areas of all images
        for i in range (self.x_shape[0]): # Loop through all images
            for j in range(self.x_shape[1]): # For all areas of one image
                func[i][j] = np.dot(self.w[j], self.x[i][j]) + self.b[j]
                self.softmax[j] = (np.e ** func[i][j]) / np.dot(np.e ** func[i])

            self.all_softmax[i] = self.softmax

        return self.all_softmax

    def cost(self):
        self.forward_prop()
        loss = 0
        for i in range(self.x_shape[0]):
            loss += -np.log(self.all_softmax[self.y])

        cost = loss / self.x_shape[0]
        return cost

    def gradient_descent(self):
        self.forward_prop()
        for i in range(self.x_shape[0]):
            # Line below is to make the gradient of the loss of the actual value negative whilst keeping others positive
            # So it reduces the weight for wrong predictions but increase the weight for right predictions later on...
            self.all_softmax[i][self.y] -= 1

            for j in range(self.x_shape[1]):
                total_x = 0
                for k in range(self.x_shape[2]):
                    total_x += self.x[i][j][k]

                actual_probability = 0
                if self.y == j:
                    actual_probability = 1
                self.w[j] -= self.alpha * (self.all_softmax[i][j] - actual_probability) * (total_x / self.x_shape[2])
                self.b[j] -= self.alpha * (self.all_softmax[i][j] - actual_probability)


def train():
    take_photos()
    print("Photo taking complete, commencing photo processing")
    all_photos = [] # This will be a 4D array, with all photos taken from different people
    for i in range(number_of_people):
        all_photos.append([])
        for j in range(number_of_photos):
            try:
                image_path = os.path.join(i, f"image{j}.png")
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                img = standardization(img)
                img = five_by_five(img)
                all_photos[i].append(img)
            except Exception as e:
                print(f"Error processing image {j} from subdirectory {i}: {e}")

    all_photos = np.array(all_photos)
    print("Image processing complete")

    print("Commencing model training")

    run = True
    total_iteration = 0
    while run:
        total_iteration += 1
        for i in range(number_of_people):

            input_layer = logistic_regression(all_photos[i])
            a1 = input_layer.forward_prop()
            a1 = five_by_five(a1)

            hidden1 = logistic_regression(a1)
            a2 = hidden1.forward_prop()
            a2 = five_by_five(a2)

            hidden2 = logistic_regression(a2)
            a3 = hidden2.forward_prop()
            a3 = five_by_five(a3)

            output = softmax(a3, i)
            cost = output.cost()
            print(f"Iteration: {total_iteration}, cost: {cost}, person: {i}")
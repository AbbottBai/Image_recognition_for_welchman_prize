import cv2
import numpy as np
import os

number_of_photos = 10
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
    img = img.astype(np.float64) # Higher precision to reduce rounding error
    sum = 0.0
    squared_sum = 0.0
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
        self.x = x # 4D array, 0 = image of person, 1 = number of 5x5 rows, 2 = number of 5x5 columns, 3 = number of pixels in 5x5 (25)
        self.x_shape = x.shape
        self.w = np.ones((self.x_shape[1], self.x_shape[2])) # 2D array, One W for each (5x5) area of an image
        self.b = np.ones((self.x_shape[1], self.x_shape[2]))
        self.sigmoid = np.zeros((self.x_shape[1], self.x_shape[2])) # For one whole image
        self.all_sigmoid = np.zeros((self.x_shape[0], self.x_shape[1], self.x_shape[2])) # 3D array containing sigmoid of all areas of all images
        self.cost = np.ones((self.x_shape[1], self.x_shape[2]))
        self.alpha = 0.0001

    def forward_prop(self):
        # Calculate the total sigmoid of one image, and then concatenate sigmoid of all of the images at the end.
        for i in range (self.x_shape[0]): # Loop through all images
            for j in range(self.x_shape[1]): # For all row boxes of one image
                for k in range(self.x_shape[2]): # For all column boxes of one image
                    self.sigmoid[j][k] = 1 / (1 + np.e ** (-(self.w[j][k] * sum(self.x[i][j][k]) + self.b[j][k])))

            self.all_sigmoid[i] = self.sigmoid

        return self.all_sigmoid


    def back_prop(self, prev_d):
        # Prev_d is a 1D array for all areas of the image
        self.forward_prop()
        current_d = [[[0 for _ in range(self.x_shape[2] * 5)] for _ in range(self.x_shape[1] * 5)] for _ in range(self.x_shape[0])]
        for i in range(self.x_shape[0]): # For all images
            for j in range(self.x_shape[1]): # For each 5x5 row of an image
                for k in range(self.x_shape[2]): # For each 5x5 column of an image
                    total_x = 0
                    for l in range(self.x_shape[3]): # Find the average value that the current w changes the final result.
                        total_x += self.x[i][j][k][l]

                    self.w[j][k] -= self.alpha * prev_d[i][j][k] * self.sigmoid[i][j][k] * (1 - self.sigmoid[i][j][k]) * (total_x/self.x_shape[3])
                    self.b[j][k] -= self.alpha * prev_d[i][j][k] * self.sigmoid[i][j][k] * (1 - self.sigmoid[i][j][k])
                    # Iteration to create new 5x5 areas with repeating current_d
                    # Current_d will be a 3D matrix, unique for each image
                    for y in range(5):
                        for z in range(5):
                            current_d[i][5*j+y][5*k+z] = ([prev_d[j][k] * self.sigmoid[i][j][k] * (1 - self.sigmoid[i][j][k])])
                            # There is a photo in your icloud if you forgot how this works
        current_d = np.array(current_d)
        return current_d


class softmax():
    def __init__(self, x, y):
        self.x = x
        self.y = y # Integer value for index of each face/a face, starting from ZERO!!!
        self.x_shape = x.shape
        self.w = np.ones((self.x_shape[1], self.x_shape[2]))
        self.b = np.ones((self.x_shape[1], self.x_shape[2]))
        self.softmax = np.zeros((self.x_shape[1], self.x_shape[2]))  # For one whole image
        self.all_softmax = [] # 3D array containing softmax of all areas of all images
        self.alpha = 0.0001

    def forward_prop(self):
        func = np.zeros((self.x_shape[0], self.x_shape[1], self.x_shape[2])) # 3D array containing all areas of all images
        for i in range (self.x_shape[0]): # Loop through all images
            func_sum = 0
            for k in range(self.x_shape[1]):  # For sum of functions of all rows
                for a in range(self.x_shape[2]): # For sum of functions of all columns
                    func_sum += np.e ** (self.w[k][a] * sum(self.x[i][k][a]) + self.b[k][a]) # Function for one area
            for j in range(self.x_shape[1]): # For each row
                for b in range(self.x_shape[2]): # For each column
                    func[i][j][b] = self.w[j][b] * sum(self.x[i][j][b]) + self.b[j][b] # Function for one area
                    self.softmax[j][b] = (np.e ** func[i][j][b]) / func_sum

            self.all_softmax.append(self.softmax)

        self.all_softmax = np.array(self.all_softmax)
        return self.all_softmax

    def cost(self):
        loss = 0
        for i in range(self.x_shape[0]):
            loss += -np.log(self.all_softmax[self.y])

        cost = loss / self.x_shape[0]
        return cost

    def gradient_descent(self):
        for i in range(self.x_shape[0]):
            # Line below is to make the gradient of the loss of the actual value negative whilst keeping others positive
            # So it reduces the weight for wrong predictions but increase the weight for right predictions later on...
            self.all_softmax[i][self.y] -= 1

            for j in range(self.x_shape[1] * self.x_shape[2]):
                total_x = 0
                for k in range(self.x_shape[2]):
                    total_x += self.x[i][j][k]

                actual_probability = 0
                if self.y == j:
                    actual_probability = 1
                self.w[j] -= self.alpha * (self.all_softmax[i][j] - actual_probability) * (total_x / self.x_shape[2])
                self.b[j] -= self.alpha * (self.all_softmax[i][j] - actual_probability)


def train():
    #take_photos()
    print("Photo taking complete, commencing photo processing")
    all_photos = [] # This will be a 5D array, with all photos taken from different people
    # 0 = person, 1 = image of person, 2 = number of 5x5 rows, 3 = number of 5x5 columns, 4 = number of pixels in 5x5 (25)
    for i in range(number_of_people):
        all_photos.append([])
        for j in range(number_of_photos):
            try:
                image_path = os.path.join(str(i), f"image{j}.png")
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
    prev_cost = 999
    while run:
        total_iteration += 1
        for i in range(number_of_people):
            # print(all_photos.shape), in case you need a reminder for what the all_image array looks like

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

            output.gradient_descent()
            current_d = hidden2.back_prop(cost)
            current_d = hidden1.back_prop(current_d)
            current_d = input_layer.back_prop(current_d)

            if prev_cost - cost <= 0.01:
                print("Training complete")
                run = False
            else:
                prev_cost = cost

train()
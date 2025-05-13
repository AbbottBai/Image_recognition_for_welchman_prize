import cv2
import numpy as np
import os
import random

number_of_photos = 1000
number_of_people = 3

def take_photos():
    print("Commencing photo capturing")
    for a in range(number_of_people):
        inp = input("Please hit enter once a new person is in front of the camera")
        output_dir = str(a)
        os.makedirs(output_dir, exist_ok=True)

        # Open the webcam (0 = default camera)
        cap = cv2.VideoCapture(0)

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
    print("Photo capturing complete")


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


class relu():
    def __init__(self, x, num_neurons, alpha):
        self.x = x # 0 = person, 1 = image of person, 2 = number of rows, 3 = number of columns
        self.num_neurons = num_neurons
        self.sqrt_neurons = int(np.sqrt(self.num_neurons))
        # w is a 4D arrays, one w per pixel. One neuron for all pixels.
        self.w = [[[[random.uniform(-0.4, 0.4) for _ in range(self.x.shape[3])] for _ in range(self.x.shape[2])] for _ in range(self.sqrt_neurons)] for _ in range(self.sqrt_neurons)]
        # W is set to random value to prevent symmetry between all neurons.
        # B is set to 0 as W already prevents symmetry
        # b is a 2D array, one b per neuron.
        self.b = [[0 for _ in range(self.sqrt_neurons)] for _ in range(self.sqrt_neurons)]
        self.output = [[[[0 for _ in range(self.sqrt_neurons)] for _ in range(self.sqrt_neurons)] for _ in range(self.x.shape[1])] for _ in range(self.x.shape[0])]
        self.alpha = alpha

    def forward_prop(self):
        self.w = np.array(self.w, dtype=np.float32)
        for i in range(len(self.w[0])):
            for j in range(len(self.w[1])):
                self.w[i][j] = self.w[i][j].T # This transposes all matrices in all neurons, ready for matrix multiplication

                for k in range(self.x.shape[0]):
                    for l in range(self.x.shape[1]):
                        self.output[k][l][i][j] = np.matmul(self.w[i][j], self.x[k][l]) + self.b[i][j]

                self.w[i][j] = self.w[i][j].T # This returns all w back to original shape after matmul.

        output = np.array(self.output)
        return output

    def back_prop(self, prev_d): # Implement matmul? Image by image
        sum_weights = np.sum(self.w)
        current_d = [[0 for _ in range(self.x.shape[1])] for _ in range(self.x.shape[0])] # 2D array, one per image
        for a in range(self.x.shape[0]):
            for b in range(self.x.shape[1]):
                total_x = np.sum(self.x[a][b]) # Sum of pixels in one image

                for i in range(len(self.w[0])):
                    for j in range(len(self.w[1])):
                        sum_w = np.sum(self.w[i][j])
                        for k in range(len(self.w[2])):
                            for l in range(len(self.w[3])):
                                old_w = self.w[i][j][k][l]
                                self.w[i][j][k][l] -= self.alpha * prev_d[a][b] * old_w * (total_x / (self.x.shape[2] * self.x.shape[3])) # Update w and b for all images at once to prevent w and b overfitting and changing drastically to one pixel

                        self.b[i][j] -= self.alpha * prev_d[a][b] * sum_w # Average w?

                current_d[a][b] = prev_d[a][b] * sum_weights # Average the weights?

        return current_d

class softmax():
    def __init__(self, x, num_neurons, alpha):
        self.x = x # 0 = person, 1 = image of person, 2 = number of rows, 3 = number of columns
        self.num_neurons = num_neurons
        # w is a 3D array which contains one w per pixel for all pixels per neuron.
        self.w = [[[random.uniform(-0.4, 0.4) for _ in range(self.x.shape[3])] for _ in range(self.x.shape[2])] for _ in range(self.num_neurons)]
        self.b = [0 for _ in range(self.num_neurons)]
        self.linear_func = [[[0 for _ in range(self.num_neurons)] for _ in range(self.x.shape[1])] for _ in range(self.x.shape[0])]
        self.output = [[[0 for _ in range(num_neurons)] for _ in range(self.x.shape[1])] for _ in range(self.x.shape[0])]
        self.y = int() # Start index from 0
        self.alpha = alpha

    def forward_prop(self):
        std_x = [[[[0 for _ in range(self.x.shape[3])] for _ in range(self.x.shape[2])] for _ in range(self.x.shape[1])] for _ in range(self.x.shape[0])]
        self.w = np.array(self.w, dtype=np.float32)
        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                total_x = np.sum(self.x[i][j])
                total_x2 = np.sum(np.square(self.x[i][j]))
                # Normal standardization is implimented so the exponent doesn't get too large and uncomputable
                mean_x = total_x / (self.x.shape[0] * self.x.shape[1] * self.x.shape[2] * self.x.shape[3])
                std_dev =  np.sqrt((total_x2/(self.x.shape[2] * self.x.shape[3])) - mean_x ** 2)
                for a in range(self.x.shape[2]):
                    for b in range(self.x.shape[3]):
                        std_x[i][j][a][b] = (self.x[i][j][a][b] - mean_x) / std_dev
                denominator = 0.0
                for c in range(self.num_neurons):
                    self.w[c] = self.w[c].T # This transposes w for matrix multiplication later.
                    self.linear_func[i][j][c] = np.matmul(self.w[c], std_x[i][j]) + self.b[c]

                    denominator += np.exp(self.linear_func[i][j][c])

                    self.w[c] = self.w[c].T # Transposes w back for use later

                for f in range(self.num_neurons):
                    self.output[i][j][f] = np.exp(self.linear_func[i][j][f]) / denominator


    def cost(self): # Image by image
        cost = [[0 for _ in range(self.x.shape[1])] for _ in range(self.x.shape[0])] # 2D array, one per image
        for i in range(self.x.shape[0]):
            self.y = i
            for j in range(self.x.shape[1]):
                cost[i][j] = -(np.log(self.output[i][j][self.y]))

        total_cost = np.sum(cost) / (self.x.shape[1] * self.x.shape[0])

        return cost, total_cost

    def gradient_descent(self): # Image by image
        for i in range(self.x.shape[0]):
            self.y = i
            for j in range(self.x.shape[1]):
                self.output[i][j][self.y] -= 1
                # Line above is to make the gradient of the loss of the actual value negative whilst keeping the others positive
                # So it reduces the weight for wrong predictions but increases the weight for right predictions later on...
                total_x = sum(self.x[i][j])

                for m in range(self.num_neurons):
                    for n in range(self.x.shape[2]):
                        for p in range(self.x.shape[3]):
                            self.w[m][n][p] -= self.alpha * self.output[i][j][m] * (total_x / (self.x.shape[2] * self.x.shape[3]))
                    self.b[m] -= self.alpha * self.output[i][j][m]


def train():
    print("commencing photo processing")
    all_photos = [] # This will be a 4D array, with all photos taken from different people
    # 0 = person, 1 = image of person, 2 = number of rows, 3 = number of columns
    for i in range(number_of_people):
        all_photos.append([])
        print(f"Processing person {i+1} out of {number_of_people+1}")
        photo_counter = 0
        for j in range(number_of_photos):
            if photo_counter >= 10:
                print(f"Currently processing photo {j+1} out of {number_of_photos}")
                photo_counter = 0
            photo_counter += 1
            
            try:
                image_path = os.path.join(str(i), f"image{j}.png")
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                img = standardization(img)
                all_photos[i].append(img)
            except Exception as e:
                print(f"Error processing image {j} from subdirectory {i}: {e}")

    all_photos = np.array(all_photos)
    print("Image processing complete")

    print("Commencing model training")

    run = True
    total_iteration = 0
    total_cost = 100
    while run:
        total_iteration += 1
        if total_cost <= 20:
            alpha = 0.0001
        else:
            alpha = 0.001
        hidden1 = relu(all_photos, 4096, alpha)
        a1 = hidden1.forward_prop()
        print("hidden layer 1 forward propagation complete")

        hidden2 = relu(a1, 2025, alpha)
        a2 = hidden2.forward_prop()
        print("hidden layer 2 forward propagation complete")

        hidden3 = relu(a2, 1024, alpha)
        a3 = hidden3.forward_prop()
        print("hidden layer 3 forward propagation complete")

        hidden4 = relu(a3, 529, alpha)
        a4 = hidden4.forward_prop()
        print("hidden layer 4 forward propagation complete")

        hidden5 = relu(a4, 256, alpha)
        a5 = hidden5.forward_prop()
        print("hidden layer 5 forward propagation complete")

        hidden6 = relu(a5, 121, alpha)
        a6 = hidden6.forward_prop()
        print("hidden layer 6 forward propagation complete")

        output = softmax(a6, number_of_people, alpha)
        output.forward_prop()
        print("Output layer forward propagation complete")
        cost, total_cost = output.cost()
        print(f"Iteration: {total_iteration}, cost: {total_cost}")

        output.gradient_descent()
        current_d = hidden6.back_prop(cost)
        print("output layer backprop complete")
        current_d = hidden5.back_prop(current_d)
        print("hidden 5 backprop complete")
        current_d = hidden4.back_prop(current_d)
        print("hidden 4 backprop complete")
        current_d = hidden3.back_prop(current_d)
        print("hidden 3 backprop complete")
        current_d = hidden2.back_prop(current_d)
        print("hidden 2 backprop complete")
        current_d = hidden1.back_prop(current_d)
        print("hidden 1 backprop complete")
        if total_cost <= 0.001:
            run = False


def predict():
    run = True
    counter = 0
    while run:
        inp = input("press t to take a photo, or q to quit").lower()
        if inp[0] == "q":
            run = False
        else:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized_gray = cv2.resize(gray, (250, 250))
                resized_gray = resized_gray.astype(np.float32)
                # Casts the original image into float, so that overflow does not occur due to negative and decimal calculations
                # being carried out on integers.

                filename = os.path.join(os.path.dirname(__file__), f"{counter}.png")
                cv2.imwrite(filename, resized_gray)  # Save the frame to file
                print(f"Saved {filename}")
                cv2.imshow('Captured Image', frame)  # Optional: show the photo

                image = f"{counter}.png"

            else:
                print("Failed to capture image")

        counter += 1


#take_photos()
train()
import cv2
import numpy as np
import os
import random
import time

number_of_photos = 200
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
                resized_gray = resized_gray.astype(np.float64)
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
    img = img.astype(np.float64)
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
        self.x = x # 0 = number of photos in batch, 1 = number of rows, 2 = number of columns
        self.num_neurons = num_neurons
        self.sqrt_neurons = int(np.sqrt(self.num_neurons))
        # w is a 4D arrays, one w per pixel. One neuron for all pixels.
        self.w = [[[[random.uniform(-0.4, 0.4) for _ in range(self.x.shape[2])] for _ in range(self.x.shape[1])] for _ in range(self.sqrt_neurons)] for _ in range(self.sqrt_neurons)]
        # W is set to random value to prevent symmetry between all neurons.
        # B is set to 0 as W already prevents symmetry
        # b is a 2D array, one b per neuron.
        self.b = [[0 for _ in range(self.sqrt_neurons)] for _ in range(self.sqrt_neurons)]
        self.output = [[[0 for _ in range(self.sqrt_neurons)] for _ in range(self.sqrt_neurons)] for _ in range(self.x.shape[0])]
        self.alpha = alpha
        self.w = np.array(self.w, dtype=np.float64)
        self.b = np.array(self.b, dtype=np.float64)

    def forward_prop(self):
        start_time = int(time.time())
        for i in range(self.sqrt_neurons):
            for j in range(self.sqrt_neurons):

                end_time = int(time.time())
                if end_time - start_time > 2:
                    start_time = end_time

                for k in range(self.x.shape[0]):
                    self.output[k][i][j] = np.sum(self.w[i][j] * self.x[k]) + self.b[i][j]

        self.output = np.array(self.output)
        return self.output

    def back_prop(self, prev_d): # Image by image
        sum_weights = np.sum(self.w)
        current_d = [[0 for _ in range(self.x.shape[1])] for _ in range(self.x.shape[0])] # 2D array, one per image
        for a in range(self.x.shape[0]):
            total_x = np.sum(self.x[a]) # Sum of pixels in one image

            for i in range(len(self.w[0])):
                for j in range(len(self.w[1])):
                    sum_w = np.sum(self.w[i][j])
                    for k in range(len(self.w[2])):
                        for l in range(len(self.w[3])):
                            old_w = self.w[i][j][k][l]
                            self.w[i][j][k][l] -= self.alpha * prev_d[a] * old_w * (total_x / (self.x.shape[1] * self.x.shape[2]))

                    self.b[i][j] -= self.alpha * prev_d[a] * sum_w # Average w?

            current_d[a] = prev_d[a] * sum_weights # Average the weights?

        return current_d

class softmax():
    def __init__(self, x, num_neurons, alpha, y):
        self.x = x # 0 = number of photos in batch, 1 = number of rows, 2 = number of columns
        self.num_neurons = num_neurons
        # w is a 3D array which contains one w per pixel for all pixels per neuron.
        self.w = [[[random.uniform(-0.4, 0.4) for _ in range(self.x.shape[2])] for _ in range(self.x.shape[1])] for _ in range(self.num_neurons)]
        self.b = [0 for _ in range(self.num_neurons)]
        self.linear_func = [[0 for _ in range(self.num_neurons)] for _ in range(self.x.shape[0])]
        self.output = [[0 for _ in range(self.num_neurons)] for _ in range(self.x.shape[0])]
        self.y = y
        self.alpha = alpha
        self.w = np.array(self.w, dtype=np.float64)
        self.b = np.array(self.b, dtype=np.float64)

    def forward_prop(self):
        # std_x = [[[[0 for _ in range(self.x.shape[3])] for _ in range(self.x.shape[2])] for _ in range(self.x.shape[1])] for _ in range(self.x.shape[0])]
        for i in range(self.x.shape[0]):
            total_x = np.sum(self.x[i])
            total_x2 = np.sum(np.square(self.x[i]))
            # Normal standardization is implimented so the exponent doesn't get too large and uncomputable
            mean_x = total_x / (self.x.shape[1] * self.x.shape[2])
            std_dev = np.sqrt((total_x2 / (self.x.shape[1] * self.x.shape[2])) - np.square(mean_x))
            for a in range(self.x.shape[1]):
                for b in range(self.x.shape[2]):
                    self.x[i][a][b] = (self.x[i][a][b] - mean_x) / std_dev
            denominator = 0.0
            for c in range(self.num_neurons):
                self.linear_func[i][c] = np.sum(self.w[c] * self.x[i]) + self.b[c]

                denominator += np.exp(self.linear_func[i][c])

            for f in range(self.num_neurons):
                self.output[i][f] = np.exp(self.linear_func[i][f]) / denominator


    def cost(self): # Image by image
        cost = [0 for _ in range(self.x.shape[0])] # 1D array, one per image
        for i in range(self.x.shape[0]):
            cost[i] = -(np.log(self.output[i][self.y]))

        total_cost = np.sum(cost) / self.x.shape[0]

        return cost, total_cost

    def gradient_descent(self): # Image by image
        for i in range(self.x.shape[0]):
            self.output[i][self.y] -= 1
            # Line above is to make the gradient of the loss of the actual value negative whilst keeping the others positive
            # So it reduces the weight for wrong predictions but increases the weight for right predictions later on...
            total_x = np.sum(self.x[i])
            # Sum x creates a 1D array from the sum of values in each index of both lists for row and column
            # Sum again creates a scalar from the 1D array
            # np.sum just combines the two methods.

            for m in range(self.num_neurons):
                for n in range(self.x.shape[1]):
                    for p in range(self.x.shape[2]):
                        self.w[m][n][p] -= self.alpha * self.output[i][m] * (total_x / (self.x.shape[1] * self.x.shape[2]))
                self.b[m] -= self.alpha * self.output[i][m]


# I had help from chatgpt with the save and load layer parameters functions. Apart from that everything is coded muself.
def save_layer_parameters(layer, filename):
    data = {f"w{i}": w for i, w in enumerate(layer.w)}
    data.update({f"b{i}": b for i, b in enumerate(layer.b)})
    np.savez(filename, **data)


def load_layer_parameters(layer, filename):
    data = np.load(filename)
    num_layers = len([k for k in data.keys() if k.startswith("w")])
    layer.w = [data[f"w{i}"] for i in range(num_layers)]
    layer.b = [data[f"b{i}"] for i in range(num_layers)]


def train():
    print("\nCommencing photo processing")
    all_photos = [] # This will be a 5D array, with all photos taken from different people
    # 0 = number of person, 1 = number of batches, 2 = number of photos in batch, 3 = number of rows, 4 = number of columns

    batch_size = 20 # WARNING: THIS HAS TO BE DIVISIBLE BY NUMBER OF PHOTOS
    num_batches_per_person = number_of_photos // batch_size
    for i in range(number_of_people):
        print(f"\nProcessing photos of person {i + 1} out of {number_of_people}")
        all_photos.append([])
        for j in range(num_batches_per_person):
            all_photos[i].append([]) # Appends a sublist for each batch to every person.
            for k in range(batch_size):
                try:
                    image_path = os.path.join(str(i), f"image{j * batch_size + k}.png")
                    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    img = standardization(img)
                    all_photos[i][j].append(img)
                except Exception as e:
                    print(f"Error processing image {j * batch_size + k} from subdirectory {i}: {e}")

    all_photos = np.array(all_photos, dtype=np.float64)
    print(f"\nAll photos array shape: {all_photos.shape}")
    print("\nImage processing complete")

    print("\nCommencing model training")

    run = True
    total_iteration = 0
    total_cost = 100
    while run:
        if total_cost <= 20:
            alpha = 0.0001
        else:
            alpha = 0.001

        for a in range(all_photos.shape[0]):
            for b in range(all_photos.shape[1]):
                hidden1 = relu(all_photos[a][b], 2025, alpha)
                a1 = hidden1.forward_prop()

                hidden2 = relu(a1, 1024, alpha)
                a2 = hidden2.forward_prop()

                hidden3 = relu(a2, 529, alpha)
                a3 = hidden3.forward_prop()

                hidden4 = relu(a3, 121, alpha)
                a4 = hidden4.forward_prop()

                output_layer = softmax(a4, number_of_people, alpha, a)
                output_layer.forward_prop()
                cost, total_cost = output_layer.cost()
                print(f"Total iteration: {total_iteration}, current person: {a + 1} out of {all_photos.shape[0]}, "
                      f"batch number: {b + 1} out of {batch_size}, cost: {total_cost}")

                output_layer.gradient_descent()
                current_d = hidden4.back_prop(cost)
                current_d = hidden3.back_prop(current_d)
                current_d = hidden2.back_prop(current_d)
                current_d = hidden1.back_prop(current_d)

                if total_iteration >= 3:
                    run = False # Terminate as soon as cost is low enough, and that it has passed a few iterations through entire dataset, to prevent overfitting to one batch.

        total_iteration += 1

    try:
        print("Beginning to package and save weights and biases")
        save_layer_parameters(hidden1, "h1.npz")
        save_layer_parameters(hidden2, "h2.npz")
        save_layer_parameters(hidden3, "h3.npz")
        save_layer_parameters(hidden4, "h4.npz")
        save_layer_parameters(output_layer, "output_layer.npz")
        print("\nAll weights and biases saved\n")

    except Exception as e:
        print(f"Error whilst packaging weights and biases: {e}")

    print("Model training complete")

def predict():
    run = True
    counter = 0
    while run:
        inp = input("press t to take a photo, or q to quit: ").lower()
        if inp[0] == "q":
            run = False
        else:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized_gray = cv2.resize(gray, (250, 250))
                resized_gray = resized_gray.astype(np.float64)
                # Casts the original image into float, so that overflow does not occur due to negative and decimal calculations
                # being carried out on integers.

            else:
                print("Failed to capture image")

            cap.release()
            cv2.destroyAllWindows()


            # Image standardization
            input_array = []
            try:
                image = standardization(resized_gray)
                input_array.append(image)
                input_array = np.array(input_array, dtype=np.float64)
                print("\nImage processing complete")
            except Exception as e:
                print(f"Error during standardization process of image: {e}")

            print("\nForward propagation commencing...")

            # Initialise layers
            alpha = 0.01

            hidden1 = relu(input_array, 2025, alpha)
            load_layer_parameters(hidden1, "h1.npz")
            print(hidden1.w[0][0][0][0])
            a1 = hidden1.forward_prop()

            hidden2 = relu(a1, 1024, alpha)
            load_layer_parameters(hidden2, "h2.npz")
            a2 = hidden2.forward_prop()

            hidden3 = relu(a2, 529, alpha)
            load_layer_parameters(hidden3, "h3.npz")
            a3 = hidden3.forward_prop()

            hidden4 = relu(a3, 121, alpha)
            load_layer_parameters(hidden4, "h4.npz")
            a4 = hidden4.forward_prop()

            output_layer = softmax(a4, number_of_people, alpha, 0)
            load_layer_parameters(output_layer, "output_layer.npz")
            output_layer.forward_prop()
            result = [[],[]]
            max_prob = 0.0
            for i in range(output_layer.num_neurons):
                if output_layer.output[0][i] > max_prob:
                    result[0] = output_layer.output[0][i]
                    result[1] = i
                    max_prob = output_layer.output[0][i]

            print(f"\nThe model predicts that this photo belongs to person {result[1]}")
            print(f"\nThe likelihood that this photo belongs to person {result[1]} is {result[0] * 100}%")

            counter += 1

            del hidden1
            del hidden2
            del hidden3
            del hidden4
            del output_layer

print("press 1 to take photos for training")
print("press 2 to train model")
print("press 3 to make prediction")
selection = int(input(">"))
if selection == 1:
    take_photos()
elif selection == 2:
    train()
elif selection == 3:
    predict()
else:
    print("Invalid input")
import cv2
import numpy as np
import os
import random
import albumentations as A

number_of_photos = 400 # Number of photos that the camera actually takes
#IMPORTANT: augmented_photos + 1 must be divisible by batch size!!!
augmented_photos = 2 # Number of photos that are augmented per photo taken by the camera
total_num_photos = number_of_photos * augmented_photos + number_of_photos

number_of_people = 3

def take_photos():
    # Augmentations (flip, rotate, crop, color augment, etc.)
    # This is to transform original image by applying augmentations. This will increase training set size, and reduce the effect of enviornment factors such as lighting on the final result.
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ])

    print("Commencing photo capturing")
    for a in range(number_of_people):
        inp = input("Please hit enter once a new person is in front of the camera")
        output_dir = str(a)
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(0) # Open the webcam (0 = default camera)
        current_photo_index = 0
        for i in range(number_of_photos):
            ret, frame = cap.read()  # Capture a frame
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized_gray = cv2.resize(gray, (128, 128))
                resized_gray = resized_gray.astype(np.float64)
                #resized_gray = np.uint8(resized_gray)
                # Casts the original image into float, so that overflow does not occur due to negative and decimal calculations
                # being carried out on integers.

                filename = os.path.join(output_dir, f"image{current_photo_index}.png")
                cv2.imwrite(filename, resized_gray)  # Save the frame to file
                print(f"Saved {filename}")

                current_photo_index += 1

                # Start of process for augmented photos
                for j in range(augmented_photos):
                    augmented = transform(image=frame)["image"]
                    gray_augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
                    resized_augmented = cv2.resize(gray_augmented, (128, 128))
                    resized_augmented = resized_augmented.astype(np.float64)

                    augmented_filename = os.path.join(output_dir, f"image{current_photo_index}.png")
                    cv2.imwrite(augmented_filename, resized_augmented)

                    current_photo_index += 1

            else:
                print("Failed to capture image")

        cap.release()
    print("Photo capturing complete")


# This is also used in normal distribution!
def standardization(img):
    img = img.astype(np.float64)
    number_of_pixels = img.shape[0] * img.shape[1]

    total_x2 = np.sum(np.square(img))
    mean = np.sum(img) / number_of_pixels

    std_deviation = np.sqrt((total_x2 / number_of_pixels) - np.square(mean))

    img = (img - mean) / std_deviation

    return img


class relu():
    def __init__(self, num_neurons):
        self.x = None # Placeholder
        self.num_neurons = num_neurons
        self.sqrt_neurons = int(np.sqrt(self.num_neurons))
        # w is a 4D arrays, one w per pixel. One neuron for all pixels.
        self.w = None
        # W is set to random value to prevent symmetry between all neurons.
        # B is set to 0 as W already prevents symmetry
        # b is a 2D array, one b per neuron.
        self.b = [[0 for _ in range(self.sqrt_neurons)] for _ in range(self.sqrt_neurons)]
        self.output = None
        self.alpha = float()
        self.b = np.array(self.b, dtype=np.float64)

    def initialize_parameters(self):
        # w is a 4D arrays, one w per pixel. One neuron for all pixels.
        self.w = [[[[random.uniform(-0.4, 0.4) for _ in range(self.x.shape[2])] for _ in range(self.x.shape[1])] for _ in range(self.sqrt_neurons)] for _ in range(self.sqrt_neurons)]
        self.w = np.array(self.w, dtype=np.float64)

    def forward_prop(self, x, require_standardizing):
        rs = require_standardizing
        self.x = x # 0 = number of photos in batch, 1 = number of rows, 2 = number of columns
        self.output = [[[0 for _ in range(self.sqrt_neurons)] for _ in range(self.sqrt_neurons)] for _ in range(self.x.shape[0])]

        # Normal distribution is implimented to prevent stack overflow at back propagation.
        if rs:
            for a in range(self.x.shape[0]):
                sum_x_per_picture = np.sum(self.x[a])
                sum_x2_per_picture = np.sum(np.square(self.x[a]))
                mean_x = sum_x_per_picture / (self.x.shape[1] * self.x.shape[2])
                std_dev = np.sqrt((sum_x2_per_picture / (self.x.shape[1] * self.x.shape[2])) - np.square(mean_x))
                for b in range(self.x.shape[1]):
                    for c in range(self.x.shape[2]):
                        self.x[a][b][c] = (self.x[a][b][c] - mean_x) / std_dev

        if self.w is None:
            self.initialize_parameters()

        for i in range(self.sqrt_neurons):
            for j in range(self.sqrt_neurons):

                for k in range(self.x.shape[0]):
                    linear_func = np.sum(self.w[i][j] * self.x[k]) + self.b[i][j]
                    # Relu begins here
                    if linear_func > 0:
                        self.output[k][i][j] = linear_func
                    else:
                        self.output[k][i][j] = 0

        self.output = np.array(self.output)
        return self.output

    def back_prop(self, prev_d, alpha): # Image by image
        self.alpha = alpha

        current_d = [[0 for _ in range(self.x.shape[1])] for _ in range(self.x.shape[0])] # 2D array, one per image

        for a in range(self.x.shape[0]):

            #w and b coefficient are calculated outside loop so that the values that are not changing do not have to be recalculated.
            w_coefficient = self.alpha * prev_d[a] * self.x[a] # This creates a 2D array, one per pixel, but also one per w.
            b_coefficient = self.alpha * prev_d[a]


            # Moved W outside of nested loop. Vectorized.

            # The 4D and 2D array shapes are different.
            # So it will match the 2D array with the 2 inner most lists, which are also one per pixel.
            self.w -= w_coefficient[None, None, :, :] * self.w
            #This is so each w only updates by multiplying by the x that it is responsible for, instead of all x values.

            for i in range(self.w.shape[0]):
                for j in range(self.w.shape[1]):
                    self.b[i][j] -= b_coefficient * np.sum(self.w[i][j]) # Average the sum w?

            current_d[a] = prev_d[a] * (np.sum(self.w) / (self.x.shape[1] * self.x.shape[2] * self.num_neurons))
            # The average is used for weights because image_sum_weights itself is too large in value, and caused overflow.

        current_d = np.array(current_d, dtype=np.float64)

        return current_d

class softmax():
    def __init__(self, num_neurons):
        self.x = None # placeholder
        self.num_neurons = num_neurons
        # w is a 3D array which contains one w per pixel for all pixels per neuron.
        self.w = None
        self.b = [0 for _ in range(self.num_neurons)]
        self.linear_func = None
        self.output = None
        self.y = None # Placeholder
        self.alpha = float()
        self.b = np.array(self.b, dtype=np.float64)

    def initialize_parameters(self):
        # w is a 3D array which contains one w per pixel for all pixels per neuron.
        self.w = [[[random.uniform(-0.4, 0.4) for _ in range(self.x.shape[2])] for _ in range(self.x.shape[1])] for _ in range(self.num_neurons)]
        self.w = np.array(self.w, dtype=np.float64)

    def forward_prop(self, x):
        self.x = x  # 0 = number of photos in batch, 1 = number of rows, 2 = number of columns
        self.linear_func = [[0 for _ in range(self.num_neurons)] for _ in range(self.x.shape[0])]
        self.output = [[0 for _ in range(self.num_neurons)] for _ in range(self.x.shape[0])]

        if self.w is None: # Checks if parameters has been defined yet.
            self.initialize_parameters()

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


    def cost(self, y): # Image by image
        self.y = y
        cost = [0 for _ in range(self.x.shape[0])] # 1D array, one per image
        for i in range(self.x.shape[0]):
            cost[i] = -(np.log(self.output[i][self.y[i]]))

        total_cost = np.sum(cost) / self.x.shape[0]

        return cost, total_cost

    def gradient_descent(self, alpha): # Image by image
        self.alpha = alpha
        for i in range(self.x.shape[0]):
            self.output[i][self.y[i]] -= 1
            # Line above is to make the gradient of the loss of the actual value negative whilst keeping the others positive
            # So it reduces the weight for wrong predictions but increases the weight for right predictions later on...
            w_coefficient = self.alpha * self.x[i] # This creates a 2D array, one per pixel, but also one per w.

            for m in range(self.num_neurons):
                w_cost = w_coefficient * self.output[i][m]
                self.w[m] -= w_cost
                self.b[m] -= self.alpha * self.output[i][m]


# I had help from chatgpt with the save and load layer parameters functions. Apart from that everything is coded myself.
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
    print("\nCommencing model training")

    batch_size = 60 # WARNING: THIS HAS TO BE DIVISIBLE BY TOTAL NUMBER OF PHOTOS

    alpha = 0.0001
    hidden1 = relu(1024)
    hidden2 = relu(529)
    hidden3 = relu(121)
    output_layer = softmax(number_of_people)

    run = True

    current_iteration = 1
    total_num_batches = (total_num_photos * number_of_people) // batch_size  # Number of batches for all people
    num_iterations = 50
    current_batch_number = 1

    current_iteration_cost = 0
    prev_five = []
    current_five = []

    while run:
        current_batch = []
        batch_y = [] # One value per image for the batch. 1D array
        for i in range(batch_size):
            try:
                current_person = random.randint(0, number_of_people-1)
                current_photo = random.randint(0, total_num_photos-1)
                batch_y.append(current_person)
                image_path = os.path.join(str(current_person), f"image{current_photo}.png")
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                img = standardization(img)
                current_batch.append(img)
            except Exception as e:
                print(f"Error processing image: {e}")

        current_batch = np.array(current_batch, dtype=np.float64)

        a1 = hidden1.forward_prop(current_batch, False)
        a2 = hidden2.forward_prop(a1, True)
        a3 = hidden3.forward_prop(a2, True)
        output_layer.forward_prop(a3)

        cost, total_cost = output_layer.cost(batch_y)
        current_iteration_cost += total_cost

        output_layer.gradient_descent(alpha)
        current_d = hidden3.back_prop(cost, alpha)
        current_d = hidden2.back_prop(current_d, alpha)
        current_d = hidden1.back_prop(current_d, alpha)


        print(f"Total iteration: {current_iteration} out of {num_iterations}, batch number: {current_batch_number} out of {total_num_batches}, cost: {total_cost}")

        current_batch_number += 1
        if current_batch_number > total_num_batches:

            current_iteration_cost = current_iteration_cost / current_batch_number
            current_five.append(current_iteration_cost)

            if len(current_five) >= 5:
                current_mean = np.mean(current_five)

                if len(prev_five) == 0:
                    past_mean = 0
                else:
                    past_mean = np.mean(prev_five)

                # abs returns the magnitude of a value
                if abs(current_mean - past_mean) <= 0.05 and alpha > 0.000001:
                    alpha = alpha * 0.1 # Reduces alpha by a factor of 10, to prevent plateauing of the cost.
                    print(f"Cost has plateaued, alpha has been reduced by a factor of 10 to {alpha}.")

                prev_five = current_five
                current_five = []

            current_iteration += 1
            current_batch_number = 1
            current_iteration_cost = 0

        if current_iteration > num_iterations:
            run = False


    try:
        print("Beginning to package and save weights and biases")
        save_layer_parameters(hidden1, "laptop and bottle/h1.npz")
        save_layer_parameters(hidden2, "laptop and bottle/h2.npz")
        save_layer_parameters(hidden3, "laptop and bottle/h3.npz")
        save_layer_parameters(output_layer, "laptop and bottle/output_layer.npz")
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
                resized_gray = cv2.resize(gray, (128, 128))
                resized_gray = resized_gray.astype(np.float64)
                # Casts the original image into float, so that overflow does not occur due to negative and decimal calculations
                # being carried out on integers.

            else:
                print("Failed to capture image")

            cap.release()


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

            # Initialise layer
            hidden1 = relu(1024)
            hidden2 = relu(529)
            hidden3 = relu(121)
            output_layer = softmax(number_of_people)

            load_layer_parameters(hidden1, "laptop and bottle/h1.npz")
            load_layer_parameters(hidden2, "laptop and bottle/h2.npz")
            load_layer_parameters(hidden3, "laptop and bottle/h3.npz")
            load_layer_parameters(output_layer, "laptop and bottle/output_layer.npz")

            a1 = hidden1.forward_prop(input_array, False)
            a2 = hidden2.forward_prop(a1, True)
            a3 = hidden3.forward_prop(a2, True)
            output_layer.forward_prop(a3)

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
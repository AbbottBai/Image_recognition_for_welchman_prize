import cv2
import numpy as np
import os
import random
import albumentations as A

number_of_photos = 200 # Number of photos that the camera actually takes
#IMPORTANT: augmented_photos + 1 must be divisible by batch size!!!
augmented_photos = 4 # Number of photos that are augmented per photo taken by the camera
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
                resized_gray = cv2.resize(gray, (250, 250))
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
                    resized_augmented = cv2.resize(gray_augmented, (250, 250))
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
    def __init__(self, num_neurons, alpha):
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
        self.alpha = alpha
        self.b = np.array(self.b, dtype=np.float64)

    def initialize_parameters(self):
        # w is a 4D arrays, one w per pixel. One neuron for all pixels.
        self.w = [[[[random.uniform(-0.4, 0.4) for _ in range(self.x.shape[2])] for _ in range(self.x.shape[1])] for _ in range(self.sqrt_neurons)] for _ in range(self.sqrt_neurons)]
        self.output = [[[0 for _ in range(self.sqrt_neurons)] for _ in range(self.sqrt_neurons)] for _ in range(self.x.shape[0])]
        self.w = np.array(self.w, dtype=np.float64)

    def forward_prop(self, x, require_standardizing):
        rs = require_standardizing
        self.x = x # 0 = number of photos in batch, 1 = number of rows, 2 = number of columns

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
                    self.output[k][i][j] = np.sum(self.w[i][j] * self.x[k]) + self.b[i][j]

        self.output = np.array(self.output)
        return self.output

    def back_prop(self, prev_d): # Image by image
        current_d = [[0 for _ in range(self.x.shape[1])] for _ in range(self.x.shape[0])] # 2D array, one per image
        for a in range(self.x.shape[0]):
            total_x = np.sum(self.x[a]) # Sum of pixels in one image
            image_sum_weights = np.sum(self.w)

            for i in range(len(self.w[0])):
                for j in range(len(self.w[1])):
                    sum_w_per_pixel = np.sum(self.w[i][j])
                    for k in range(len(self.w[2])):
                        for l in range(len(self.w[3])):
                            old_w = self.w[i][j][k][l]
                            self.w[i][j][k][l] -= self.alpha * prev_d[a] * old_w * (total_x / (self.x.shape[1] * self.x.shape[2]))

                    self.b[i][j] -= self.alpha * prev_d[a] * sum_w_per_pixel # Average w?

            current_d[a] = prev_d[a] * (image_sum_weights / (self.x.shape[1] * self.x.shape[2] * self.num_neurons)) # Average the weights?

        return current_d

class softmax():
    def __init__(self, num_neurons, alpha):
        self.x = None # placeholder
        self.num_neurons = num_neurons
        # w is a 3D array which contains one w per pixel for all pixels per neuron.
        self.w = None
        self.b = [0 for _ in range(self.num_neurons)]
        self.linear_func = None
        self.output = None
        self.y = None # Placeholder
        self.alpha = alpha
        self.b = np.array(self.b, dtype=np.float64)

    def initialize_parameters(self):
        # w is a 3D array which contains one w per pixel for all pixels per neuron.
        self.w = [[[random.uniform(-0.4, 0.4) for _ in range(self.x.shape[2])] for _ in range(self.x.shape[1])] for _ in range(self.num_neurons)]
        self.linear_func = [[0 for _ in range(self.num_neurons)] for _ in range(self.x.shape[0])]
        self.output = [[0 for _ in range(self.num_neurons)] for _ in range(self.x.shape[0])]
        self.w = np.array(self.w, dtype=np.float64)

    def forward_prop(self, x):
        self.x = x  # 0 = number of photos in batch, 1 = number of rows, 2 = number of columns

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
    num_batches_per_person = total_num_photos // batch_size

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

    alpha = 0.001
    hidden1 = relu(2025, alpha)
    hidden2 = relu(1024, alpha)
    hidden3 = relu(529, alpha)
    hidden4 = relu(121, alpha)
    output_layer = softmax(number_of_people, alpha)

    run = True
    total_iteration = 0
    while run:

        for a in range(all_photos.shape[0]):
            for b in range(all_photos.shape[1]):
                a1 = hidden1.forward_prop(all_photos[a][b], False)
                a2 = hidden2.forward_prop(a1, True)
                a3 = hidden3.forward_prop(a2, True)
                a4 = hidden4.forward_prop(a3, True)
                output_layer.forward_prop(a4)
                cost, total_cost = output_layer.cost(a)
                print(f"Total iteration: {total_iteration}, current person: {a + 1} out of {all_photos.shape[0]}, "
                      f"batch number: {b + 1} out of {all_photos.shape[1]}, cost: {total_cost}")

                output_layer.gradient_descent()
                current_d = hidden4.back_prop(cost)
                current_d = hidden3.back_prop(current_d)
                current_d = hidden2.back_prop(current_d)
                current_d = hidden1.back_prop(current_d)

                if total_iteration >= 3:
                    run = False

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
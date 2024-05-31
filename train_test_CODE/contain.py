import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import random
import math

def get_images(folder_name):
        import numpy as np
        import os
        import math
        import os
        import cv2
        import numpy as np

        # Initialize an empty list to store the normalized image data
        normalized_image_data = []


        # normalized_image_data = []
        folder_path = '/home/shared-nearmrs/endovis_mono/' + str(folder_name) + "/frame_rgb/"
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp'))]

        # Iterate through the image files, load them, and normalize the data
        for image_file in image_files:
            # image = cv2.imread(os.path.join(folder_path, image_file),cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(os.path.join(folder_path, image_file))
            image = cv2.resize(image, (640, 512))
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Normalize the image data by dividing by 255
            normalized_image = image / 255.0   
            # normalized_image = image

            normalized_image_data.append(normalized_image)
            
        # normalized_image_array = np.stack(normalized_image_data, axis=0)
        normalized_image_array = np.array(normalized_image_data)
        # print(normalized_image_array.shape)
        return normalized_image_array
def get_raw_label(folder_name):
        import json
        # Initialize an empty list to store the translation matrices
        translation_matrices = []
        json_folder_path = '/home/shared-nearmrs/endovis_mono/'+ str(folder_name) + "/frame_data/"

        # Iterate through the JSON files in the folder
        for json_file in os.listdir(json_folder_path):
            if json_file.endswith('.json'):
                # Read the JSON data from the file
                with open(os.path.join(json_folder_path, json_file), 'r') as f:
                    json_data = json.load(f)

                # Extract the camera pose (translation matrix)
                camera_pose = json_data.get('camera-pose')

                if camera_pose:
                    # Convert the camera pose list into a NumPy array
                    camera_pose_matrix = np.array(camera_pose)

                    # Append the translation matrix to the list
                    translation_matrices.append(camera_pose_matrix)

        # Stack the translation matrices into a NumPy array
        translation_matrix_array = np.array(translation_matrices)
        return translation_matrix_array

##################################################

def get_data(folder_name):
    from keras.applications.vgg16 import preprocess_input
    from keras.preprocessing import image

    def get_images(folder_name):
        import numpy as np
        import os
        import math
        import os
        import cv2
        import numpy as np

        # Initialize an empty list to store the normalized image data
        normalized_image_data = []


        # normalized_image_data = []
        folder_path = '/home/shared-nearmrs/endovis_mono/' + str(folder_name) + "/frame_rgb/"
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp'))]

        # Iterate through the image files, load them, and normalize the data
        for image_file in image_files:
            # image = cv2.imread(os.path.join(folder_path, image_file),cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(os.path.join(folder_path, image_file))
            # image = cv2.resize(image, (640, 512))
            image = cv2.resize(image, (426, 341))

            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Normalize the image data by dividing by 255
            normalized_image = image / 255.0   
            # normalized_image = image

            normalized_image_data.append(normalized_image)
            
        # normalized_image_array = np.stack(normalized_image_data, axis=0)
        normalized_image_array = np.array(normalized_image_data)
        # print(normalized_image_array.shape)
        return normalized_image_array

    def stack_images(image1, image2):
        stacked_images = np.stack([image1, image2])
        return stacked_images

    img = get_images(folder_name)
    # img = preprocess_input(get_images(folder_name))

    stacked_images=[]
    for i in range(len(img)-1):
        stacked_images.append(stack_images(img[i],img[i+1]))

    return (np.array(stacked_images))

##################################################

def get_label(folder_name):

    def get_raw_label(folder_name):
        import json
        # Initialize an empty list to store the translation matrices
        translation_matrices = []
        json_folder_path = '/home/shared-nearmrs/endovis_mono/'+ str(folder_name) + "/frame_data/"

        # Iterate through the JSON files in the folder
        for json_file in os.listdir(json_folder_path):
            if json_file.endswith('.json'):
                # Read the JSON data from the file
                with open(os.path.join(json_folder_path, json_file), 'r') as f:
                    json_data = json.load(f)

                # Extract the camera pose (translation matrix)
                camera_pose = json_data.get('camera-pose')

                if camera_pose:
                    # Convert the camera pose list into a NumPy array
                    camera_pose_matrix = np.array(camera_pose)

                    # Append the translation matrix to the list
                    translation_matrices.append(camera_pose_matrix)

        # Stack the translation matrices into a NumPy array
        translation_matrix_array = np.array(translation_matrices)
        return translation_matrix_array

    def combine_translations(t1, t2):
        # Check if t1 and t2 are 4x4 matrices
        if t1.shape != (4, 4) or t2.shape != (4, 4):
            raise ValueError("Both input matrices should be 4x4 matrices")
        # Calculate the combined transformation matrix
        combined_matrix = np.matmul(t2, np.linalg.inv(t1))
        return combined_matrix
    
    def compose_transformations(matrix_A, matrix_B):
        # Extract rotation and translation components from matrices A and B
        rotation_A = matrix_A[:3, :3]
        translation_A = matrix_A[:3, 3]
        # Calculate the inverse of matrix As
        inverse_rotation_A = rotation_A.T  # Transpose of the rotation matrix
        inverse_translation_A = -np.dot(inverse_rotation_A, translation_A) 
        inverse_A = np.eye(4)
        inverse_A[:3, :3] = inverse_rotation_A
        inverse_A[:3, 3] = inverse_translation_A
        # Create the composed transformation matrix B * inverse(A)
        composed_matrix = matrix_B * inverse_A
        return composed_matrix

    import math
    def decompose_transformation_matrix(matrix):
        if matrix.shape != (4, 4):
            raise ValueError("Input matrix must be a 4x4 transformation matrix.")

        translation = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]

        # Extract rotation angles (XYZ) using Euler angles
        # Assuming the input matrix is in rotation-translation order
        # You can change the order based on your specific use case.
        sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
        singular = sy < 1e-6

        if not singular:
            x_angle = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y_angle = math.atan2(-rotation_matrix[2, 0], sy)
            z_angle = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x_angle = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y_angle = math.atan2(-rotation_matrix[2, 0], sy)
            z_angle = 0

        # Convert rotation angles to degrees
        x_angle = math.degrees(x_angle)
        y_angle = math.degrees(y_angle)
        z_angle = math.degrees(z_angle)

        return np.array([translation[0], translation[1], translation[2], x_angle, y_angle, z_angle])

    matrixes = get_raw_label(folder_name)
    stacked_matrixes=[]
    for i in range(len(matrixes)-1):
        stacked_matrixes.append(decompose_transformation_matrix(combine_translations(matrixes[i],matrixes[i+1])))
    stacked_matrixes = np.array(stacked_matrixes)
    return stacked_matrixes

###################### MODELS ######################

def get_VGG_model(X,y):
    tfk = tf.keras
    tfkl = tf.keras.layers
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)


    input_shape = X.shape[1:]
    output_shape = y.shape[1:]

    supernet1 = tfk.applications.VGG16(
        include_top=False,
        weights="imagenet",
        # weights= None,
        input_shape=input_shape[1:]
    )

    def build_model(X):
        # Use the supernet as feature extractor
        supernet1.trainable = False

        inputs = tfk.Input(shape=input_shape)
        
        x1 = tfkl.Lambda(lambda x: x[:, 0])(inputs)
        x2 = tfkl.Lambda(lambda x: x[:, 1])(inputs)

        x1 = supernet1(x1)
        x2 = supernet1(x2)

        x1 = tfkl.Flatten(name='Flattening1')(x1)
        x1 = tfkl.Dense(500, activation='relu')(x1)

        x2 = tfkl.Flatten(name='Flattening2')(x2)
        x2 = tfkl.Dense(500, activation='relu')(x2)

        x = tfkl.Concatenate(name='Concatenation')([x1, x2])
        x = tfkl.Dense(500, activation='relu')(x)
        x = tfkl.Dropout(0.35)(x)
        x = tfkl.Dense(250, activation='relu')(x)
        x = tfkl.Dense(units=6, activation='linear')(x)
        outputs = x

        # Connect input and output through the Model class
        tl_model = tfk.Model(inputs=inputs, outputs=outputs, name='model')
        tl_model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=tfk.optimizers.legacy.Adam(), metrics=['mse','mae'])
        return tl_model
    ########################
    # Compile the model

    tl_model = build_model(input_shape)
    print(tl_model.summary())

    return tl_model

##################################################

def get_resnet50_model(X,y):
    tfk = tf.keras
    tfkl = tf.keras.layers
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)


    input_shape = X.shape[1:]
    output_shape = y.shape[1:]

    supernet1 = tfk.applications.ResNet50(
        include_top=False,
        weights= "imagenet",
        # weights= None,
        input_shape=input_shape[1:]
    )

    def build_model(X):
        # Use the supernet as feature extractor
        supernet1.trainable = True

        inputs = tfk.Input(shape=input_shape)
        
        x1 = tfkl.Lambda(lambda x: x[:, 0])(inputs)
        x2 = tfkl.Lambda(lambda x: x[:, 1])(inputs)

        x1 = supernet1(x1)
        x2 = supernet1(x2)

        x1 = tfkl.Flatten(name='Flattening1')(x1)
        x1 = tfkl.Dense(500, activation='relu')(x1)

        x2 = tfkl.Flatten(name='Flattening2')(x2)
        x2 = tfkl.Dense(500, activation='relu')(x2)

        x = tfkl.Concatenate(name='Concatenation')([x1, x2])
        x = tfkl.Dense(500, activation='relu')(x)
        x = tfkl.Dropout(0.35)(x)
        x = tfkl.Dense(250, activation='relu')(x)
        x = tfkl.Dense(units=6, activation='linear')(x)
        outputs = x

        # Connect input and output through the Model class
        tl_model = tfk.Model(inputs=inputs, outputs=outputs, name='model')
        tl_model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=tfk.optimizers.legacy.Adam(), metrics=['mse','mae'])
        return tl_model
    ########################
    # Compile the model

    tl_model = build_model(input_shape)
    print(tl_model.summary())

    return tl_model

##################################################

def get_Custom_model(X,y):
    
    tfk = tf.keras
    tfkl = tf.keras.layers
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)


    input_shape = X.shape[1:]
    output_shape = y.shape[1:]

    def build_model(X):
        # Use the supernet as feature extractor
    

        inputs = tfk.Input(shape=input_shape)
        
        x1 = tfkl.Lambda(lambda x: x[:, 0])(inputs)
        x2 = tfkl.Lambda(lambda x: x[:, 1])(inputs)

        x1 = tfkl.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = tfkl.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = tfkl.MaxPooling2D()(x1)
        x1 = tfkl.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = tfkl.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = tfkl.MaxPooling2D()(x1)
        x1 = tfkl.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = tfkl.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = tfkl.MaxPooling2D()(x1)

        x2 = tfkl.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = tfkl.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = tfkl.MaxPooling2D()(x2)
        x2 = tfkl.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = tfkl.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = tfkl.MaxPooling2D()(x2)
        x2 = tfkl.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = tfkl.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = tfkl.MaxPooling2D()(x2)

        x1 = tfkl.Flatten(name='Flattening1')(x1)
        x1 = tfkl.Dense(256, activation='relu')(x1)

        x2 = tfkl.Flatten(name='Flattening2')(x2)
        x2 = tfkl.Dense(256, activation='relu')(x2)

        x = tfkl.Concatenate(name='Concatenation')([x1, x2])
        x = tfkl.Dense(128, activation='relu')(x)
        x = tfkl.Dropout(0.35)(x)
        x = tfkl.Dense(64, activation='relu')(x)
        x = tfkl.Dense(units=6, activation='linear')(x)
        outputs = x

        # Connect input and output through the Model class
        tl_model = tfk.Model(inputs=inputs, outputs=outputs, name='model')
        tl_model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=tfk.optimizers.legacy.Adam(), metrics=['mse','mae'])
        return tl_model
    ########################
    # Compile the model

    tl_model = build_model(input_shape)
    print(tl_model.summary())

    return tl_model

##################################################

def get_Thick_VGG_model(X,y):
    tfk = tf.keras
    tfkl = tf.keras.layers
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)


    input_shape = X.shape[1:]
    output_shape = y.shape[1:]

    supernet1 = tfk.applications.VGG16(
        include_top=False,
        weights="imagenet",
        # weights= None,
        input_shape=input_shape[1:]
    )

    def build_model(X):
        # Use the supernet as feature extractor
        supernet1.trainable = False
        # supernet1.trainable = True

        # supernet2.trainable = False

        inputs = tfk.Input(shape=input_shape)
        
        x1 = tfkl.Lambda(lambda x: x[:, 0])(inputs)
        x2 = tfkl.Lambda(lambda x: x[:, 1])(inputs)

        x1 = supernet1(x1)
        x2 = supernet1(x2)

        x1 = tfkl.Flatten(name='Flattening1')(x1)
        x1 = tfkl.Dense(500, activation='relu')(x1)
        x1 = tfkl.Dense(256, activation='relu')(x1)
        x1 = tfkl.Dense(128, activation='relu')(x1)

        x2 = tfkl.Flatten(name='Flattening2')(x2)
        x2 = tfkl.Dense(500, activation='relu')(x2)
        x2 = tfkl.Dense(256, activation='relu')(x2)
        x2 = tfkl.Dense(128, activation='relu')(x2)


        x = tfkl.Concatenate(name='Concatenation')([x1, x2])
        # x = tf.stack([x1, x2], axis=1)

        x = tfkl.Dense(500, activation='relu')(x)
        x = tfkl.Dense(256, activation='relu')(x)
        x = tfkl.Dense(128, activation='relu')(x)


        x = tfkl.Dense(units=6, activation='linear')(x)
        outputs = x

        # Connect input and output through the Model class
        tl_model = tfk.Model(inputs=inputs, outputs=outputs, name='model')
        tl_model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=tfk.optimizers.legacy.Adam(), metrics=['mse','mae'])
        return tl_model
    
    ########################
    # Compile the model

    tl_model = build_model(input_shape)
    print(tl_model.summary())

    return tl_model

##################################################

def plot_and_save(model,history,comment):

    from datetime import datetime
    folder_name = str(comment) + ' ' + str(datetime.now())
    import os
    os.makedirs(folder_name)

    model.save(folder_name + '/(MODEL)')
    X_test = get_data('v1')
    np.save(folder_name + '/Prediction_v1.npy', model.predict(X_test))

    
    best_epoch = np.argmin(history['val_loss'])
    plt.figure(figsize=(17,4))
    plt.plot(history['loss'], label='MSE', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_loss'], label='val_MSE', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Mean Squared Error (Loss)')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()
    plt.savefig(folder_name + '/loss.png')  # Save the plot as an image file


    plt.figure(figsize=(17,4))
    plt.plot(history['mae'], label='MAE', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_mae'], label='val_MAE', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()
    plt.savefig(folder_name + '/mae.png')  # Save the plot as an image file

    plt.figure(figsize=(17,4))
    # plt.plot(history['mae'], label='MAE', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_mae'], label='val_MAE', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()
    plt.savefig(folder_name + '/mae.png')  # Save the plot as an image file





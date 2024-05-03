import tensorflow as tf
from tensorflow.keras.applications import MobileNet
import os
import imghdr
import cv2
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import pandas as pd

data_dir = 'Train' 
image_exts = ['jpeg','jpg', 'bmp', 'png']
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            
def apply_color_distortion(image, brightness_delta=0.1, contrast_range=(0.9, 1.1), saturation_range=(0.9, 1.1), hue_delta=0.02):
    # Randomly adjust the brightness of the image
    image = tf.image.random_brightness(image, max_delta=brightness_delta)
    
    # Randomly adjust the contrast of the image
    image = tf.image.random_contrast(image, lower=contrast_range[0], upper=contrast_range[1])
    
    # Randomly adjust the saturation of the image
    image = tf.image.random_saturation(image, lower=saturation_range[0], upper=saturation_range[1])
    
    # Randomly adjust the hue of the image
    image = tf.image.random_hue(image, max_delta=hue_delta)
    
    # Clip the image to maintain pixel values between 0 and 1
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

            
def read_and_decode(filename, label, reshape_dims):
    # Read the image file
    img = tf.io.read_file(filename)
    # Decode the image as a JPEG file, specifying the number of color channels
    img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS)
    # Convert image to floating point values and normalize the pixel values to [0, 1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize the image to the specified dimensions
    img = tf.image.resize(img, reshape_dims)

    # Apply color distortion with the provided label
    img = apply_color_distortion(img)

    return img

pattern = r'train/'

print (tf.io.gfile.glob("train/*"))
CLASS_NAMES = [item.numpy().decode("utf-8") for item in
               tf.strings.regex_replace(
                 tf.io.gfile.glob("train/*"),
                 pattern, "")]

print("These are %s classes available " %(len(CLASS_NAMES)))

for index, class_name in enumerate(CLASS_NAMES, start=1):
    print(f"{index}. {class_name}")
    
cloth = tf.io.gfile.glob("train/Jacket/*.jpg")
f, ax = plt.subplots(1, 5, figsize=(32,32))
print(cloth)
for idx, filename in enumerate(cloth[:5]):
  print(filename)
  print(idx)
  img = read_and_decode(filename,CLASS_NAMES, [IMG_HEIGHT, IMG_WIDTH])
  ax[idx].imshow((img.numpy()));
  ax[idx].axis('off')
  
absolutePath = "train/"
for class_name in CLASS_NAMES:
    for ext in image_exts:
        pattern = f"{absolutePath}{class_name}/*.{ext}"
        class_paths = tf.io.gfile.glob(pattern)

        paths_df = pd.DataFrame({'path': class_paths})
        paths_df['class'] = class_name
        df = pd.concat([df, paths_df], ignore_index=True)

print(df)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(df["path"], df["class"], test_size=0.20)

X_train_df = pd.DataFrame(X_train, columns=["path"])
X_test_df = pd.DataFrame(X_test, columns=["path"])

trainData = pd.concat([X_train_df, y_train], axis=1)
testData = pd.concat([X_test_df, y_test], axis=1)

trainData.to_csv("train.csv",index=False, header=False)
testData.to_csv("test.csv",index=False, header=False)
print(trainData)
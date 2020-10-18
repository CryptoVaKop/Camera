import cv2 as cv
import torch
from torchvision import models, transforms
import torchvision.transforms.functional as tf
from PIL import Image
import sys
import json


# Create dictionary with image class names from json file
Classes = {}
with open("./ImageNet1000.json", "r") as jsonFile:
    Classes = json.loads(jsonFile.read())

# Create pretrained neural network "Resnet18"
Net = models.resnet18(pretrained=True)
Net.eval()

# Open camera
camera = cv.VideoCapture(0)

if not camera.isOpened():
    print("Could not open the camera")
    sys.exit(-1)

# Main loop. Read the frame from the camera,
# show the captured frame in the window,
# recognize the image by Resnet18
while True:

    # Read the frame from the camera
    _, frame = camera.read()
    if frame is None:
        print(" < < <  Game over!  > > > ")
        break

    # Show captured frame in the window
    cv.imshow("Camera", frame)

    # Wait for press "ESCAPE" to exit
    k = cv.waitKey(delay=10)
    if k == 27:
        break

    # Conver color space from BGR to RGB format
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Create the PIL(Python image library) image from the frame
    image = Image.fromarray(frame)

    # Prepare image
    image = tf.resize(image, 256)
    image = tf.center_crop(image, 224)
    image = tf.to_tensor(image)
    image = tf.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Regognize image
    outputs = Net(image.view(1, 3, 224, 224))
    _, preds = torch.max(outputs, 1)
    className = Classes[str(preds.item())]

    # Update window title with image class name
    cv.setWindowTitle("Camera", className)

sys.exit(0)

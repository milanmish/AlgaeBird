import os
import shutil
import tensorflow as tf
import numpy as np
import cv2
from keras.models import load_model

source_folder = r"D:/DCIM/100MEDIA"
cwd = os.path.dirname(os.path.realpath(__file__))
destination_folder = cwd + "/drone_data"

if os.path.exists(source_folder):
    print(str(source_folder))

    for file_name in os.listdir(source_folder):
        source = str(source_folder) + "\\" + file_name
        if os.path.isfile(source):
            shutil.copy(source, destination_folder)

HAB_Detect = load_model('C:/Users/25milanbm/ElectronHAB/classificationDir/models/detection.h5')

pabw = open(cwd + '/pab.txt', 'w')
npabw = open(cwd + '/npab.txt', 'w')
npabw.close()
npabw.close()

for image in os.listdir(destination_folder):
    image_path = os.path.join(destination_folder, image)
    img = cv2.imread(image_path)
    resize = tf.image.resize(img, (256,256))
    predictVal = HAB_Detect.predict(np.expand_dims(resize/255, 0))
    if predictVal > 0.5: 
        print('Predicted class likely has an algae bloom {}'.format(image_path))
        with open(cwd + '/pab.txt', 'a') as f:
            f.write(image_path + '\n')  # Only write the path to the file
    else:
        print('Predicted class likely does not have an algae bloom {}'.format(image_path))
        with open('C:/Users/25milanbm/ElectronHAB/classificationDir/npab.txt', 'a') as f:
            f.write(image_path + '\n')  # Only write the path to the file

with open('C:/Users/25milanbm/ElectronHAB/classificationDir/pab.txt', 'r') as f:
    positive_labels = f.read()

with open('C:/Users/25milanbm/ElectronHAB/classificationDir/npab.txt', 'r') as f:
    negative_labels = f.read()

# Send the output data back to the Electron main process
print(f'POSITIVE LABELS: {positive_labels}')
print(f'NEGATIVE LABELS: {negative_labels}')
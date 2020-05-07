from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

import numpy as np
from keras.applications import VGG16

print('=============== 0000 =============')
img_path = '/home/mxd/.keras/datasets/tigger.jpg'
img = image.load_img(img_path, target_size = (224, 224))

print('=============== 1111 =============')
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print('=============== 2222 =============')
model = VGG16(weights='imagenet', include_top=False)
preds = model.predict(x)

print('=============== 3333 =============')
print(preds.shape)
print('Predicted:', decode_predictions(preds, top=3)[0])

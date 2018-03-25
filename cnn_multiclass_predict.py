import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.models import load_model
import h5py

classifier = load_model('03_23_18_140PM_my_model_multiclass10.h5') #load the model that was created using cnn_multiclass.py

test_image = image.load_img('predictions/pose5.jpg', target_size = (64, 64)) #folder predictions with jpgs that I want to test
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
 
result = classifier.predict(test_image) # returns array

if result[0][0] == 1:
	prediction = 'bridge'
elif result[0][1] == 1:
	prediction = 'childspose'
elif result[0][2] == 1:
	prediction = 'downwarddog'
elif result[0][3] == 1:
	prediction = 'mountain'
elif result[0][4] == 1:
	prediction = 'plank'
elif result[0][5] == 1:
	prediction = 'seatedforwardbend'
elif result[0][6] == 1:
	prediction = 'tree'
elif result[0][7] == 1:
	prediction = 'trianglepose'
elif result[0][8] == 1:
	prediction = 'warrior1'
elif result[0][9] == 1:
	prediction = 'warrior2'

print(result)
print(prediction)



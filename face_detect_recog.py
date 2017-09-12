import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
from pattern_recog_func import interpol_im, pca_svm_pred, pca_X, rescale_pixel, svm_train, leave_one_out_test


imagePath = "./Images"

images = []
names = []

for parent_folders in listdir(imagePath):
	# print(parent_folders)
	count = 0
	for files in listdir(join(imagePath, parent_folders)):
		if ".png" in files:
			images.append(join(join(imagePath, parent_folders), files))
			count += 1
	names.append(count)

	# print(len(images))
# print(images)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cascPath = 'haarcascade_frontalface_default.xml'

# instantiate the haar cascade object
# This is basically a classifier:
# http://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
faceCascade = cv2.CascadeClassifier(cascPath)

images_prepared = []

y2 = []
for i in range(len(names)):
	for x in np.zeros(names[i]) + i:
		y2.append(x)

y3 = []

for i in range(len(images)):
	# Read the image
	# print(image)
	image = cv2.imread(images[i])  # this does the same as mpimg.imread
	# print(image)
	# Detect faces in the image
	# if small features are identified as faces, change scaleFactor to 1.2
	# watch what happens when minSize is set to (70, 70)
	faces = faceCascade.detectMultiScale(
	                                     image,
	                                     scaleFactor=1.1,
	                                     minNeighbors=5,
	                                     minSize=(30, 30),
	                                     )

	# print('faces type', type(faces))
	# print('Coordinates of faces:\n', faces)
	# print("Found {:d} faces!".format(len(faces)))
	# each entry in faces gives [x, y, w, h]
	if len(faces) != 0:
		faces = faces[0]
		x, y, w, h = faces[0], faces[1], faces[2], faces[3]

		# image = image[y:y+h, x:x+w]

		image_interp = interpol_im(image, 45, 60)
		images_prepared.append(image_interp)
		
		y3.append(y2[i])

#Create our X data and y targets
X = np.vstack(np.array(images_prepared))


y = y2
print("X shape: " + str(X.shape))
print("y length: " + str(len(y2)))

# ************************************* c) Training and Validation *************************************
md_pca, X_proj = pca_X(X, n_comp = 50)
md_clf = svm_train(X_proj, y, gamma = 0.001, C = 100)

incorrect = 0
correct = 0
size = len(X)
for i in range(size):
    predict = leave_one_out_test(X, y, i)
    if(predict == y[i]):
        correct += 1.
    else:
    	incorrect += 1.
    	# print("--------> index, actual digit, svm_prediction: {} {} {}".format(i, y[i], predict))
print("")
print("Failed {:3.0f} out of {} times.".format(incorrect, size))
print("Success Rate of {:5.1f}%".format(( (size - incorrect) / size)*100))


# exit()

names_dict = {0: 'gilbert', 1: 'janet', 2: 'luke'}

whoswho = cv2.imread("whoswho.jpg")


plt.imshow(whoswho)
plt.show()
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
    
# whoswho_gray = cv2.cvtColor(whoswho, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
                                     whoswho,
                                     scaleFactor=1.3,
                                     minNeighbors=5,
                                     minSize=(50, 50),
                                     flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                                     )
        
md_pca, X_proj = pca_X(X, n_comp = 50)
                                         
md_clf = svm_train(X_proj, y)
    
print("")                                     
for key, value in names_dict.items():
	print("Idenity of person {}: {}".format(key, value))

print("")

person = 0
for (x, y, w, h) in faces:
    im = whoswho[y:y+h, x:x+w]
    prediction = pca_svm_pred(im, md_pca, md_clf)[0]
    
    print("PCA + SVM prediction for person {}: {}".format(person, names_dict[person]))
    
    person += 1

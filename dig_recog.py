import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from pattern_recog_func import interpol_im, pca_svm_pred, pca_X, rescale_pixel, svm_train, leave_one_out_test

'''
Part I shows that an import aspect of the pattern
Otherwise even a sophisticated classifier such as SVM can be thrown off and give erroneous predictions.
'''


dig_data = load_digits()
X = dig_data.data
y = dig_data.target


dig_img = dig_data.images

'''
Part a) use the first 60 images in X to train.
Then apply md_clf to the next 20 images to perform validation.
Then print out the success rate.
'''

# Training data
md_clf = svm_train(X[0:60], y[0:60])

# Prediction data
md_predictions = md_clf.predict(X[61:81])

# True data
md_targets = y[61:81]

size = len(md_targets)
incorrect = 0.

print("Mis-identifications:")
for i in range(size):
	if(md_predictions[i] != md_targets[i]):
		incorrect += 1.
		print("--------> index, actual digit, svm_prediction: {} {} {}".format(60+i, md_targets[i], md_predictions[i]))

if incorrect == 0:
	print("None")
	
perc_correct = (size - incorrect) / size

print("Total number of mis-identifications: {:2.0f}".format(incorrect))
print("Success rate: {}".format(perc_correct))



'''
Part b) Testing Test the classifier obtained in part a) on an image taken outside of the sklearn digit
data set
'''

unseen = mpimg.imread("unseen_dig.png")[:, :, 0]
unseen_interpoled_flat = interpol_im(unseen, plot_new_im = True)

plt.imshow(dig_img[15], cmap="Greys")
plt.grid("off")
plt.title("X[15]")
plt.show()

unseen_interpoled_rescaled = rescale_pixel(X, unseen_interpoled_flat)

pred_flat = pred = md_clf.predict(unseen_interpoled_flat.reshape(1, -1))[0]
pred_rescaled = md_clf.predict(unseen_interpoled_flat.reshape(1, -1))[0]

print("")
print("Test the classifier on an outside image: ")
print("Prediction (rescaled):\t{}".format(pred_flat))
print("Prediction (non-rescaled):\t{}".format(pred_rescaled))

print("Correct Number:\t\t5")

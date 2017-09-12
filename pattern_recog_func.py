import argparse  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import interp2d

from sklearn import svm

def interpol_im(im, dim1 = 8, dim2 = 8, plot_new_im = False, cmap = 'binary', grid_off = False):

 	if((len(im.shape)) == 3):
		im = im[:, :, 0]
	# print(im.shape)

	x = np.arange(im.shape[1])
	y = np.arange(im.shape[0])

	f2d = interp2d(x, y, im)

	x_new = np.linspace(0, im.shape[1], dim1)
	y_new = np.linspace(0, im.shape[0], dim2)

	im = f2d(x_new, y_new)

	# Create a 1D array of pixels to return
	let_im_flat = im.flatten()
	
	#Plot the interpolated image if asked to	
	if plot_new_im:
		plt.grid("on")
		if grid_off:
		    plt.grid("off")
		plt.imshow(im, cmap = cmap)
		plt.show()


	#Return the interpolated, flattened image array
	return let_im_flat

def pca_svm_pred(imfile, md_pca, md_clf, dim1 = 45, dim2 = 60):

	img_interp_flat = interpol_im(imfile, dim1 = dim1, dim2 = dim2, plot_new_im = True)

	img_interp_flat = img_interp_flat.reshape(1, -1)
    
	X_proj = md_pca.transform(img_interp_flat)

	predict = md_clf.predict(X_proj)

	return predict

'''
returns md_pca and X_proj, where X_proj contains
the projections of the data array X in the PCA space
'''
def pca_X(X, n_comp = 10):
	md_pca = PCA(n_comp, whiten = True)  
	X_proj = md_pca.fit_transform(X)

	return md_pca, X_proj

def rescale_pixel(X, unseen, ind = 0):
    
	unseen_rescaled = np.array((unseen * 15), dtype=np.int)
	inverted_background_unseen = []

	for i in range(len(unseen_rescaled)):
		if(unseen_rescaled[i] == 0):
			inverted_background_unseen.append(15)
		elif unseen_rescaled[i] == 15:
			inverted_background_unseen.append(0)
		else:
			inverted_background_unseen.append(unseen_rescaled[i])

	return np.array(inverted_background_unseen)

def svm_train(X, y, gamma = 0.001, C = 100):

	# instantiating an SVM classifier
	md_clf = svm.SVC(gamma=gamma, C=C)

	# apply SVM to training data and draw boundaries.
	md_clf.fit(X, y)
	# Use SVM-determined boundaries to make
	# a prediction for the test data point.
	# clf.predict(Xtest_proj)

	return md_clf

def leave_one_out_test(X, y, select_idx, n_comp = 50):
    
	Xtest = X[select_idx].reshape(1, -1)
	ytest = np.array([y[select_idx]])
    
	Xtrain = np.delete(X, select_idx, axis = 0)
	ytrain = np.delete(y, select_idx)

	mdtrain_pca, Xtrain_proj = pca_X(Xtrain, n_comp = n_comp)
	Xtest_proj = mdtrain_pca.transform(Xtest)
	md_clf = svm_train(Xtrain_proj, ytrain)

	return md_clf.predict(Xtest_proj)
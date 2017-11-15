# Traffic Sign Classification Project
---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/number.png "Visualization"
[image2]: ./examples/grayscaled.png "Grayscaling"
[image3]: ./examples/normalized.png "Normalizing"
[image4]: ./examples/1.jpg "Traffic Sign 1"
[image5]: ./examples/2.jpg "Traffic Sign 2"
[image6]: ./examples/3.jpg "Traffic Sign 3"
[image7]: ./examples/4.jpg "Traffic Sign 4"
[image8]: ./examples/5.jpg "Traffic Sign 5"

---

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799x32x32x3
* The size of the validation set is 4410x32x32x3
* The size of test set is 12630x32x32x3
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is organized

![alt text][image1]

### Design and Test a Model Architecture

As a first step, the images are converted to grayscale because it is obtained more accuracy by processing one channel images than three channel -RGB- images. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Then,  images are normalized between 0-1 in order to make training less sensitive to the scale of the data

Here is an example of an original image and an normalized image:

![alt text][image3]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x38 	|
| RELU					|	Activation method: Rectified Linear Unit    |
| Max pooling	      	| 2x2 stride,  outputs 14x14x38 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64    |
| RELU          | Activation method: Rectified Linear Unit     |
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Fully connected		| (5x5x64,100) sized layer 	|
| RELU				| Activation method: Rectified Linear Unit 	|
| Dropout			| Regularization method: Dropout 	|
|	Hidden layer | (100x100) sized layer   	|
| RELU				| Activation method: Rectified Linear Unit 	|
| Dropout			| Regularization method: Dropout 	|
|	Output layer |	(100x43) sized layer	|
 
To train the model, two convolutional layers and two hidden layers are created. Different layer, kernel and stride sizes are used to get the best accuracy. According to studies in the literature, ReLu activation function gives better accuracy than other functions. Therefore, the layers are activated by ReLu function. Dropout is utilized as the regularization method due to successful results

The data is split three categories which are train, validation and test sets. Batch_size and epochs variable are defined to specify how many iterations will be executed. In each epoch, whole data is shuffled and train data is split into batches and trained separately and after each epoch accuracy value is obtained by using validation set. After getting satisfying results, the algorithm is executed on test set in order to see how it works on an unseen data. By this iterative method, data is trained multiple times in different orders. This method is preferred because it is more accurate than single training method.  

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.959
* test set accuracy of 0.937


### Testing the Model on New Images

Here are five German traffic signs

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield              |  Yield                  |
| Speed limit (120 km/h)  | Speed limit (20 km/h) | 
| Right-of-away at the next intersection	| Right-of-away at the next intersection	|
| Keep right	      		| Keep right	 				|
| No entry			| No entry   							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.937

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

|Image-1 | Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|- | 1.0        			| Yield  									| 
|- | 0.0     				| No vehicles							|
|- | 0.0				      | No passing							|

|Image-2 | Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|- | 0.806        			| Speed limit (20 km/h)  									| 
|- | 0.13     				| Speed limit (100 km/h)							|
|- | 0.0423				      | Speed limit (120 km/h)							|

|Image-3 | Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|- | 1.0        			| Right-of-away at the next intersection		| 
|- | 0.0     				| Pedestrians						|
|- | 0.0				      | Beware of ice/snow							|

|Image-4 | Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|- | 1.0        			| Keep right  									| 
|- | 0.0     				| Yield						|
|- | 0.0				      | No entry							|

|Image-5 | Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|- | 1.0        			| No entry 									| 
|- | 0.0     				| Turn left ahead							|
|- | 0.0				      | Stop							|

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
[image4]: ./examples/1.JPG "Traffic Sign 1"
[image5]: ./examples/2.JPG "Traffic Sign 2"
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

The algorithm classified 4 of 5 traffic signs correctly. It misclassified second image because of possible similarity of the other signs which are speed limits. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield              |  Yield                  |
| Speed limit (120 km/h)  | Speed limit (20 km/h) | 
| Right-of-away at the next intersection	| Right-of-away at the next intersection	|
| Keep right	      		| Keep right	 				|
| No entry			| No entry   							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.937

In this algorithm, 32x32 sized images are classified and the ones,which are not 32x32, resized first and classified. This approach may cause negative results because the sign might be in a small portion of the image and when it is scaled down, this sign may be unclear. On the other hand, process time will be much longer in case of using large size images

As can be seen in the table below, the algorithm gives more certainty for correct predictions

<table>
  <tr>
    <td colspan="2" align="center"><b>Image-1</b></td>
  </tr>
  <tr>
    <td>Probability</td>
    <td>Prediction</td>
  </tr>
  <tr>
    <td>1.0</td>
    <td>Yield</td>
  </tr>
  <tr>
    <td>0.0</td>
    <td>No vehicles	</td>
  </tr>
  <tr>
    <td>0.0</td>
    <td>No passing</td>
  </tr>
</table>

<table>
  <tr>
    <td colspan="2" align="center"><b>Image-2</b></td>
  </tr>
  <tr>
    <td>Probability</td>
    <td>Prediction</td>
  </tr>
  <tr>
    <td>0.806</td>
    <td>Speed limit (20 km/h)</td>
  </tr>
  <tr>
    <td>0.13</td>
    <td>Speed limit (100 km/h)</td>
  </tr>
  <tr>
    <td>0.0423</td>
    <td>Speed limit (120 km/h)</td>
  </tr>
</table>

<table>
  <tr>
    <td colspan="2" align="center"><b>Image-3</b></td>
  </tr>
  <tr>
    <td>Probability</td>
    <td>Prediction</td>
  </tr>
  <tr>
    <td>1.0</td>
    <td>Right-of-away at the next intersection</td>
  </tr>
  <tr>
    <td>0.0</td>
    <td>Pedestrians</td>
  </tr>
  <tr>
    <td>0.0</td>
    <td>Beware of ice/snow</td>
  </tr>
</table>

<table>
  <tr>
    <td colspan="2" align="center"><b>Image-4</b></td>
  </tr>
  <tr>
    <td>Probability</td>
    <td>Prediction</td>
  </tr>
  <tr>
    <td>1.0</td>
    <td>Keep right</td>
  </tr>
  <tr>
    <td>0.0</td>
    <td>Yield</td>
  </tr>
  <tr>
    <td>0.0</td>
    <td>No entry</td>
  </tr>
</table>

<table>
  <tr>
    <td colspan="2" align="center"><b>Image-4</b></td>
  </tr>
  <tr>
    <td>Probability</td>
    <td>Prediction</td>
  </tr>
  <tr>
    <td>1.0</td>
    <td>No entry</td>
  </tr>
  <tr>
    <td>0.0</td>
    <td>Turn left ahead</td>
  </tr>
  <tr>
    <td>0.0</td>
    <td>Stop</td>
  </tr>
</table>

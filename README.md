# Traffic signs recognition in TensorFlow2

This is an exercise in image recognition. The goal is to train a simple convolutional network to recognize some traffic signs.

### Data

#### Basis of the dataset

Our final dataset is the result of compilation from a number of sources. The basis is a LISA (Laboratory for Intelligent & Safe Automobiles) dataset available [here](http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html). LISA dataset is a set of frames with annotations, out of which we extracted images of traffic signs with labels. We chose to keep only 12 categories of **US traffic signs**, as they were best represented. Here is the list:

1. Added Lane
2. Left Curve
3. Right Curve
4. Keep Right
5. Lane Ends
6. Merge
7. Pedestrian Crossing
8. School
9. Signal Ahead
10. Stop
11. Stop Ahead
12. Yield

Any traffic sign which does not fall in one of the above categories (or even if it does, but not a US sign) will not be recognized by our neural net.

As some of the images were in grayscale, we had to convert everything to grayscale. Next we threw away images which were too small (some signs were as small as 6x6 pixels), and we also purged very low quality images. At the end of this stage we were left with around 5,000 images.

#### Dataset augmentation

After we formed the basis of the dataset as described above, the images were not even nearly uniformly distributed between the selected 12 categories. Some categories had noticeably smaller number of images than the other. Initial training confirmed that the dataset will benefit from augmentation, with focus of this augmentation on the under-represented categories. We augmented the dataset in three different ways:

##### 1) Youtube



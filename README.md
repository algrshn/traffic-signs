# Traffic signs recognition in TensorFlow2

This is an exercise in image recognition. The goal is to train a simple convolutional network to recognize some traffic signs.

### Data

#### Basis of the dataset

My final dataset is the result of compilation from a number of sources. The basis is a LISA (Laboratory for Intelligent & Safe Automobiles) dataset available [here](http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html). LISA dataset is a set of frames with annotations, out of which I extracted images of traffic signs with labels. I chose to keep only 12 categories of **US traffic signs**, as they were best represented. Here is the list:

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

Any traffic sign which does not fall in one of the above categories (or even if it does, but not a US sign) will not be recognized by the neural net.

As some of the images were in grayscale, I had to convert everything to grayscale. Next I threw away images which were too small (some signs were as small as 6x6 pixels), and I also purged very low quality images (yes, I had to actually look at all of them). At the end of this stage I was left with around 5,000 images.

#### Dataset augmentation

After I formed the basis of the dataset as described above, the images were not even nearly uniformly distributed between the selected 12 categories. Some categories had noticeably smaller number of images than the other. Initial training confirmed that the dataset will benefit from augmentation, with focus of this augmentation on the under-represented categories. I augmented the dataset in three different ways:

##### 1) Youtube

This proved to be time consuming, but nevertheless the most effective way. There are lost of dash cam videos on youtube. You can stop video when you see a traffic sign which you want for your dataset, make a print screen, and later (after you aggregated lots of print screens) crop and label the pictures. As the basis dataset had lots of stop signs in it, but not so many merge, lane ends, and yield signs, I focused more on highway than on city videos. I spent around 30 hours on watching youtube, and generated around 500-600 new images from that. It is not a lot percentage wise, but as those images were mostly in badly needed categories, this improved accuracy very substantially.

##### 2) Drive around and take pictures

I asked my wife to take pictures of traffic signs as we drive. We got 250 photos, out of which only under 100 were used for the selected 12 categories. We live in Canada and our left curve, right curve, pedestriant crossing signs differ from the US versions. We also deliberately took pictures of signs outside of 12 categories to use as negative examples (more on this later).

##### 3) European datasets

[Belgium traffic sign dataset](https://btsd.ethz.ch/shareddata/) is publicly available at the time of writing, [German dataset](https://benchmark.ini.rub.de?section=gtsrb&subsection=dataset) was available for downloading when this work was done (maybe you still can download it from somewhere else). As European stop signs look the same as the US, I took this category from the above datasets. That was easy, but didn't make much difference as I already had a good number of stop signs in the dataset.

After augmentation I ended up with **6,180** reasonable quality images in 12 categories.

#### Negative examples

 






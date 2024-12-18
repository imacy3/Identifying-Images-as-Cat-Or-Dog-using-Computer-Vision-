I have no idea how the images should look so I hope they're right

1 Load and Display Images
    I did it the way I did cause I copied and pasted it from the previous project :)

2 Object Recognition
    First generate a vocabulary is called which reads the file with the name of the images splitting up the class labels so the path can be called properly. Next I use skimage.features ORB and load the image. I detect_and_extract the features of the gray image getting descriptors as well. Then I preform KMeans on it to create a vocab.
    Next I train the classifier using SVM. Again I have to read the file locations but this time I save the class labels. I do the same thing with extracting features but this time with the vocab and also mapping. I use CountVectorizer from sklearn.feature_extraction.text as well. Next I fit the data and return the classifier Before calling the classify images where I predict which label it should be with.

3 Image Segmentation
    The first time I do is grayscale the image and make it an array. Next I threshold the image to get the binary image using a high threshold of 150 and a low threshold of 50. I find strong and weak edges before I go through and do hysteresis on it in a nested for loop using a stack keeping track of what I visited already and what I haven't visited. 
    For grow regions I a seed of (100, 100) and a threshold of 160. I then find the shape of the image and while I still have points to visit I go through and find the neighbors. Once I find the neighbors I go through and check if it has been visited and is similar to the seed. If it is I add it to a list. I then return the image of those points. 
    For splitting/merging the region I find the mean of the image and use that as a threshold making anything above the threshold 255 and anything below or equal to 100. To merge call the split regions function before doing a for loop to go through the labels and find the mean intensity of the region. Next I do some sorbel math and find the neighbor info. If their absolute value is less than the threshold I set the merged_region of the neighbor to the label. 

4 Image Segmentation w/ K-Means
    I didn't do this. It didn't work for me :)
#Author: Ilana Macy
#Date: 5/7/2024
#Prof: Dr. Emily Hand
#Project 3 Computer Vision (CS485)

import PIL
from PIL import ImageDraw, ImageFont
import PIL.Image
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from skimage.feature import ORB
from sklearn.feature_extraction.text import CountVectorizer


def load_img(file_name):
    img = PIL.Image.open(file_name)
    return img.convert('L')

def display_img(image):
    if not isinstance(image, (PIL.JpegImagePlugin.JpegImageFile, PIL.Image.Image, matplotlib.text.Text)):
        image = PIL.Image.fromarray(image)
        image.show()
    elif isinstance(image, matplotlib.text.Text):
        temp_fig = 'temp_fig.png'
        plt.savefig(temp_fig, format='png', bbox_inches='tight', pad_inches=0)
        image = PIL.Image.open(temp_fig)
        image.show()
        os.remove(temp_fig)
    elif isinstance(image, (PIL.JpegImagePlugin.JpegImageFile, PIL.Image.Image)):
        image.show()

'''
    OBJECT RECOGNITION
'''

def generate_vocabulary(train_data_file):
    num_clusters=100
    with open(train_data_file, 'r') as file:
        image_paths = file.readlines()
    image_paths = [path.split()[0] for path in image_paths]

    all_descriptors = []
    orb = ORB(n_keypoints=100)
    for image_path in image_paths:
        image = load_img(image_path) 
        gray = image.convert('L')   
        orb.detect_and_extract(gray)
        all_descriptors.extend(orb.descriptors)

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(all_descriptors)
    visual_vocabulary = kmeans.cluster_centers_
    return visual_vocabulary


def extract_features(image, vocabulary):
    image = image.convert('L')
    orb = ORB(n_keypoints=100)
    orb.detect_and_extract(image)
    descriptors = orb.descriptors
    image_descriptors = [' '.join(map(str, descriptor)) for descriptor in descriptors]
    vocabulary = [' '.join(map(str, word)) for word in vocabulary]

    vectorizer = CountVectorizer(vocabulary=vocabulary)
    feature_vect = vectorizer.fit_transform(image_descriptors).toarray().flatten()

    return feature_vect

def train_classifier(train_data_file, vocab):
    classifier_type='svm'
    with open(train_data_file, 'r') as file:
        lines = file.readlines()
    image_paths = [line.split()[0] for line in lines]
    labels = [int(line.split()[1]) for line in lines]

    features = []
    for image_path in image_paths:
        image = load_img(image_path)  
        feature_vect = extract_features(image, vocab)
        features.append(feature_vect)
    
    if classifier_type == 'svm':
        classifier = SVC(kernel='linear', C=1.0)
    elif classifier_type == 'perceptron':
        classifier = Perceptron()
    elif classifier_type == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=5)

    classifier.fit(features, labels)
    return classifier

def classify_image(classifier, test_img, vocabulary):
    features = extract_features(test_img, vocabulary)
    features = features.reshape(1, -1) 
    predicted_class = classifier.predict(features)
    return predicted_class[0] 

'''
    IMAGE SEGMENTATION
'''

def threshold_image(image):
    high_thresh = 150
    low_thresh = 50
    strong_edges = image > high_thresh
    weak_edges = (image >= low_thresh) & (image <= high_thresh)

    #hysteresis
    visited = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if strong_edges[i, j] and not visited[i, j]:
                visited[i, j] = 1
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                                if weak_edges[nx, ny] and not visited[nx, ny]:
                                    strong_edges[nx, ny] = True
                                    visited[nx, ny] = True
                                    stack.append((nx, ny))
    return strong_edges

def grow_regions(image):
    seed = (100, 100)
    threshold = 160
    print(np.max(image), " ", np.min(image))
    height, width = image.shape
    segmented = np.zeros_like(image)
    visited = np.zeros_like(image, dtype=bool)
    visited[seed] = True
    points_to_visit = [seed]
    
    while points_to_visit:
        current_point = points_to_visit.pop()
        
        neighbors = [
            (current_point[0] + 1, current_point[1]),
            (current_point[0] - 1, current_point[1]),
            (current_point[0], current_point[1] + 1),
            (current_point[0], current_point[1] - 1)
        ]
        
        for neighbor in neighbors:
            if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
                if not visited[neighbor] and abs(int(image[neighbor]) - int(image[seed])) < threshold:
                    visited[neighbor] = True
                    points_to_visit.append(neighbor)
                    segmented[neighbor] = image[neighbor]
    return segmented

def split_regions(image):
    threshold = np.mean(image)
    regions = np.zeros_like(image, dtype=np.uint8)
    regions[image > threshold] = 255
    regions[image <= threshold] = 100

    return regions


def merge_regions(image):
    '''merged_img = np.copy(image)
    return merged_img.astype(np.uint8)'''
    threshold = 10  # Adjust the threshold as needed
    image = image.astype(np.float64)

    regions = split_regions(image)
    merged_regions = np.copy(regions)

    for label in np.unique(regions)[1:]:
        region_mask = regions == label
        region_mean_intensity = np.mean(image[region_mask])

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor_mask = np.roll(region_mask, (dx, dy), axis=(0, 1))
                neighbor_mean_intensity = np.mean(image[neighbor_mask])
                
                if abs(region_mean_intensity - neighbor_mean_intensity) < threshold:
                    merged_regions[neighbor_mask] = label

    return merged_regions

def segment_image(image):
    img = image.convert('L')
    img = np.array(img)
    
    binary_image = threshold_image(img)
    binary_image_with_text = add_text_to_image(binary_image, "Binary Image", 'black')

    region_map_grow = grow_regions(img)
    grow_image_with_text = add_text_to_image(region_map_grow, "Grow Image", 'white')

    #region_map_split = split_regions(img)

    region_map_merge = merge_regions(img)
    merge_image_with_text = add_text_to_image(region_map_merge, "Merge Image", 'black')

    return binary_image_with_text, grow_image_with_text, merge_image_with_text
   
'''
it wasn't working and i honestly don't care :)
def kmeans_segment(image):
    image.show()
    image = image.convert('L')

    flattened_image = np.array(image)

    flattened_image = flattened_image.reshape((-1, 1))

    silhouette_scores = []
    print('flattened_img: ', flattened_image)
    print(type(flattened_image))
    img = PIL.Image.fromarray(flattened_image)
    img.show()

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(flattened_image)
        silho = silhouette_score(flattened_image, labels)
        print('silho score for k: ', k, " : ", silho)
        silhouette_scores.append(silho)
    optimal_k = np.argmax(silhouette_scores) + 2
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    labels = kmeans.fit_predict(flattened_image)
    print("optimal: ", optimal_k)
    print("cluster_centers: ", kmeans.cluster_centers_)
    print("inertia: ", kmeans.inertia_)
    print(labels.shape)

    #if labels.size == flattened_image.shape[0] * flattened_image.shape[1]:
    segmented_image = labels.reshape(image.size[::-1])#, flattened_image.shape[1])
    return segmented_image
'''

def add_text_to_image(image, text, color):
    image = PIL.Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((10, 10), text, fill=color, font=font)
    return image
#!/usr/bin/env python
# coding: utf-8


import numpy as np
from PIL import Image
from scipy.spatial import distance
from time import perf_counter
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans


def get_RGB_pixels(file_name):
    '''This functions takes an image and converts it into a numpy array with size(n, 3), where n is the
    number of pixels in the image and each row is the Red, Green, and Blue components of the pixel respectively'''
    assert isinstance(file_name, str), "File name should be a string."
    
    # Open the image with PIL
    img = Image.open(file_name)
    
    # Display the img
    display(img)
    
    # Convert the image to RGB components
    RGB_components = np.array(img.getdata()).astype(int)
    
    return RGB_components


# Beach picture RGB components
beach_RGB = get_RGB_pixels('beach.bmp')

# Football RGB components
football_RGB = get_RGB_pixels("football.bmp")

# Here are the RGB components for a picture of my dog
brooks_RGB = get_RGB_pixels("brooks.jpg")


def testval_in(RGB_array, test_array, indx):
    '''This function tests an index of the RGB component array to see if that index or that RGB
    component is already in the array. The purpose is to ensure that the medoids are unique.'''
    
    val_array = RGB_array[test_array]
    if indx in test_array:
        return False # The index is in the medoids array
    elif RGB_array[indx] in val_array:
        return False # The RGB component value is in the medoids array
    else:
        return True


def initial_centers_kminus1(RGB_array, k):
    '''This function selects the center most point as the first cluster center by finding the point
    that is closest to the mean. The other k-1 cluster centers are chosen randdomly and tested to make 
    sure that all cluster centers are unique'''
    
    # Get the shape of the RGB components array
    RGB_shape = RGB_array.shape
    
    # The first cluster medoid will be the point closest to the mean
    mean = np.mean(RGB_array, axis = 0)
    center_indx = np.array([np.argmin(distance.cdist(mean.reshape(1,3), RGB_array))])
    
    # Make sure that none of the centers are duplicates
    
    while center_indx.shape[0] < k:
        rand_indx = np.random.choice(RGB_array.shape[0], 1)
        if testval_in(RGB_array, center_indx, rand_indx):
            center_indx = np.append(center_indx, rand_indx)
    
    return center_indx

def initial_centers_allrand(RGB_array, k):
    '''This function randomly selects all cluster centers.'''
    
    # Get the shape of the RGB components array
    RGB_shape = RGB_array.shape
    
    # Select a random point
    center_indx = np.random.choice(beach_RGB.shape[0], 1)
    
    # Make sure that none of the centers are duplicates
    while center_indx.shape[0] < k:
        rand_indx = np.random.choice(RGB_array.shape[0], 1)
        if testval_in(RGB_array, center_indx, rand_indx):
            center_indx = np.append(center_indx, rand_indx)
    
    return center_indx


def calc_class_labels(RGB_array, centers, d_metric = 'euclidean'):  
    '''This function assigns data points labels to each cluster based on the specified distance metric.'''
    
    # Use the numpy argmin function to determine the cluster center the data point is closesst to 
    # and assign it that label
    
    class_labels = np.argmin(distance.cdist(RGB_array, RGB_array[centers], metric = d_metric), axis = 1)
    
    return class_labels


def calc_new_centers(RGB_array, old_centers, labels, d_metric = 'euclidean'):
    '''This function calculates new cluster centers using the partiotioning around medoids (PAM) algorithm. The
    algorithm computes the distance between each current cluster center and their assigned data points, then swaps
    each data point with the cluster center and perfroms the same distance calculation. If the swapped data point
    has a smaller distance to all other points within the cluster that data point is kept as the new cluster center.'''
    
    # Create an array to store new centers by copying current centers 
    new_centers = np.copy(old_centers)
    
    # Get the number of k clusters
    k = new_centers.shape[0]
    
    # Loop through each cluster center    
    for c in range(k):
        
        # Get an array of all the data points in the cluster
        cluster_indx_arr = np.where(labels == c)[0]
        
        # Calculate the sum of the distances between the cluster center and all points within that cluster
        cluster_dist = np.sum(distance.cdist(RGB_array[new_centers[c]].reshape(1,3), RGB_array[cluster_indx_arr], metric = d_metric))
        
        # Now we loop through each data point and swap the point with the cluster center and
        # perofrm the same sum of distance calculations.
        for i in cluster_indx_arr:
            
            # Calculate the sum of distance between the point and all other points in the cluster
            i_dist = np.sum(distance.cdist(RGB_array[i].reshape(1,3), RGB_array[cluster_indx_arr], metric = d_metric))
            
            # If the clusters sum of distance to points is greater than the points sum of distance
            # we swap them and the point becomes the new cluster center
            if cluster_dist > i_dist:    
                
                # Update cluster center
                new_centers[c] = i
                
                # Update cluster center sum of distance
                cluster_dist = i_dist
    
    return new_centers


def terminate_alg(old_centers, new_centers):
    '''This function compares the RGB components of the old centers to the RGB components of the current
    centers to see if the algorithm has converged. If the RGB components are the same then it has converged.'''
    
    return np.array_equiv(np.sort(old_centers,axis=0), np.sort(new_centers,axis=0))

def show_image(RGB_array, centers, labels, height, width, new_file_name):
    '''This function takes the cluster centers and labels and assigns each data point within the cluster
    to teh cluster center RGB components to decompress the image. It then displays the image and saves it.'''
    
    image_array = np.empty((len(labels), 3))
    
    for i, x in enumerate(labels):
        image_array[i] = RGB_array[centers[x]]
    
    image_array = image_array.reshape((height, width, 3))
    
    image_array = image_array.astype(int)
    fig = plt.figure()
    plt.imshow(image_array)
    plt.axis('off')
    plt.savefig(new_file_name, transparent = True, bbox_inches = 'tight', pad_inches = 0)
    plt.show()

def k_medoids1(RGB_array, k, d_metric = 'euclidean'):
    '''This function initiates the K-medoids algortihm, by putting all of the functions together. To initialize we use the 
    k-1 method of chossing centers where one of the centers is specifically chosen as the most central point. 
    Function returns the class labels and the centers.'''
    
    # Start the timer to track how long it takes for the algorithm to converge
    t1_start = perf_counter()
    
    # Initialize the cluster centers
    cetners = initial_centers_kminus1(RGB_array, k)
    
    # Set the terminate variable to False, will be used to terminate algorithm when 
    # convergence crieria are met
    terminate = False
    
    # Track the iterations of the algorithm
    iterations = 0
    
    # Loop through each step of the algorithm until convergence
    while terminate == False:

        # Assign class labels
        class_labels = calc_class_labels(RGB_array, cetners, d_metric)
        
        # Store a copy of the current centers to test for convergence
        old_cetners = np.copy(cetners)
        
        # Calculate new centers
        cetners = calc_new_centers(RGB_array, old_cetners, class_labels, d_metric)
        
        # Check the old centers with new centers to see if we can terminate the algorithm
        terminate = terminate_alg(RGB_array[old_cetners], RGB_array[cetners])
        
        # Count the iterations of the algorithm for performance comparison purposes
        iterations += 1
    
    # Stop the timer
    t1_stop = perf_counter()
    
    # Calc minutes to convergence
    converge_mins = round((t1_stop - t1_start)/60, 2)
    
    
    print('{} Clusters'.format(k))
    print('Distance Metric = ', d_metric)
    print("It took {} iterations {} minutes for the algorithm to converge.".format(iterations, converge_mins))
    
    return class_labels, cetners


def k_medoids2(RGB_array, k, d_metric = 'euclidean'):
    '''This function initiates the K-medoids algortihm, by putting all of the functions together. To initialize we use the 
    of randomly choosing ALL cluster centers. Function returns the class labels and the centers.'''
    
    # Start the timer to track how long it takes for the algorithm to converge
    t1_start = perf_counter()
    
    # Initialize the cluster centers
    cetners = initial_centers_allrand(RGB_array, k)
    
    # Set the terminate variable to False, will be used to terminate algorithm when 
    # convergence crieria are met
    terminate = False
    
    # Track the iterations of the algorithm
    iterations = 0
    
    # Loop through each step of the algorithm until convergence
    while terminate == False:

        # Assign class labels
        class_labels = calc_class_labels(RGB_array, cetners, d_metric)
        
        # Store a copy of the current centers to test for convergence
        old_cetners = np.copy(cetners)
        
        # Calculate new centers
        cetners = calc_new_centers(RGB_array, old_cetners, class_labels, d_metric)
        
        # Check the old centers with new centers to see if we can terminate the algorithm
        terminate = terminate_alg(RGB_array[old_cetners], RGB_array[cetners])
        
        # Count the iterations of the algorithm for performance comparison purposes
        iterations += 1
    
    # Stop the timer
    t1_stop = perf_counter()
    
    # Calc minutes to convergence
    converge_mins = round((t1_stop - t1_start)/60, 2)
    
    
    print('{} Clusters'.format(k))
    print('Distance Metric = ', d_metric)
    print("It took {} iterations {} minutes for the algorithm to converge.".format(iterations, converge_mins))
    
    return class_labels, cetners

# Beach 

labels_beach_4_E, medoids_beach_4_E = k_medoids1(beach_RGB, 4, "euclidean")
labels_beach_8_E, medoids_beach_8_E = k_medoids1(beach_RGB, 8, "euclidean")
labels_beach_16_E, medoids_beach_16_E = k_medoids1(beach_RGB, 16, "euclidean")
labels_beach_32_E, medoids_beach_32_E = k_medoids1(beach_RGB, 32, "euclidean")


# Football
labels_football_4_E, medoids_football_4_E = k_medoids1(football_RGB, 4, "euclidean")
labels_football_8_E, medoids_football_8_E = k_medoids1(football_RGB, 8, "euclidean")
labels_football_16_E, medoids_football_16_E = k_medoids1(football_RGB, 16, "euclidean")
labels_football_32_E, medoids_football_32_E = k_medoids1(football_RGB, 32, "euclidean")


# Brooks k-medoids1
labels_brooks_4_E, medoids_brooks_4_E = k_medoids1(brooks_RGB, 4, "euclidean")
labels_brooks_8_E, medoids_brooks_8_E = k_medoids1(brooks_RGB, 8, "euclidean")
labels_brooks_16_E, medoids_brooks_16_E = k_medoids1(brooks_RGB, 16, "euclidean")
labels_brooks_32_E, medoids_brooks_32_E = k_medoids1(brooks_RGB, 32, "euclidean")


# Brooks k-medoids2
labels_brooks2_4_E, medoids_brooks2_4_E = k_medoids2(brooks_RGB, 4, "euclidean")
labels_brooks2_8_E, medoids_brooks2_8_E = k_medoids2(brooks_RGB, 8, "euclidean")
labels_brooks2_16_E, medoids_brooks2_16_E = k_medoids2(brooks_RGB, 16, "euclidean")
labels_brooks2_32_E, medoids_brooks2_32_E = k_medoids2(brooks_RGB, 32, "euclidean")


# Show/Save Beach
show_image(beach_RGB, medoids_beach_4_E, labels_beach_4_E, 214, 320,"beach_4_E.jpg")
show_image(beach_RGB, medoids_beach_8_E, labels_beach_8_E, 214, 320,"beach_8_E.jpg")
show_image(beach_RGB, medoids_beach_16_E, labels_beach_16_E, 214, 320,"beach_16_E.jpg")
show_image(beach_RGB, medoids_beach_32_E, labels_beach_32_E, 214, 320,"beach_32_E.jpg")

# Show/Save football
show_image(football_RGB, medoids_football_4_E, labels_football_4_E, 214, 320,"football_4_E.jpg")
show_image(football_RGB, medoids_football_8_E, labels_football_8_E, 214, 320,"football_8_E.jpg")
show_image(football_RGB, medoids_football_16_E, labels_football_16_E, 214, 320,"football_16_E.jpg")
show_image(football_RGB, medoids_football_32_E, labels_football_32_E, 214, 320,"football_32_E.jpg")

# Show/Save Brooks
show_image(brooks_RGB, medoids_brooks_4_E, labels_brooks_4_E, 267, 200,"brooks_4_E.jpg")
show_image(brooks_RGB, medoids_brooks_8_E, labels_brooks_8_E, 267, 200,"brooks_8_E.jpg")
show_image(brooks_RGB, medoids_brooks_16_E, labels_brooks_16_E, 267, 200,"brooks_16_E.jpg")
show_image(brooks_RGB, medoids_brooks_32_E, labels_brooks_32_E, 267, 200,"brooks_32_E.jpg")

# Now let's compare to K-means

def show_image_kmeans(RGB_array, centers, labels, height, width, new_file_name):
    '''This function takes the cluster centers and labels and assigns each data point within the cluster
    to teh cluster center RGB components to decompress the image. It then displays the image and saves it.'''
    
    image_array = np.empty((len(labels), 3))
    
    for i, x in enumerate(labels):
        image_array[i] = centers[x]
    
    image_array = image_array.reshape((height, width, 3))
    
    image_array = image_array.astype(int)
    fig = plt.figure()
    plt.imshow(image_array)
    plt.axis('off')
    plt.savefig(new_file_name, transparent = True, bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    

kmeans_brooks4 = KMeans(n_clusters=4).fit(brooks_RGB)
kmeans_brooks8 = KMeans(n_clusters=8).fit(brooks_RGB)
kmeans_brooks16 = KMeans(n_clusters=16).fit(brooks_RGB)
kmeans_brooks32 = KMeans(n_clusters=32).fit(brooks_RGB)

kmeans_brooks4_labels,kmeans_brooks4_centers = kmeans_brooks4.labels_, kmeans_brooks4.cluster_centers_
kmeans_brooks4_centers = kmeans_brooks4_centers.astype(int)
kmeans_brooks8_labels,kmeans_brooks8_centers = kmeans_brooks8.labels_, kmeans_brooks8.cluster_centers_
kmeans_brooks8_centers = kmeans_brooks8_centers.astype(int)
kmeans_brooks16_labels,kmeans_brooks16_centers = kmeans_brooks16.labels_, kmeans_brooks16.cluster_centers_
kmeans_brooks16_centers = kmeans_brooks16_centers.astype(int)
kmeans_brooks32_labels,kmeans_brooks32_centers = kmeans_brooks32.labels_, kmeans_brooks32.cluster_centers_
kmeans_brooks32_centers = kmeans_brooks32_centers.astype(int)

show_image_kmeans(brooks_RGB, kmeans_brooks4_centers, kmeans_brooks4_labels, 267, 200,"kmeans_brooks4.jpg")
show_image_kmeans(brooks_RGB, kmeans_brooks8_centers, kmeans_brooks8_labels, 267, 200,"kmeans_brooks8.jpg")
show_image_kmeans(brooks_RGB, kmeans_brooks16_centers, kmeans_brooks16_labels, 267, 200,"kmeans_brooks16.jpg")
show_image_kmeans(brooks_RGB, kmeans_brooks32_centers, kmeans_brooks32_labels, 267, 200,"kmeans_brooks32.jpg")

kmeans_beach4 = KMeans(n_clusters=4).fit(beach_RGB)
kmeans_beach8 = KMeans(n_clusters=8).fit(beach_RGB)
kmeans_beach16 = KMeans(n_clusters=16).fit(beach_RGB)
kmeans_beach32 = KMeans(n_clusters=32).fit(beach_RGB)

kmeans_beach4_labels,kmeans_beach4_centers = kmeans_beach4.labels_, kmeans_beach4.cluster_centers_
kmeans_beach4_centers = kmeans_beach4_centers.astype(int)
kmeans_beach8_labels,kmeans_beach8_centers = kmeans_beach8.labels_, kmeans_beach8.cluster_centers_
kmeans_beach8_centers = kmeans_beach8_centers.astype(int)
kmeans_beach16_labels,kmeans_beach16_centers = kmeans_beach16.labels_, kmeans_beach16.cluster_centers_
kmeans_beach16_centers = kmeans_beach16_centers.astype(int)
kmeans_beach32_labels,kmeans_beach32_centers = kmeans_beach32.labels_, kmeans_beach32.cluster_centers_
kmeans_beach32_centers = kmeans_beach32_centers.astype(int)


show_image_kmeans(beach_RGB, kmeans_beach4_centers, kmeans_beach4_labels, 214, 320,"kmeans_beach4.jpg")
show_image_kmeans(beach_RGB, kmeans_beach8_centers, kmeans_beach8_labels, 214, 320,"kmeans_beach8.jpg")
show_image_kmeans(beach_RGB, kmeans_beach16_centers, kmeans_beach16_labels, 214, 320,"kmeans_beach16.jpg")
show_image_kmeans(beach_RGB, kmeans_beach32_centers, kmeans_beach32_labels,  214, 320,"kmeans_beach32.jpg")

kmeans_football4 = KMeans(n_clusters=4).fit(football_RGB)
kmeans_football8 = KMeans(n_clusters=8).fit(football_RGB)
kmeans_football16 = KMeans(n_clusters=16).fit(football_RGB)
kmeans_football32 = KMeans(n_clusters=32).fit(football_RGB)

kmeans_football4_labels,kmeans_football4_centers = kmeans_football4.labels_, kmeans_football4.cluster_centers_
kmeans_football4_centers = kmeans_football4_centers.astype(int)
kmeans_football8_labels,kmeans_football8_centers = kmeans_football8.labels_, kmeans_football8.cluster_centers_
kmeans_football8_centers = kmeans_football8_centers.astype(int)
kmeans_football16_labels,kmeans_football16_centers = kmeans_football16.labels_, kmeans_football16.cluster_centers_
kmeans_football16_centers = kmeans_football16_centers.astype(int)
kmeans_football32_labels,kmeans_football32_centers = kmeans_football32.labels_, kmeans_football32.cluster_centers_
kmeans_football32_centers = kmeans_football32_centers.astype(int)

show_image_kmeans(football_RGB, kmeans_football4_centers, kmeans_football4_labels, 412, 620,"kmeans_football4.jpg")
show_image_kmeans(football_RGB, kmeans_football8_centers, kmeans_football8_labels, 412, 620,"kmeans_football8.jpg")
show_image_kmeans(football_RGB, kmeans_football16_centers, kmeans_football16_labels, 412, 620,"kmeans_football16.jpg")
show_image_kmeans(football_RGB, kmeans_football32_centers, kmeans_football32_labels, 412, 620,"kmeans_football32.jpg")




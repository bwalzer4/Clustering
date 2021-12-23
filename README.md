# Clustering
Practical implementation of clustering algorithms.

## Image Compression using K-mean and K-Medoids Clustering

Image compression is a type of method to reduce the storage and transmission size of a digital image. There are numerous compression techniques, but this project explores using  clustering algorithms to compress the pixels of a digital image. A mnaully implemented K-medoids clustering model is compared against scikit-learn K-means clustering.

### K-Medoids

K-medoids clustering algorithm is an unsupervised learning approach that partitions data points into groups by minimizing the distance between the center of the group and each data point in that group. In K-medoids the center is a data point from the data set known as the medoid. Using K-medoids we can take a digital images Red, Green, and Blue (RGB) pixels and reduce the number of RGB colors in the image to the number of clusters. Each pixel will be assigned a cluster which has a representative RGB color. For example, if we have an image with 10,000 pixels there may be 2,500 unique RGB colors in the image, by applying a K-medoids algorithm we can reduce the number of RGB colors to k clusters. The smaller k is the smaller the compressed image storage size, but the image quality is poorer, and vice versa the larger k is the larger the compressed image size, and better image quality.

The following steps detail the manual implementation of the K-medoids algorithm. The algorithm empolys a Partitionaing Around Medoids (PAM) technique that calculates new cluster centers by computing the distance between each current cluster center and their assigned data points, then swapping each data point with the center and recalculating the distance. If the swapped data point has a smaller distance to all other points within the cluster than that data point is kept as the new cluster center.

1. Extract RGB pixels from a digital image
2. Initialize 1 cluster center to be the mean (warm start) of the data points, initialize the other k-1 cluster centers randomly
3. Partition data points into each cluster using Euclidean Distance (L2 norm)
4. Identify new cluster center using PAM
5. Repeat Steps 3-4 until algorithm terminates when old cluster centers are the same as new cluster centers

### K-Means

K-means clustering algorithm is another unsupervised laearning approach that is nearly indentical to K-medoids, with the exception that the cluster centers are chosen using the mean of each cluster. Using the mean of the data points within a cluster to chose the centroid often results in the cluster center not being one of the data points from the data set. This can be problematic when working with categorical data as mean center may not make sense. Another difference between the two methods is the computational complexity. Since K-means uses a simple mean calculation to determine the center of the cluster it requires much less computation time than K-medoids, which has to perform the PAM algorithm to determine cluster centers.

### Image Compression Results

Three images of varying sizes were chosen to compare performance on each method. The figure below shows the original images and the compressed images at cluster sizes of 4, 8, 16, and 32.

 <p align="center">
  </p>
<figure>
  <p align="center">
    <img src="https://github.com/bwalzer4/Clustering/blob/main/visuals/beach.bmp?raw=True" />
  </p>
</figure>

 <p align="center">
  </p>
<figure>
  <p align="center">
    <img src="https://github.com/bwalzer4/Clustering/blob/main/visuals/football.bmp?raw=True" />
  </p>
</figure>

 <p align="center">
  </p>
<figure>
  <p align="center">
    <img src="https://github.com/bwalzer4/Clustering/blob/main/visuals/brooks.jpg?raw=True" />
  </p>
</figure>


# MachineLearningAlgorithmsCPP
Translate python code in the book "Machine Learning in Action" to C++
* Use Eigen library to process matrix.

## k-NN
1. **createDataSet()**
  * Create a sample dataset.
2. **file2matrix(filename)**
  * Read data from file.
3. **mat2map()**
  * Convert `Eigen::MatrixXf` to `std::map<std::vector<double>, int>`.
4. **euclidean_distance()**
  * Calculate the Euclidean Distance between 2 `std::vector<double>`.
5. **classify0()**
  * Classify the input vector with kNN algorithm.
6. **autoNorm()**
  * Data normalization.
7. **datingClassTest()**
  * Split dataset into traning and testing datasets.
  * Evaluate the classifier.

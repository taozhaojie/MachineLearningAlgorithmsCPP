# MachineLearningAlgorithmsCPP
Translate python code in the book "Machine Learning in Action" to C++
* Use Eigen library to process matrix.
* Use boost library.

## k-NN
1. **createDataSet()**
  * Create a sample dataset.
2. **file2matrix()**
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

## Decision Tree
1. **createDataSet()**
  * Create a sample dataset.
2. **calcShannonEnt()**
  * Calculate the Shannon Entropy.
3. **splitDataSet()**
  * Split the dataset.
4. **chooseBestFeatureToSplit()**
  * Split the dataset, and choose the feature that can have highest information gain.
5. **majorityCnt()**
  * Return the class with highest frequency.
6. **createTree()**
  * Create a decision tree, store the data in `std::map<int, boost::any>`.

## Naive Bayes

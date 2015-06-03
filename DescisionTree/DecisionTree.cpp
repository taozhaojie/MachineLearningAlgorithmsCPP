#include <iostream>
#include <fstream>
#include <utility>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <cstddef>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>

using namespace std;

class DecisionTree {
private:
	Eigen::MatrixXf dataMat; // the matrix of dataset
	int nrow; // matrix row number
	int ncol; // matrix column number

public:
	int idx_label; // index of the column contains tables

	// create sample dataset
	void createDataSet()
	{
		// 1 for yes, 0 for no, 2 for maybe
		dataMat.resize(0,0);
		dataMat.resize(5,3);
		dataMat(0,0) = 1; dataMat(0,1) = 1; dataMat(0,2) = 1;
		dataMat(1,0) = 1; dataMat(1,1) = 1; dataMat(1,2) = 1;
		dataMat(2,0) = 1; dataMat(2,1) = 0; dataMat(2,2) = 0;
		dataMat(3,0) = 0; dataMat(3,1) = 1; dataMat(3,2) = 0;
		dataMat(4,0) = 0; dataMat(4,1) = 1; dataMat(4,2) = 0;
		nrow = dataMat.rows();
		ncol = dataMat.cols();
	}

	double calcShannonEnt(Eigen::MatrixXf &mat)
	{
		int numEntries = mat.rows();
		std::map<int,int> labelCounts;
		for (int i = 0; i < numEntries; ++i)
		{
			Eigen::VectorXf featVec = mat.row(i);
			int currentLabel = (int)featVec(mat.cols()-1);
			if (labelCounts.find(currentLabel) == labelCounts.end())
				labelCounts[currentLabel] = 1;
			else
				labelCounts[currentLabel] ++;
		}
		double shannonEnt = 0.0;
		for (std::map<int, int>::iterator it = labelCounts.begin();
			it != labelCounts.end(); ++it)
		{
			double prob = (it->second)/(double)numEntries;
			shannonEnt -= prob * log2(prob);
		}
		return shannonEnt;
	}

	Eigen::MatrixXf splitDataSet(int axis, double value)
	{
		Eigen::MatrixXf retDataSet;
		std::vector<vector<double>> temp;

		for (int i = 0; i < nrow; ++i)
		{
			Eigen::VectorXf v1 = dataMat.row(i);
			std::vector<double> featVec(v1.data(), v1.data() + v1.size());
			if (featVec[axis] == value)
			{
				featVec.erase(featVec.begin() + axis);
				temp.push_back(featVec);
			}
		}
		retDataSet.resize(temp.size(),(ncol-1));

		for(unsigned int i = 0; i < temp.size(); ++i)
		{
			for(int j = 0; j < (ncol-1); ++j)
			{
				retDataSet(i,j) = temp[i][j];
			}
		}
		return retDataSet;
	}

	int chooseBestFeatureToSplit()
	{
		int numFeatures = ncol - 1;
		double baseEntropy = calcShannonEnt(dataMat);
		double bestInfoGain = 0.0;
		int bestFeature = -1;
		for (int i = 0; i < numFeatures; ++i)
		{
			Eigen::VectorXf featList = dataMat.col(i);
			std::set<double> uniqueVals;
			for (int j = 0; j < nrow; ++j)
			{
				uniqueVals.insert(featList(j));
			}
			double newEntropy = 0.0;

			for (double value : uniqueVals)
			{
				Eigen::MatrixXf subDataSet = splitDataSet(i, value);
				double prob = subDataSet.rows() / (double)nrow;
				newEntropy += prob * calcShannonEnt(subDataSet);
			}
			double infoGain = baseEntropy - newEntropy;
			if (infoGain > bestInfoGain)
			{
				bestInfoGain = infoGain;
				bestFeature = i;
			}
		}
		return bestFeature;
	}

};

int main()
{
	DecisionTree trees;
	trees.createDataSet();
	trees.idx_label = 2;


	//cout << trees.splitDataSet(0,0) << endl;
	
	cout << trees.chooseBestFeatureToSplit() << endl;
	
	return 0;
}
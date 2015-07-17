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
#include <boost/any.hpp>

using namespace std;

class DecisionTree {
private:
	int nrow; // matrix row number
	int ncol; // matrix column number
	int idx_label; // index of the column contains tables

public:
	Eigen::MatrixXf dataMat; // the matrix of dataset
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
		idx_label = 2;
	}

	double calcShannonEnt(Eigen::MatrixXf &mat, int idx)
	{
		int numEntries = mat.rows();
		std::map<int,int> labelCounts;
		for (int i = 0; i < numEntries; ++i)
		{
			Eigen::VectorXf featVec = mat.row(i);
			int currentLabel = (int)featVec(idx); // the label idx
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

	Eigen::MatrixXf splitDataSet(Eigen::MatrixXf &dt, int axis, double value)
	{
		Eigen::MatrixXf retDataSet;
		std::vector<vector<double>> temp;

		for (int i = 0; i < nrow; ++i)
		{
			Eigen::VectorXf v1 = dt.row(i);
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

	int chooseBestFeatureToSplit(Eigen::MatrixXf &dt)
	{
		int numFeatures = ncol - 1;
		double baseEntropy = calcShannonEnt(dt, idx_label);
		double bestInfoGain = 0.0;
		int bestFeature = -1;
		for (int i = 0; i < numFeatures; ++i)
		{
			Eigen::VectorXf featList = dt.col(i);
			std::set<double> uniqueVals;
			for (int j = 0; j < nrow; ++j)
			{
				uniqueVals.insert(featList(j));
			}
			double newEntropy = 0.0;

			for (double value : uniqueVals)
			{
				Eigen::MatrixXf subDataSet = splitDataSet(dt, i, value);
				double prob = subDataSet.rows() / (double)nrow;
				newEntropy += prob * calcShannonEnt(subDataSet, idx_label-1);
			}
			double infoGain = baseEntropy - newEntropy;
			if (infoGain > bestInfoGain)
			{
				bestInfoGain = infoGain;
				bestFeature = i;
			}
		}
		return bestFeature; // this is the index of best feature
	}

	double majorityCnt(Eigen::VectorXf &classList)
	{
		std::map<double,int> classCounts;
		for (int i = 0; i < classList.size(); ++i)
		{
			double element = classList(i);
			if (classCounts.find(element) == classCounts.end())
				classCounts[element] = 1;
			else
				classCounts[element] ++;
		}
		int max_cnt = 0;
		double max_key = 0;
		for (std::map<double, int>::iterator it = classCounts.begin();
				it != classCounts.end(); ++it)
		{
			int val = it->second;
			double key = it->first;
			if (val > max_cnt)
			{
				max_cnt = val;
				max_key = key;
			}
		}
		return max_key; // this is the class have highest freq
	}

	boost::any createTree(Eigen::MatrixXf &dt)
	{
		int n = dt.cols(); // no. of columns
		Eigen::VectorXf classList = dt.col(n-1); // last column
		for (int i = 0; i < (classList.size() - 1); ++i)
		{
			if (classList(i) != classList(i+1))
				return classList(0); // double
		}
		if (dt.rows() == 1)
			return majorityCnt(classList); // int
		int bestFeat = chooseBestFeatureToSplit(dt);
		std::map<int, boost::any> myTree;
		Eigen::VectorXf featValues = dt.col(bestFeat);
		std::set<double> uniqueVals;
		for (int j = 0; j < featValues.size(); ++j)
		{
			uniqueVals.insert(featValues(j));
		}
		for (double value : uniqueVals)
		{
			std::map<int, boost::any> temp;
			Eigen::MatrixXf tmp;
			tmp = splitDataSet(dt, bestFeat, value);
			try
			{
				std::map<int, boost::any> v = boost::any_cast<std::map<int, boost::any>>(createTree(tmp));
				temp[value] = v;
				myTree[bestFeat] = temp;
			}
			catch(boost::bad_any_cast& e)
			{
				try{ int v = boost::any_cast<int>(createTree(tmp)); return v; }
				catch(boost::bad_any_cast& e)
				{ 
					double v = boost::any_cast<double>(createTree(tmp)); 
					return v;
				}
			}
			
		}
		return myTree;
	}

};

int main()
{
	DecisionTree trees;
	trees.createDataSet();
	trees.createTree(trees.dataMat);
	return 0;
}
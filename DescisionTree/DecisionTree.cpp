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
	double shannonEnt;

public:
	// index of the column contains tables
	int idx_label;
	
	// create sample dataset
	void createDataSet()
	{
		// 1 for yes, 0 for no
		dataMat.resize(0,0);
		dataMat.resize(5,3);
		dataMat(0,0) = 1;dataMat(0,1) = 1;dataMat(0,2) = 1;
		dataMat(1,0) = 1;dataMat(1,1) = 1;dataMat(1,2) = 1;
		dataMat(2,0) = 1;dataMat(2,1) = 0;dataMat(2,2) = 0;
		dataMat(3,0) = 0;dataMat(3,1) = 1;dataMat(3,2) = 0;
		dataMat(4,0) = 0;dataMat(4,1) = 1;dataMat(4,2) = 0;
		nrow = dataMat.rows();
		ncol = dataMat.cols();
	}

	void calcShannonEnt()
	{
		int numEntries = nrow;
		std::map<int,int> labelCounts;
		for (int i = 0; i < numEntries; ++i)
		{
			Eigen::VectorXf feetVec = dataMat.row(i);
			int currentLabel = (int)feetVec(idx_label);
			if (labelCounts.find(currentLabel) == labelCounts.end())
				labelCounts[currentLabel] = 1;
			else
				labelCounts[currentLabel] ++;
		}
		shannonEnt = 0.0;
		for (std::map<int, int>::iterator it = labelCounts.begin();
			it != labelCounts.end(); ++it)
		{
			double prob = (it->second)/(double)numEntries;
			shannonEnt -= prob * log2(prob);
			cout << shannonEnt << endl;
		}
	}

	void showPrivate()
	{
		cout << dataMat << endl;
		cout << shannonEnt << endl;
	}
};

int main()
{
	DecisionTree ins;
	ins.createDataSet();
	ins.idx_label = 2;

	ins.calcShannonEnt();
	ins.showPrivate();
	
	return 0;
}
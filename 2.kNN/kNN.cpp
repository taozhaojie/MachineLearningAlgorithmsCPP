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

class kNN {

private:
	Eigen::MatrixXf dataMat; // the matrix of dataset
	int nrow; // matrix row number
	int ncol; // matrix column number

	std::set<int> types; // all types

	// convert Eigen::Matrix to std::map
	std::map<std::vector<double>, int> mat2map()
	{
		std::map<std::vector<double>, int> dataSet;
		int mat_size = dataMat.rows();
		for(int i = 0; i != mat_size; ++i)
		{
			Eigen::VectorXf v1 = dataMat.row(i);
			std::vector<double> v2(v1.data(), v1.data() + v1.size());

			int type = v2[idx_label];
			v2.erase(v2.begin() + idx_label);
			dataSet[v2] = type;
			v2.clear();
		}
		return dataSet;
	}

	// calculate euclidean distance of 2 vectors
	double euclidean_distance(std::vector<double>& v1, std::vector<double>& v2)
	{
		if (v1.size() != v2.size())
		{
			cout << "size not equl" << endl;
			return -1;
		}

		else
		{
			double sum = 0;
			for(std::size_t i = 0; i != v1.size(); i++)
			{
				sum += pow((v1[i] - v2[i]), 2);
			}
			return sqrt(sum);
		}
	}

public:
	// index of the column contains tables
	int idx_label;
	
	// create sample dataset
	void createDataSet()
	{
		dataMat.resize(0,0);
		dataMat.resize(4,3);
		dataMat(0,0) = 1.0;
		dataMat(0,1) = 1.1;
		dataMat(0,2) = 0;
		dataMat(1,0) = 1.0;
		dataMat(1,1) = 1.0;
		dataMat(1,2) = 0;
		dataMat(2,0) = 0;
		dataMat(2,1) = 0;
		dataMat(2,2) = 1;
		dataMat(3,0) = 0;
		dataMat(3,1) = 0.1;
		dataMat(3,2) = 1;
	}

	// read data from file
	void file2matrix(std::string filename)
	{
		ifstream in(filename);
		string tmp;
		std::vector<std::string> vec_str;
		std::vector<double> vec_f;
		std::vector<std::vector<double>> vec_all;
		while (!in.eof()) {
			getline(in, tmp, '\n');
			if (tmp == "") break;
			boost::split(vec_str,tmp,boost::is_any_of("\t"));
			for(std::vector<string>::iterator it = vec_str.begin();
				it != vec_str.end(); ++it)
			{
				vec_f.push_back(atof(it->c_str()));
			}
			vec_all.push_back(vec_f);
			tmp.clear();
			vec_f.clear();
			vec_str.clear();
		}
		nrow = vec_all.size(); // to know the matrix size
		ncol = vec_all[0].size();
		
		dataMat.resize(0,0);
		dataMat.resize(nrow,ncol);
		for(int i = 0; i < nrow; ++i)
		{
			for(int j = 0; j < ncol; ++j)
			{
				dataMat(i,j) = vec_all[i][j];
			}
		}
	}

	// classifier
	int classify0(std::vector<double>& inX, std::size_t k)
	{
		std::map<double, std::vector<int>> dis; // distance, types
		std::vector<int> neighbours; // type of k neighbours

		// convert matrix to map container
		std::map<std::vector<double>, int> dataSet = mat2map();

		// calculate distance, map the distance to the class
		// use vector for class, in case same distances
		for(std::map<std::vector<double>, int>::iterator it = dataSet.begin();
			it != dataSet.end(); ++it)
		{
			std::vector<double> inY = it->first;
			double distance = euclidean_distance(inX, inY);
			dis[distance].push_back(it->second);
			types.insert(it->second);
		}

		//get types of k neighbours, put into vector
		for(std::map<double, std::vector<int>>::iterator it = dis.begin();
			it != dis.end(), neighbours.size() < k; ++it)
		{
			std::vector<int> vec = it->second;
			for(std::vector<int>::iterator it = vec.begin();
				it != vec.end(); ++it)
			{
				neighbours.push_back(*it);
			}
		}

		// count the occurrence of each type in the k neighbours
		std::map<int, int> temp;
		for(int type : types)
		{
			int type_count = std::count(neighbours.begin(), neighbours.end(), type);
			temp[type_count] = type;
		}

		return temp.rbegin()->second;
	}

	void autoNorm()
	{
		for(int i = 0; i < ncol; ++i)
		{
			if (i != idx_label)
			{
				dataMat.col(i).normalize();
			}
		}
	}

	void datingClassTest()
	{
		double hoRatio = 0.10;
		file2matrix("datingTestSet2.txt"); // readin the matrix
		idx_label = 3;	
		autoNorm(); // normalize the matrix

		int testingSize = hoRatio * nrow;
		int trainingSize = nrow - testingSize;
		Eigen::MatrixXf trainingSet = dataMat.block(0,0,trainingSize,ncol);
		Eigen::MatrixXf testingSet = dataMat.block(trainingSize,0,testingSize,ncol);
		dataMat.resize(0,0);
		dataMat = trainingSet; // pass the training set to dataMat

		int err_count = 0;
		for(int i = 0; i < testingSize; ++i)
		{
			Eigen::VectorXf v1 = testingSet.row(i);
			std::vector<double> vec(v1.data(), v1.data() + v1.size());
			int label = vec[idx_label];
			vec.erase(vec.begin() + idx_label);

			int result = classify0(vec, 3);
			cout << "the classifier came back with: " << result << ", the real answer is: " << label << endl;
			if (result != label)
				err_count ++;
		}
		double err_rate = err_count / (double)testingSize;
		cout << "the total error rate is " << err_rate << endl;
	}

};

int main()
{
	kNN ins;
	ins.datingClassTest();

	return 0;
}
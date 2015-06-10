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

using namespace std;

class NaiveBayes {

private:
	
	Eigen::MatrixXf dataMat;
	int nrow; // matrix row number
	int ncol; // matrix column number
	std::vector<std::vector<std::string>> postingList;
	std::vector<int> classVec;
	std::vector<std::string> vocabList;
	Eigen::VectorXf p1Vect;
	Eigen::VectorXf p0Vect;
	double pAbusive;

public:
	void loadDataSet()
	{
		postingList = {{"my", "dog", "has", "flea", "problems", "help", "please"},
					{"maybe", "not", "take", "him", "to", "dog", "park", "stupid"},
					{"my", "dalmation", "is", "so", "cute", "I", "love", "him"},
					{"stop", "posting", "stupid", "worthless", "garbage"},
					{"mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"},
					{"quit", "buying", "worthless", "dog", "food", "stupid"}};
		classVec = {0,1,0,1,0,1};
	}

	void createVocabList()
	{
		std::set<std::string> vocabListSet;
		for (std::vector<std::string> document : postingList)
		{
			for (std::string word : document)
				vocabListSet.insert(word);
		}
		std::copy(vocabListSet.begin(), vocabListSet.end(), std::back_inserter(vocabList));
	}

	Eigen::VectorXf setOfWords2Vec(std::vector<std::string> & inputSet)
	{
		std::vector<float> returnVec(vocabList.size(), 0);
		for (std::string word : inputSet)
		{
			size_t idx = std::find(vocabList.begin(), vocabList.end(), word) - vocabList.begin();
			if (idx == vocabList.size())
				cout << "word: " << word << "not found" << endl;
			else
				returnVec.at(idx) = 1;
		}
		Eigen::Map<Eigen::VectorXf> v(returnVec.data(),returnVec.size());
		return v;
	}

	void trainNB0()
	{
		pAbusive = std::accumulate(classVec.begin(), classVec.end(), 0) / (double)nrow;
		std::vector<Eigen::VectorXf> vec;
		for (std::vector<std::string> document : postingList)
		{
			vec.push_back(setOfWords2Vec(document));
		}
		ncol = vec[0].size();
		nrow = vec.size();
		dataMat.resize(nrow, ncol);
		for (int i = 0; i < nrow; ++i)
		{
			dataMat.row(i) = vec[i];
		}
		std::vector<float> tmp(ncol, 1);
		Eigen::Map<Eigen::VectorXf> p0Num(tmp.data(),ncol);
		Eigen::Map<Eigen::VectorXf> p1Num(tmp.data(),ncol);
		double p0Denom = 2.0;
		double p1Denom = 2.0;
		for (int i = 0; i < nrow; ++i)
		{
			if (classVec[i] == 1)
			{
				p1Num += dataMat.row(i);
				p1Denom += dataMat.row(i).sum();
			}
			else
			{
				p0Num += dataMat.row(i);
				p0Denom += dataMat.row(i).sum();
			}
		}
		p1Vect = (p1Num / p1Denom).array().log();
		p0Vect = (p0Num / p0Denom).array().log();
	}

	int classifyNB()

};

int main()
{
	NaiveBayes bayes;
	bayes.loadDataSet();
	bayes.createVocabList();

	bayes.trainNB0();

	return 0;
}
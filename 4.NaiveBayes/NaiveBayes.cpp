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
#include <boost/tokenizer.hpp>
#include <sstream>
#include <ctime>

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

	std::string readFile(std::string & filename)
	{
		std::ifstream in(filename);
		std::string str((std::istreambuf_iterator<char>(in)),
			            std::istreambuf_iterator<char>());
		return str;
	}

	void printVec(std::vector<std::string> &v)
	{
		for (auto it = v.begin(); it != v.end(); ++it)
		{
			cout << *it << ", ";
		}
		cout << endl;
	}

	void vec2mat()
	{
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
	}

	void vec2mat2(std::vector<int> &trainingIndex)
	{
		std::vector<std::vector<std::string>> temp;
		for (int idx : trainingIndex)
		{
			temp.push_back(postingList.at(idx));
		}

		std::vector<Eigen::VectorXf> vec;
		for (std::vector<std::string> document : temp)
		{
			vec.push_back(bagOfWords2VecMN(document));
		}
		ncol = vec[0].size();
		nrow = vec.size();
		dataMat.resize(nrow, ncol);
		for (int i = 0; i < nrow; ++i)
		{
			dataMat.row(i) = vec[i];
		}
	}

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
		std::vector<float> tmp(ncol, 1);
		std::vector<float> tmp2(ncol, 1);
		Eigen::Map<Eigen::VectorXf> p0Num(tmp.data(),ncol);
		Eigen::Map<Eigen::VectorXf> p1Num(tmp2.data(),ncol);
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

	int classifyNB(Eigen::VectorXf & vec2Classify)
	{
		double p1 = (vec2Classify.array() * p1Vect.array()).sum() + log(pAbusive);
		double p0 = (vec2Classify.array() * p0Vect.array()).sum() + log(1 - pAbusive);
		if (p1 > p0)
			return 1;
		else
			return 0;
	}

	void testingNB()
	{
		loadDataSet();
		createVocabList();
		vec2mat();
		trainNB0();
		std::vector<std::string> testEntry = {"love", "my", "dalmation"};
		Eigen::VectorXf doc = setOfWords2Vec(testEntry);
		int classifyResult = classifyNB(doc);
		cout << boost::algorithm::join(testEntry, ", ") << " classified as: " << classifyResult << endl;
		testEntry.clear();
		testEntry = {"stupid", "garbage"};
		doc = setOfWords2Vec(testEntry);
		classifyResult = classifyNB(doc);
		cout << boost::algorithm::join(testEntry, ", ") << " classified as: " << classifyResult << endl;
	}

	Eigen::VectorXf bagOfWords2VecMN(std::vector<std::string> & inputSet)
	{
		std::vector<float> returnVec(vocabList.size(), 0);
		for (std::string word : inputSet)
		{
			size_t idx = std::find(vocabList.begin(), vocabList.end(), word) - vocabList.begin();
			if (idx == vocabList.size())
				cout << "word: " << word << "not found" << endl;
			else
				returnVec.at(idx) += 1;
		}
		Eigen::Map<Eigen::VectorXf> v(returnVec.data(),returnVec.size());
		return v;
	}

	std::vector<std::string> textParse(std::string & bigString)
	{
		std::vector<std::string> vec;
		boost::tokenizer<> tok(bigString);
		for(boost::tokenizer<>::iterator beg = tok.begin(); beg != tok.end(); ++ beg)
		{
		    vec.push_back(*beg);
		}
		return vec;
	}

	void spamTest()
	{
		// readin file
		for (int i = 1; i != 26; ++i)
		{
			std::ostringstream ss1;
			std::ostringstream ss0;
			ss1 << "email/spam/" << i << ".txt";
			ss0 << "email/ham/" << i << ".txt";
			std::string filename1 = ss1.str();
			std::string filename0 = ss0.str();

			std::string str = readFile(filename1);
			std::vector<std::string> wordList = textParse(str);
			postingList.push_back(wordList);
			classVec.push_back(1);
			str = readFile(filename0);
			wordList = textParse(str);
			postingList.push_back(wordList);
			classVec.push_back(0);
		}

		createVocabList();

		// create trainingset and testset
		std::vector<int> trainingSet;
		std::vector<int> testSet;
		for (int i = 0; i != 50; ++i)
			trainingSet.push_back(i);

		srand(time(0));
		for (int i = 0; i != 10; ++i)
		{
			unsigned int len = trainingSet.size();
			int randIndex = rand() % len;
			testSet.push_back(trainingSet.at(randIndex));
			trainingSet.erase(trainingSet.begin() + randIndex);
		}

		vec2mat2(trainingSet);
		trainNB0();

		// test
		int errorCount = 0;
		for (int docIndex : testSet)
		{
			Eigen::VectorXf wordVector = bagOfWords2VecMN(postingList.at(docIndex));
			cout << docIndex << ": " << classifyNB(wordVector) << " " << classVec.at(docIndex) << endl;

			if (classifyNB(wordVector) != classVec.at(docIndex))
			{
				errorCount ++;
				cout << "classification error" << endl;
			}
		}
		double err_rate = errorCount / 10.0;
		cout << "the error rate is: " << err_rate << endl;
	}
};

int main()
{
	NaiveBayes bayes;
	
	//bayes.testingNB();

	/*std::ifstream in("email/ham/1.txt");
	std::string str((std::istreambuf_iterator<char>(in)),
	                 std::istreambuf_iterator<char>());

	std::vector<std::string> v = bayes.textParse(str);

	for (auto it = v.begin(); it != v.end(); ++it)
	{
		cout << *it << endl;
	}*/

	bayes.spamTest();

	return 0;
}
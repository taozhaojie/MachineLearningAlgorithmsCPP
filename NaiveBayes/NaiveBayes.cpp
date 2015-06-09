#include <iostream>
#include <fstream>
#include <utility>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <cstddef>

using namespace std;

class NaiveBayes {

private:
	
	std::vector<std::vector<std::string>> postingList;
	std::vector<int> classVec;
	std::vector<std::string> vocabList;

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

	std::vector<int> setOfWords2Vec(std::vector<std::string> & inputSet)
	{
		std::vector<int> returnVec(vocabList.size(), 0);
		for (std::string word : inputSet)
		{
			size_t idx = std::find(vocabList.begin(), vocabList.end(), word) - vocabList.begin();
			if (idx == vocabList.size())
				cout << "word: " << word << "not found" << endl;
			else
				returnVec.at(idx) = 1;
		}
		return returnVec;
	}

};

int main()
{
	NaiveBayes bayes;
	bayes.loadDataSet();
	bayes.createVocabList();

	std::vector<std::string> testInput = {"my", "dog", "has", "flea", "problems", "help", "please"};
	std::vector<int> testVec = bayes.setOfWords2Vec(testInput);

	for (auto it = testVec.begin(); it != testVec.end(); ++it)
	{
		cout << *it;
	}

	return 0;
}
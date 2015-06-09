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
	std::set<std::string> vocabList;

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
		for (std::vector<std::string> document : postingList)
		{
			for (std::string word : document)
				vocabList.insert(word);
		}
	}

	std::vector<int> setOfWords2Vec(std::vector<std::string>> & inputSet)
	{
		std::vector<int> returnVec(inputSet.size(), 0);
		for (std::string word : inputSet)
		{
			int idx = std::find(vocabList.begin(), vocabList.end(), word) - vocabList.begin();
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
	

	return 0;
}
#include <iostream>
#include <utility>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include "math.h"

using namespace std;

class kNN {

private:
	std::map<std::vector<double>, int> dataSet;
	std::map<double, std::vector<int>> dis;
	std::vector<int> neighbours;
	std::set<int> types;

public:
	void createDataSet()
	{
		dataSet[{1.0,1.1}] = 0;
		dataSet[{1.0,1.0}] = 0;
		dataSet[{0,0}] = 1;
		dataSet[{0,0.1}] = 1;
	}

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
			for(unsigned int i = 0; i != v1.size(); i++)
			{
				sum += pow((v1[i] - v2[i]), 2);
			}
			return sqrt(sum);
		}
	}

	int classify0(std::vector<double>& inX, unsigned int k)
	{
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

		std::map<int, int> temp;
		for(int type : types)
		{
			int type_count = std::count(neighbours.begin(), neighbours.end(), type);
			temp[type_count] = type;
		}

		return temp.rbegin()->second;
	}

	void showPrivate()
	{
		for(std::vector<int>::iterator it = neighbours.begin();
			it != neighbours.end(); ++it)
		{
			cout << *it << endl;
		}

	}

};

int main()
{
	kNN ins;
	ins.createDataSet();

	std::vector<double> test_vec = {0, 0};
	int result = ins.classify0(test_vec, 3);
	
	cout << result << endl;

	return 0;
}
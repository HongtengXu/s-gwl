#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include "math.h"
#include <time.h>
#include <map>
#include <list>

using namespace std;

class Network
{
    typedef map<string, int , less<string> > MapString2Int;
	typedef map<int, list<int>, less<int> > MapInt2List;
	typedef map<int, string, less<int> > MapInt2Name;
    
	MapInt2Name mNames;
	MapString2Int mapName;
    
public:
	int **neighbor; //neighbor of each node of the network
	int size; //number of nodes
	int maxDeg; //maximum degree of the network
    int *deg; //degree of each node
    int numOfEdge; //number of edges
	char* name; //name of the network
    
    //constructor, this function takes the network file and construct the graph of it. It finds the number of edges and the maximu degree of the network and assign to each node an ID.
    //Input parameter nam is the name of the file of the network.
	Network(char *nam);
	
    //constructor.does nothing
    Network(void);
    
	//destructor
    ~Network(void);
    
	//makes random networks by removing some nodes randomly.
    //Input parameter remNum determines the number of nodes are to be removed from network.
    void randomNodeRemoval(int remNum);
    
    //makes random network by removing edge randomly.
    //Input parameter remNum determines the number of edges are to be removed randomly.
    void randomEdgeRemoval(int remNum);
    
    //makes random network by adding edge randomly.
    //Input parameter addNum determines the number of edges are to be added randomly.
    void randomEdgeAddition(int addNum);
    
	//finds the corresponding name of a detremined ID.
    //Input parameter id is the ID of the node we're looking for its name.
    //Output is the name of the corresponding node.
    string getName(int id)
	{
		return mNames[ id ];
	};
    
	//finds the corresponding id of a detremined name.
    //Input parameter name is the name of the node we're looking for its ID.
    //Output is the ID of the corresponding node.
	int getID(string name)
	{
		return mapName[ name ];
	};
private:
};
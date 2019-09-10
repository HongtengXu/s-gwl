#include "Network.h"
#include "math.h"

//constructor, this function takes the network file and construct the graph of it. It finds the number of edges and the maximu degree of the network and assign to each node an ID.
//Input parameter nam is the name of the file of the network.
Network::Network(char *nam)
{
    try
	{
		name = nam;
		string fileName = name;
        string line;
		string token;
        int id1, id2;
        
		MapInt2List mapNeighbor;
        
		ifstream networkFile(fileName.c_str());
        if(!networkFile) //there is not such file
        {
            cout << "Error: file doesn't exist"<<endl;
            exit (1);
        }
        
        numOfEdge = 0; //number of edges of the network
		maxDeg = 0;    //max degree of the network
        
		while (getline(networkFile, line))  //reading network file
		{
			istringstream tokenizer(line);
			string token;
			cout << line <<endl;
			// cout << tokenizer <<endl;
            
            //read the first node in a row
			getline(tokenizer, token, '\t');
            
            if(token.length()==0) //the input node is incorrect
            {
                cout << "Error: No node in first column" <<endl;
                exit (1);
            }
            
            //assign first node of the neywork an ID
            mapName.insert(MapString2Int::value_type(token,(int)mapName.size()));
            id1 = mapName[ token ];
            
            //creat the corresponding list for its neighbors
			mNames.insert(MapInt2Name::value_type(id1, token));
			list<int> neighbs1;
			mapNeighbor.insert(MapInt2List::value_type( id1, neighbs1));
            
            //read the second node in a row
			getline(tokenizer, token, '\t');
            
            if(token.length()==1)//the input node is incorrect
            {
                cout << "Error: No node in second column" <<endl;
                exit (1);
            }
            if(token.at(token.length()-1)==13)
			{
				token = token.substr(0,token.length()-1);
			}
            
            //assign second node of the neywork an ID
            mapName.insert(MapString2Int::value_type(token,(int)mapName.size()));
            id2 = mapName[ token ];
			
            //creat the corresponding list for its neighbors
            mNames.insert(MapInt2Name::value_type(id2, token));
			list<int> neighbs2;
			mapNeighbor.insert(MapInt2List::value_type( id2, neighbs2));
            
            //insert first node in neigbor list of the second nod and vise versa.
			mapNeighbor[ id1 ].push_front( id2 );
			mapNeighbor[ id2 ].push_front( id1 );
            
		}
        
		size = mapName.size(); //number of nodes of the network
        
		//finding the neighbors and edges of the network with respect to mapNeigbor
        neighbor = new int*[size];
		deg = new int[size];
        
		list<int> tempList;
		list<int>::iterator it;
        
		for(int i=0; i<size; i++)
		{
            
            tempList = mapNeighbor[ i ];
			tempList.sort();
			tempList.unique();
            
            
			deg[i] = tempList.size();
			neighbor[i] = new int[deg[i]];
            
			numOfEdge += deg[i];
            
			if(deg[i] > maxDeg)
				maxDeg = deg[i];
            int j;
			for(j=0,it=tempList.begin(); it!= tempList.end() ; it++, j++)
				neighbor[i][j] = *it;
		}
        
		numOfEdge = numOfEdge / 2;
        
	}
	catch (exception &e)
	{
        cerr << "Error in input file" << endl;
        exit(1);
    }
}


//makes random network by removing edge randomly.
//Input parameter remNum determines the number of edges are to be removed randomly.
void Network::randomEdgeRemoval(int remNum)
{
	int counter;  //counts the number of edges of the network
	int inCounter; //number of neighbors for a node after removing the edges
	int remCounter = 0; //number of removed nodes
    //temporary
	int temp;
    
	maxDeg = 0; //maximum degree of the network
    
    srand ( (unsigned int)time(NULL) );
    
	bool *removedEdge = new bool[numOfEdge];
    
	for(int i=0; i<numOfEdge; i++)
	{
		removedEdge[i] = false;
	}
    
    //number of removed edges are to be equal to remNum
	for(int i=0; i<remNum; i++)
	{
        //selecting an edge randomely to be removed
        temp = rand() % numOfEdge;
		while( removedEdge[ temp ] )
			temp = rand() % numOfEdge;
		removedEdge[temp] = true; //edge is removed
	}
    
    //for each node of the network
    for( counter=0; counter<size; counter++)
	{
		inCounter = 0;
        int * node;
        node = new int[size];
        //for each neighbor of the node
		for(int k=0; k<deg[counter]; k++)
		{
			if(  counter < neighbor[counter][k])
			{
				if (!removedEdge[ remCounter ])
				{
                    node[inCounter] = neighbor[counter][k];
                    inCounter++;
				}
				else
				{
					for(int i=0; i<deg[ neighbor[counter][k] ]; i++)
						if(neighbor[ neighbor[counter][k] ][i] == counter)
							neighbor[ neighbor[counter][k] ][i] = -1;
                    
				}
				remCounter++;
			}
			else if( counter > neighbor[counter][k] && neighbor[counter][k] >= 0 )
			{
				node[inCounter] = neighbor[counter][k];
				inCounter++;
			}
            
		}
        
		delete [] neighbor[counter];
		neighbor[counter] = new int[inCounter];
		deg[counter] = inCounter;
        
        //find the maximum degree after removing some edges
		if(deg[counter] > maxDeg)
			maxDeg = deg[counter];
        //deifining the neighbors after removing soem edges
		for(int i=0; i<inCounter; i++)
			neighbor[counter][i] = node[i];
        
		delete [] node;
	}
	//number of edges aftre removing some edges
    numOfEdge -= remNum;
    
    delete [] removedEdge;
}

//makes random networks by removing some nodes randomly.
//Input parameter remNum determines the number of nodes are to be removed from network.
void Network::randomNodeRemoval(int remNum)
{
	int counter=0;
	int inCounter; //number of neighbors for a node after removing the nodes
    
	int temp;
    
	maxDeg = 0;
    
	bool *removedNode = new bool[numOfEdge];
	int *renode = new int[size]; //remained nodes
    
    srand ( (unsigned int)time(NULL) );
    
	for(int i=0; i<numOfEdge; i++)
	{
		removedNode[i] = false;
	}
    
    //number of removed nodes are to be equal to remNum
	for(int i=0; i<remNum; i++)
	{
        //selecting a node randomly to be removed
        temp = rand() % size;
		while( removedNode[ temp ])
			temp = rand() % size;
		removedNode[temp] = true; //node is removed
	}
    
    
	int **neighbor1 = new int*[size - remNum];
	int *deg1 = new int[size - remNum];
    
	counter = 0;
    
    //putting the remain nodes in array renode
	for(int j=0; j<size; j++)
	{
		if( removedNode[j] )
			continue;
		renode[j] = counter;
		counter++;
	}
    
	counter = 0;
	numOfEdge = 0;
	for(int j=0; j<size; j++)
	{
		if( removedNode[j] )
			continue; //do nothing
        
		inCounter = 0;
		int * node; //array of remained neighbors
		node = new int[size];
        //for all neighbors of the node
		for(int k=0; k<deg[j]; k++)
		{
			if( !removedNode[neighbor[j][k]] ) //neighbor is not removed
			{
				node[inCounter] = renode[neighbor[j][k]];
				inCounter++;
			}
		}
        
		neighbor1[counter] = new int[inCounter];
		deg1[counter] = inCounter;
		numOfEdge += inCounter; //number of edges after removing some nodes
        
		if(deg1[counter] > maxDeg)
			maxDeg = deg1[counter]; //finding the new maxDeg
        
		for(int i=0; i<inCounter; i++)
			neighbor1[counter][i] = node[i];
        
		delete [] node;
		counter++;
	}
    
	size = size - remNum; //new size of the netqork after removing some nodes
    
    //removes the old neighbors
    for(int j=0; j<size; j++)
    {
        delete [] neighbor[j];
    }
    delete [] neighbor;
	delete [] deg;
	
    neighbor = neighbor1;
	deg = deg1;
    
	numOfEdge = numOfEdge / 2;
    
    //memory leak
    for(int j=0; j<(size - remNum); j++)
    {
        delete [] neighbor1[j];
    }
    delete [] neighbor1;
    delete [] deg1;
    delete [] removedNode;
}

//makes random network by adding edge randomly.
//Input parameter addNum determines the number of edges are to be added randomly.
void Network::randomEdgeAddition(int addNum)
{
    
	int counter;
	int inCounter;
    
	int temp1 = 0;
	int temp2 = 0;
    
	maxDeg = 0;
    
	srand ( (unsigned int)time(NULL) );
    
	int *addedEdge1 = new int[2 * addNum];
	int *addedEdge2 = new int[2 * addNum];
    
	bool reRand;
	int j ;
    
	for(int i=0; i<addNum; i++)
	{
		reRand = true;
		while(reRand)
		{
			//check to selected nodes for the edge are not the same
            //and the edge is not the same as one exicsted edges
            while(reRand)
			{
				temp1 = rand() % size;
				temp2 = rand() % size;
				reRand = false;
                
				if( temp1 == temp2 )
					reRand = true;
                
				for(int j=0; j<deg[ temp1 ]; j++)
					if(neighbor[ temp1 ][j] == temp2)
						reRand = true;
			}
            
			//sort the arrays not to generate repetive edges
            //check the edge isn't produced before.
            for( j=2*i-1; j>=0; j-- )
			{
				if( ( addedEdge1[j]>temp1 ) || ( addedEdge1[j]==temp1 && addedEdge2[j]>temp2 ) )
				{
					addedEdge1[j+1] = addedEdge1[j];
					addedEdge2[j+1] = addedEdge2[j];
				}
				else
				{
					if( addedEdge1[j]==temp1 && addedEdge2[j]==temp2 )
						reRand = true;
					break;
				}
			}
		}
        
		addedEdge1[j+1] = temp1;
		addedEdge2[j+1] = temp2;
        
		for( j=2*i; j>=0; j-- )
		{
			if( ( addedEdge1[j]>temp2 ) || ( addedEdge1[j]==temp2 && addedEdge2[j]>temp1 ) )
			{
				addedEdge1[j+1] = addedEdge1[j];
				addedEdge2[j+1] = addedEdge2[j];
			}
			else
				break;
		}
        
		addedEdge1[j+1] = temp2;
		addedEdge2[j+1] = temp1;
	}
    
	int **neighbor1 = new int*[size];
	int *deg1 = new int[size];
    
	counter = 0;
    
	numOfEdge = 0;
	int addCounter = 0;
    
	for(counter=0; counter<size; counter++)
	{
		inCounter = 0;
		int * node;
		node = new int[size];
		for(int k=0; k<deg[counter]; k++)
		{
			if( addedEdge1[addCounter] == counter && addedEdge2[addCounter] < neighbor[counter][k])
			{
				node[inCounter] = addedEdge2[addCounter];
				inCounter++;
				addCounter++;
			}
			node[inCounter] = neighbor[counter][k];
			inCounter++;
		}
		while(addedEdge1[addCounter] == counter)
		{
			node[inCounter] = addedEdge2[addCounter];
			inCounter++;
			addCounter++;
		}
		neighbor1[counter] = new int[inCounter];
		deg1[counter] = inCounter;
		numOfEdge += inCounter;
        
		if(deg1[counter] > maxDeg)
			maxDeg = deg1[counter];
        
		for(int i=0; i<inCounter; i++)
			neighbor1[counter][i] = node[i];
		delete [] node;
	}
    
    for(int j=0; j<size; j++)
    {
        delete [] neighbor[j];
    }
    delete [] neighbor;
    
	delete [] deg;
	neighbor = neighbor1;
	deg = deg1;
	numOfEdge = numOfEdge /2;
    
    //memory leak
    for(int j=0; j<size; j++)
    {
        delete [] neighbor1[j];
    }
    delete [] neighbor1;
    
    delete [] addedEdge1;
    delete [] addedEdge2;
    delete [] deg1;
    
}
//constructor
Network::Network(void)
{
}
//destructor
Network::~Network(void)
{
}

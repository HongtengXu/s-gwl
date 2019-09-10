#include "Alignment.h"
#include <fstream>
#include <cstdlib>

using namespace std;
//constructor
//finds the smaller network and the maximum degree of the input networks
//Inputs are two files of networks net1 and net2
Alignment::Alignment( Network net1, Network net2)
{
	//compare networks to find the biggest one
    if( net1.size > net2.size )
	{
		reverse = true;
		network1 = net2;
		network2 = net1;
	}
	else
	{
		reverse = false;
		network1 = net1;
		network2 = net2;
	}
    
	//maximum degree of the network
    if(network1.maxDeg > network2.maxDeg)
		maxDeg = network1.maxDeg;
	else
		maxDeg = network2.maxDeg;

	similarity = new float*[network1.size];

	for(int i=0; i<network1.size; i++)
	{
		similarity[i] = new float[network2.size];
	}
}

// This function calculates the similarity between node1 in the first network and node2 in the second network given the similarity of their neighbors
float Alignment::computeSimilarity(int node1, int node2)
{
	bool *al;
	float *sim;
	int last;

	float tempS = 0;

	int deg1 = network1.deg[node1];
	int deg2 = network2.deg[node2];
    
    // We find a good alignment of neighbors of node1 and node2 heuristicly and then the similarity of node1 and node2 correspondes to the similarity of the aligned neighbors
	if (network1.deg[node1] <= network2.deg[node2])
	{
		al = new bool[ network2.deg[node2] ];
		sim = new float[ network1.deg[node1] ];
		for(int i=0; i<network2.deg[node2]; i++)
		{
			al[i] = false;
		}
		for(int i=0; i<network1.deg[node1]; i++)
		{
			sim[i] = 0;
		}
		for(int i=0; i<network1.deg[node1]; i++)
		{
			for(int j=0; j<network2.deg[node2]; j++)
			{
				if( !al[j] && sim[i] < similarity[ network1.neighbor[node1][i] ][ network2.neighbor[node2][j] ] )
				{
					sim[i] = similarity[ network1.neighbor[node1][i] ][ network2.neighbor[node2][j] ];
					last = j;
				}
			}
			al[last] = true;
			tempS += sim[i];
		}
	}
	else
	{
		al = new bool[network1.deg[node1]];
		sim = new float[ network2.deg[node2] ];
		for(int i=0; i<network1.deg[node1]; i++)
		{
			al[i] = false;
		}
		for(int i=0; i<network2.deg[node2]; i++)
		{
			sim[i] = 0;
		}
		for(int i=0; i<network2.deg[node2]; i++)
		{
			for(int j=0; j<network1.deg[node1]; j++)
			{
				if( !al[j] && sim[i] < similarity[ network1.neighbor[node1][j] ][ network2.neighbor[node2][i] ])
				{
					sim[i] = similarity[ network1.neighbor[node1][j] ][ network2.neighbor[node2][i] ];
					last = j;
				}
			}
			al[last] = true;
			tempS += sim[i];
		}
	}

		if( deg2 >= deg1 && deg2 > 0 )
			tempS /= deg2;
		else if ( deg1 > deg2 && deg1 > 0)
			tempS /= deg1;
		else
			tempS = 1.0;


	delete [] al;
	delete [] sim;
	return tempS;
}


// This function calculates the similarity values of nodes by an iterative approach on it iterations using b and c as weighting parameters. Please see the paper.
void Alignment::setSimilarities(int iteration, double bb, double cc)
{
	float ** tempSim;

	tempSim = new float*[network1.size];

	for(int i=0; i<network1.size; i++)
	{
		tempSim[i] = new float[network2.size];
	}

	for(int i=0; i<network1.size; i++)
		for(int j=0; j<network2.size; j++)
		{
			similarity[i][j] = 1.0;
		}

    // If we want to initiate the similarities by biological similarity values such as blast
	if( bb > 0 )
	{
        
        cout << "Initializing the similarity values from a file" << endl;
        
		float *sim1 = new float[network1.size];
		float *sim2 = new float[network2.size];

		for(int i=0; i<network1.size; i++)
			sim1[ i ] = 0;
		for(int i=0; i<network2.size; i++)
			sim2[ i ] = 0;



		string line;
		string token;
		int id1, id2;



		string f1,f2,f3;

		f1 = network1.name;
		f1.append("-");
		f1.append(network1.name);
		f1.append(".val");
		ifstream file1(f1.c_str());

		while (getline(file1, line))
		{

			istringstream tokenizer(line);
			string token;

			getline(tokenizer, token, '\t');

			id1 = network1.getID( token );

			getline(tokenizer, token, '\t');

			if(token.at(token.length()-1)=='\n')
				token = token.substr(0,token.length()-1);

			id2 =  network1.getID( token );

			if ( id1 == id2 )
			{
				getline(tokenizer, token, '\t');

				if(token.at(token.length()-1)=='\n')
					token = token.substr(0,token.length()-1);

				sim1[ id1 ] = (float)atof(token.c_str());
			}

		}




		f2 = network2.name;
		f2.append("-");
		f2.append(network2.name);
		f2.append(".val");

		ifstream file2(f2.c_str());

		while (getline(file2, line))
		{

			istringstream tokenizer(line);
			string token;

			getline(tokenizer, token, '\t');

			id1 = network1.getID( token );

			getline(tokenizer, token, '\t');

			if(token.at(token.length()-1)=='\n')
				token = token.substr(0,token.length()-1);

			id2 =  network1.getID( token );

			if ( id1 == id2 )
			{
				getline(tokenizer, token, '\t');

				if(token.at(token.length()-1)=='\n')
					token = token.substr(0,token.length()-1);

				sim2[ id1 ] = (float)atof(token.c_str());
			}

		}

		if( reverse )
		{
			f3 = network2.name;
			f3.append("-");
			f3.append(network1.name);
			f3.append(".val");
		}
		else
		{
			f3 = network1.name;
			f3.append("-");
			f3.append(network2.name);
			f3.append(".val");
		}

		ifstream file3(f3.c_str());

		while (getline(file3, line))
		{

			istringstream tokenizer(line);
			string token;

			getline(tokenizer, token, '\t');

			id1 = network1.getID( token );

			getline(tokenizer, token, '\t');

			if(token.at(token.length()-1)=='\n')
				token = token.substr(0,token.length()-1);

			id2 =  network1.getID( token );

			getline(tokenizer, token, '\t');

			if(token.at(token.length()-1)=='\n')
				token = token.substr(0,token.length()-1);

			similarity[ id1 ][ id2 ] = (float)( ( 1 - bb )* 1.0 + bb * 2 * atof(token.c_str()) / ( sim1[ id1 ] + sim2[ id2 ] ) );
		}
	}

	int maxSim1;
	int maxSim2;
	int minSim1;
	int minSim2;
	double sumSim;

	ofstream simFile( "simLog.txt");

    // The iterative process. At each iteration the similarity will be calculated for all nodes and then we update the similarity values at the end of the iteration.
	for(int it=0; it<iteration; it++)
	{
        cout << "Iteration: " << it <<"/"<<iteration << endl;
		maxSim1 = 0;
		maxSim2 = 0;
		minSim1 = 0;
		minSim2 = 0;
		sumSim = 0;

		for(int i=0; i<network1.size; i++)
		{
			if( i % 1000 == 0)
			for(int j=0; j<network2.size; j++)
			{
				tempSim[i][j] = (float)( ( 1 - cc ) * similarity[i][j] + cc * computeSimilarity( i, j ) );
			}
		}
		for(int i=0; i<network1.size; i++)
		{
			for(int j=0; j<network2.size; j++)
			{
				similarity[i][j] = tempSim[i][j];
				sumSim += similarity[i][j] / network2.size;
				if( similarity[i][j] > similarity[ maxSim1 ][ maxSim2 ] )
				{
					maxSim1 = i;
					maxSim2 = j;
				}
				else if( similarity[i][j] < similarity[ minSim1 ][ minSim2 ] )
				{
					minSim1 = i;
					minSim2 = j;
				}
			}
		}
		sumSim /= network1.size;

		simFile << it << " --------------------------------------" << endl;
		simFile << "Average : " << ( sumSim / network1.size ) << endl;
		simFile << "Maximum : " << " deg[" << maxSim1 << "]=" << network1.deg[maxSim1] << ", deg[" << maxSim2 << "]=" << network2.deg[maxSim2] << ", similarity[" << maxSim1 << "][" << maxSim2 << "]=" << similarity[maxSim1][maxSim2] << endl;
		simFile << "Minimum : " << " deg[" << minSim1 << "]=" << network1.deg[minSim1] << ", deg[" << minSim2 << "]=" << network2.deg[minSim2] << ", similarity[" << minSim1 << "][" << minSim2 << "]=" << similarity[minSim1][minSim2] << endl;

	}
    
    for(int j=0; j<network1.size; j++)
    {
        delete [] tempSim[j];
    }
    delete [] tempSim;

}

//produce a mapping between nodes of two network with respect to input parameter a.
//Input parameter a acontrols the factor edgeWeight in assigning the scores to the nodes. a should be between 0 and 1.
void Alignment::align(double aa)
{

	float **score;

	score = new float*[network1.size];

	for(int i=0; i<network1.size; i++)
	{
		score[i]= new float[network2.size];
	}
	float ss;
	float epsil = 0;

	for(int i=0; i<network1.size; i++)
		for(int j=0; j<network2.size; j++)
		{
			score[i][j]= 0;
		}
	float *tempScore1;
	float *tempScore2;

    // Assigning initial scores corresponding to degrees
	tempScore1 = new float[network1.size];
	for(int i=0; i<network1.size; i++)
	{
		tempScore1[i] = 0;
		for(int j=0; j<network1.deg[i]; j++)
			tempScore1[i] += (float)( 1.0 / network1.deg[network1.neighbor[i][j]] );
	}
	tempScore2 = new float[network2.size];
	for(int i=0; i<network2.size; i++)
	{
		tempScore2[i] = 0;
		for(int j=0; j<network2.deg[i]; j++)
			tempScore2[i] += (float)( 1.0 / network2.deg[network2.neighbor[i][j]] );
	}
	ofstream detailFile( "alignmentDetails.txt");

    // Assigning similarity between nodes of two netwroks just based on their degrees
	for(int i=0; i<network1.size; i++)
		for(int j=0; j<network2.size; j++)
		{
            score[i][j] = (tempScore1[i]<tempScore2[j])?tempScore1[i]/maxDeg:tempScore2[j]/maxDeg;
            ss = score[i][j];
            score[i][j] = (float)( (1 - aa) * ss +  aa * similarity[i][j] );
            if( score[i][j] < 0 )
                detailFile << "--------------------------  " << i << " : " << network1.deg[i] << " ----> " << j << " : " << network2.deg[j] << "    Score : " << score[i][j] << endl;

		}


	int nodeSize = network1.size;
	alignment = new int[nodeSize];
	int *nodeScore = new int[nodeSize];
	bool *aligned1 = new bool[nodeSize];
	float *minus = new float[nodeSize];
	bool *aligned2 = new bool[network2.size];
	float *negScore1 = new float[nodeSize];
	float *negScore2 = new float[network2.size];
	int **posScore;

	posScore = new int*[network1.size];

	for(int i=0; i<network1.size; i++)
	{
		posScore[i]= new int[network2.size];
	}



	for(int i=0; i<nodeSize; i++)
	{
		nodeScore[i] = 0;
		aligned1[i] = false;
		negScore1[i] = 0;
	}

	for(int i=0; i<network2.size; i++)
	{
		aligned2[i] = false;
		negScore2[i] = 0;
	}

	for(int i=0; i<nodeSize; i++)
		for(int j=0;j<network2.size; j++)
		{
			posScore[i][j]=0;
			if(score[i][j]>score[i][nodeScore[i]])
				nodeScore[i] = j;
		}
	int maxScore;

	bool ast = true;

    int progress=0;
    // main alignment procedure. At each iteraiton, we find the best alignment and then update the score accordingly
	for(int i=0; i<nodeSize; i++)
	{
        if ( (progress+1) <= (10*(i+1)/nodeSize) + 0.0000001)
        {
            progress++;
            cout << 10*progress << "%"<< endl;
        }
        
		maxScore = -1;
		int cnt = 0;
        
        // finding the best alignment
		for(int j=0; j<nodeSize; j++)
		{
			if(!aligned1[j])
			{
				if(maxScore < 0 )
					maxScore = j;
				else if(score[j][nodeScore[j]] > score[maxScore][nodeScore[maxScore]])
				{
					maxScore = j;
				}
				cnt++;
			}
		}
		if( i+ cnt != 5626 && ast )
		{

				ast = false;
		}

        // updating the alignment vector
		alignment[maxScore] = nodeScore[maxScore];
		aligned1[maxScore] = true;
		aligned2[nodeScore[maxScore]] = true;
		aa -= epsil;

		detailFile << maxScore << " : " << network1.deg[maxScore] << " ----> " << nodeScore[maxScore] << " : " << network2.deg[nodeScore[maxScore]] << "    Score : " << score[maxScore][nodeScore[maxScore]] << "     Similarity" << aa * similarity[maxScore][nodeScore[maxScore]] << endl;

        // decreesing the score of neighbors because of dependency
		for(int j=0; j<network1.deg[maxScore]; j++)
			negScore1[network1.neighbor[maxScore][j]] += (float)( 1.0 / network1.deg[maxScore] );

		for(int k=0; k<network2.deg[nodeScore[maxScore]]; k++)
			negScore2[network2.neighbor[nodeScore[maxScore]][k]] += (float)( 1.0 / network2.deg[ nodeScore[maxScore] ] );

        // aligning the neighbors with degree one
		for(int j=0; j<network1.deg[maxScore]; j++)
			for(int k=0; k<network2.deg[nodeScore[maxScore]]; k++)
				if( !aligned1[network1.neighbor[maxScore][j]] && !aligned2[network2.neighbor[nodeScore[maxScore]][k]])
				{
					if(network1.deg[network1.neighbor[maxScore][j]]==1 && network2.deg[network2.neighbor[nodeScore[maxScore]][k]]==1)
					{
						alignment[network1.neighbor[maxScore][j]] = network2.neighbor[nodeScore[maxScore]][k];

						aligned1[network1.neighbor[maxScore][j]] = true;
						aligned2[network2.neighbor[nodeScore[maxScore]][k]] = true;

						detailFile << "$ " << network1.neighbor[maxScore][j] << " : " << network1.deg[network1.neighbor[maxScore][j]] << " ----> " << network2.neighbor[nodeScore[maxScore]][k] << " : " << network2.deg[network2.neighbor[nodeScore[maxScore]][k]] << "    Score : " << score[network1.neighbor[maxScore][j]][network2.neighbor[nodeScore[maxScore]][k]] << endl;

						aa -= epsil;
						i++;
					}
				}

        // encouraging the neighbors of aligned nodes to align together by increasing the alignment score among them
		for(int j=0; j<network1.deg[maxScore]; j++)
			for(int k=0; k<network2.deg[nodeScore[maxScore]]; k++)
					posScore[network1.neighbor[maxScore][j]][network2.neighbor[nodeScore[maxScore]][k]]++;


        // updating scores
		for(int j=0; j<network1.deg[maxScore]; j++)
			if( !aligned1[network1.neighbor[maxScore][j]] )
				for( int k=0; k<network2.size; k++ )
				{
                    score[ network1.neighbor[maxScore][j] ][k] = ( ( tempScore1[ network1.neighbor[maxScore][j] ] - negScore1[ network1.neighbor[maxScore][j] ] ) < ( tempScore2[k] - negScore2[k] ) )?( tempScore1[ network1.neighbor[maxScore][j] ] - negScore1[ network1.neighbor[maxScore][j] ] )/maxDeg : ( tempScore2[k] - negScore2[k] )/maxDeg;
					ss =  ((float)posScore[ network1.neighbor[maxScore][j] ][k])/maxDeg + score[ network1.neighbor[maxScore][j] ][k];
					score[ network1.neighbor[maxScore][j] ][k] =  (float)( (1 - aa) * ss + aa * similarity[ network1.neighbor[maxScore][j] ][k] );

				}
        // updating scores
		for(int k=0; k<network2.deg[nodeScore[maxScore]]; k++)
			if( !aligned2[network2.neighbor[nodeScore[maxScore]][k]] )
				for( int j=0; j<network1.size; j++ )
				{
                    score[j][ network2.neighbor[nodeScore[maxScore]][k] ] = ( ( tempScore1[j] - negScore1[j] ) < ( tempScore2[ network2.neighbor[nodeScore[maxScore]][k] ] - negScore2[ network2.neighbor[nodeScore[maxScore]][k] ] ) )?( tempScore1[j] - negScore1[j] )/maxDeg : ( tempScore2[ network2.neighbor[nodeScore[maxScore]][k] ] - negScore2[ network2.neighbor[nodeScore[maxScore]][k] ])/maxDeg;
					ss = ((float)posScore[j][ network2.neighbor[nodeScore[maxScore]][k] ])/maxDeg + score[j][ network2.neighbor[nodeScore[maxScore]][k] ];
					score[j][ network2.neighbor[nodeScore[maxScore]][k] ] = (float)( (1 - aa) * ss + aa * similarity[j][ network2.neighbor[nodeScore[maxScore]][k] ] );
				}

        // updating scores
		for(int k=0; k<network2.deg[nodeScore[maxScore]]; k++)
			if( !aligned2[network2.neighbor[nodeScore[maxScore]][k]] )
				for( int j=0; j<network1.size; j++ )
				{
					if( nodeScore[ j ] == network2.neighbor[nodeScore[maxScore]][k] )
					{
							for(int l=0;l<network2.size; l++)
								if(score[ j ][l]>score[j][nodeScore[j]])
									nodeScore[j] = l;
					}
				}
        
        // updating scores
		for(int j=0; j<network1.deg[maxScore]; j++)
			if( !aligned1[network1.neighbor[maxScore][j]] )
				for( int k=0; k<network2.size; k++ )
				{
					if(score[ network1.neighbor[maxScore][j] ][k]>score[ network1.neighbor[maxScore][j] ][nodeScore[ network1.neighbor[maxScore][j] ]])
									nodeScore[ network1.neighbor[maxScore][j] ] = k;
				}

        // We don't want to align a node that is already aligned again
		for(int j=0; j<nodeSize; j++)
		{
			if(aligned2[nodeScore[j]])
			{
				score[j][nodeScore[j]] = -1;
			}
		}
        
        // finding the best candidate alignments for each node again
		for(int j=0; j<nodeSize; j++)
		{
			if(aligned2[nodeScore[j]])
			{
				for(int k=0;k<network2.size; k++)
					if(score[j][k]>score[j][nodeScore[j]] && !aligned2[k])
						nodeScore[j] = k;
			}
		}

	}
    cout << endl;

    for(int j=0; j<network1.size; j++)
    {
        delete [] score[j];
        delete [] posScore[j];
    }
    delete [] score;
    delete [] posScore;
    
    delete [] tempScore1;
	delete [] tempScore2;
	delete [] aligned1;
	delete [] aligned2;
	delete [] negScore1;
	delete [] negScore2;
    delete [] minus;
    delete [] nodeScore;
    
    // evaluate the alignment
	evaluate();

}


//calculate the evaluation measurments EC (Edge Correctness), IC (Interaction Correctness), NC (Node Correctness), CCCV and CCCE (largest Common Connected subraph with recpect to Vertices and Edges)
void Alignment::evaluate(void)
{
	CCCV = getCCCV(); //calculate CCCV
	CCCE = getCCCE(); //calculate CCCE
	EC = getEC();     //calculate Edge Correctness
	NC = getNC();     //calculate Node Correctness
	IC = getIC();     //calculate Interaction Correctness
}

//calculate CCCV
//return the number of vertices of largest common connected subgraph of the alignment
int Alignment::getCCCV(void)
{
    int *subGraph;
    int compNum = 1; //number of connected components
	int *q = new int[network1.size]; //nodes that are already processed
	comp = new int[network1.size]; //dtermines the connected component each node belongs to.
	for(int i=0; i<network1.size; i++)
	{
		comp[i] = network1.size;
		q[i] = i;
	}
    
	int last = 0;
    
	//for each node of the network
    for(int i=0; i<network1.size; i++)
	{
		if(comp[i]==network1.size)
		{
			q[0] = i;
			comp[i] = compNum;
			compNum++;
			last = 1;
            //finds all connected nodes tho the node i that is not alredy in a connected component
			for(int k=0; k<last; k++)
				for(int j=0; j<network1.deg[q[k]]; j++)
                    //the node is not already processed
					if( comp[q[k]] < comp[network1.neighbor[q[k]][j]])
					{
						for( int l=0; l < network2.deg[alignment[q[k]]]; l++ )
							if(network2.neighbor[alignment[q[k]]][l] == alignment[network1.neighbor[q[k]][j]])
							{
								comp[network1.neighbor[q[k]][j]] = comp[q[k]];
								q[last] = network1.neighbor[q[k]][j];
								last++;
							}
					}
		}
	}
    
	subGraph = new int[compNum-1]; //array of connected components
	for(int i=0; i<compNum-1; i++)
		subGraph[i] = 0;
	for(int i=0; i<network1.size; i++)
		subGraph[comp[i]-1]++; //number of nodes in a definit connected component
    
	//find the component with maximum nodes
    maxComp = 0;
	for(int i=0; i<compNum-1; i++)
	{
		if(subGraph[maxComp] < subGraph[i])
			maxComp = i;
	}
    
    int temp = subGraph[maxComp];
    
    //memory leak
    delete [] subGraph;
    delete [] q;
    
	return temp;
}

//calculate the evaluation measurment CCCE
//return the number of edges of largest common connected subgraph of the alignment
int Alignment::getCCCE(void)
{
	int edgeComp = 0;
    //for each node of first network
	for(int i=0; i<network1.size; i++)
	{
        //for each neighbor of node i
		for(int j=0; j<network1.deg[i]; j++)
            //for each neighbor l of a node in second network that is aligned with node i
			for( int l=0; l < network2.deg[alignment[i]]; l++ )
				if(network2.neighbor[ alignment[i] ][l] == alignment[network1.neighbor[i][j]])
					if( comp[i]-1 == maxComp)
						edgeComp++;
	}
    
	return ( edgeComp / 2 );
}

//calculate the evaluation measurment EC
//returns the percent of edges that are mapped correctly in alignment
float Alignment::getEC(void)
{
	int totalScore=0;
    
	//for each node i in first network
    for(int i=0; i<network1.size; i++)
	{
        //for each neighbor j of node i
		for(int j=0; j<network1.deg[i]; j++)
			//for each neighbor l of a node in second network that is aligned with node i
            for( int l=0; l < network2.deg[alignment[i]]; l++ )
				if(network2.neighbor[ alignment[i] ][l] == alignment[ network1.neighbor[i][j] ])
					totalScore++;
	}
    
	//minimum number of edges of two networks
    int minEdge = ( network1.numOfEdge > network2.numOfEdge)? network2.numOfEdge : network1.numOfEdge;
    //calculate EC(edge correctness)
	return ( (float) totalScore ) / ( 2 * minEdge );
}

//calculate the evaluation measurment NC
//returns percent of nodes that are mapped correctly in alignment
float Alignment::getNC(void)
{
	int nc = 0;
    //for each node in fist network
	for( int i=0; i<network1.size; i++)
        //check wether or not a node has aligned correctly.
        if( network1.getID( network2.getName ( alignment[i] ) ) == i )
			nc++;
    
	return ( (float)nc / network1.size );
}

//calculate the evaluation measurment IC
//returns percent of interactions that are mapped correctly in alignment
float Alignment::getIC(void)
{
	int ic = 0; //interaction correctness
    //for each node i of first network
	for(int i=0; i<network1.size; i++)
	{
        //for each node j of neighbors of node i
		for(int j=0; j<network1.deg[i]; j++)
            //for each neighbor l of a node in second network that is aligned with node i
			for( int l=0; l < network2.deg[alignment[i]]; l++ )
                //compare the neighbores of two aligned nodes to check wether or not they are aligned.
				if(network2.neighbor[ alignment[i] ][l] == alignment[network1.neighbor[i][j]])
					if( network1.getID( network2.getName ( alignment[i] ) ) == i && network1.getID( network2.getName ( alignment[network1.neighbor[i][j]] ) )== network1.neighbor[i][j] )
						ic++;
	}
    
	int minEdge = ( network1.numOfEdge > network2.numOfEdge)? network2.numOfEdge : network1.numOfEdge;
    
	return ( (float) ic ) / ( 2 * minEdge );
}

//print the evaluation measurments in a file with input parameter name
//Input parameter name determines the file that result are to be written in.
void Alignment::outputEvaluation(string name)
{
    
    // print in consule
	cout << "==============================================================================" << endl;
	cout << "--> Largest Connected Component:   Nodes = " << CCCV <<"   Edges = " << CCCE << endl;
    cout << "--> Edge Correctness           :   " << EC << endl;
    cout << "==============================================================================" << endl;
    
    
	string outFile = name;
    //add a definit suffix to the file
	outFile.append(".eval");
	ofstream outputFile( outFile.c_str());
    
    //print in file
	outputFile << "==============================================================================" << endl;
	outputFile << "--> Largest Connected Component  :   Nodes= " << CCCV <<"    Edges= " << CCCE << endl;
	outputFile << "--> Edge Correctness             :   " << EC << endl;
    outputFile << "--> Node Correctness             :   " << NC << endl;
    outputFile << "--> Interaction Correctness      :   " << IC << endl;
}

//print the alignment(mapping) in a file with input parameter name
//Input parameter name determines the file that mapping is to be written in.
void Alignment::outputAlignment(string name)

{
    
	string alignFile = name;
    
	//insert a definite suffix for the alignment file
    alignFile.append(".alignment");
    
    
	ofstream alignmentFile( alignFile.c_str());
    
	if(reverse) //the second input network is smaller
		for(int i=0; i<network1.size; i++)
			alignmentFile << network1.getName( alignment[ i ] ) << " -> " << network2.getName( i )<< endl;
	else //the first input network is smaller
		for(int i=0; i<network1.size; i++)
			alignmentFile << network1.getName( i ) << " -> " << network2.getName( alignment[ i ] )<< endl;
}
//instructor
Alignment::Alignment(void)
{
}
//destructor
Alignment::~Alignment(void)
{
}

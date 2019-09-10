#pragma once

#include "Network.h"
#include <iostream>
#include <cstdlib>
#include "math.h"
#include <time.h>

using namespace std;
class Alignment
{
public:
	float **similarity; // The similarity values of nodes

	//measures for evaluating the alignment
    float EC, IC, NC;
	int CCCV,CCCE;
    
	int *alignment; //array of alignment of two networks
	int *comp; //array of connected components in the resulted alignment
	int maxComp; //maximum connected component of the resulted alignment
	int maxDeg; //max degree of two networks
	bool reverse; //determine which networks is bigger
	Network network1;
	Network network2;
    
    //constructor
    //finds the smaller network and the maximum degree of the input networks
    //Inputs are two files of networks net1 and net2
	Alignment( Network net1, Network net2 );
	
    //constructor.does nothing !
    Alignment(void);
    
    //destructor
	~Alignment(void);
    
    //produce a mapping between nodes of two network with respect to input parameter a.
    //Input parameter a acontrols the factor edgeWeight in assigning the scores to the nodes. a should be between 0 and 1.
    void align(double a);
    
    // calculates the similarity values of nodes by an iterative approach on it iterations using b and c as weighting parameters. Please see the paper.
	void setSimilarities(int it, double b, double c);
    
	//calculate the evaluation measurments EC (Edge Correctness), IC (Interaction Correctness), NC (Node Correctness), CCCV and CCCE (largest Common Connected subraph with recpect to Vertices and Edges)
	void evaluate(void);
    
	//calculate CCCV
    //return the number of vertices of largest common connected subgraph of the alignment
    int getCCCV(void);
    
    //calculate the evaluation measurment CCCE
    //return the number of edges of largest common connected subgraph of the alignment
    int getCCCE(void);
    
    //calculate the evaluation measurment EC
    //returns the percent of edges that are mapped correctly in alignment
	float getEC(void);
    
    //calculate the evaluation measurment NC
    //returns percent of nodes that are mapped correctly in alignment
    float getNC(void);
    
    //calculate the evaluation measurment IC
    //returns percent of interactions that are mapped correctly in alignment
	float getIC(void);
    
    //print the evaluation measurments in a file with input parameter name
	//Input parameter name determines the file that result are to be written in.
    void outputEvaluation(string name);
    
    //print the alignment(mapping) in a file with input parameter name
	//Input parameter name determines the file that mapping is to be written in.
	void outputAlignment(string name);
private:
	float computeSimilarity(int node1, int node2);
};

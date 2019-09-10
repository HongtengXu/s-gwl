
#include "Alignment.h"
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

using namespace std;

int main(int argc, char* argv[])
{
	double aa = 0.0001, bb = 0, cc = 1; // weighting parameters in the paper
    
	int it = 2;     // Number of iterations when calculating the similarity
	int rr = 0;     // The type of randomness
	int pp = 0;     // The percentage of randomness
	int nn = 1;     // The number of random networks
	
    char* name1;    // The name of the first network
    char* name2;    // The name of second network

	try
	{
		
        if(argc < 3) {
            cout << "There should be two files as input!" <<endl;
            return -1;
        }
        else //input arguments
        {
            int i = 1; //counter for input parameters
            name1 = argv[ i++ ]; //first network
            name2 = argv[ i++ ]; //second network
            
            while (i<argc) //check all the input parameters
            {
                if ( ( strlen(argv[i]) == 2 ) && ( argv[i][0]=='-' ) && ( argv[i][1]=='a' || argv[i][1]=='b' || argv[i][1]=='c' || argv[i][1]=='i' ||argv[i][1]=='r' ||argv[i][1]=='n') && ( i + 1 < argc) )
                {
                    i++;
                    if( argv[i-1][1]=='a' )
                    {
                        aa = atof(argv[i]);
                        if( aa <0 || aa > 1)
                        {
                            cout << "Error : value of a must be between 0 and 1 :" << endl;
                            return -1;
                        }
                    }
                    else if( argv[i-1][1]=='b' )
                    {
                        bb = atof(argv[i]);
                        if( bb <0 || bb > 1)
                        {
                            cout << "Error : value of b must be between 0 and 1 :" << endl;
                            return -1;
                        }
                    }
                    else if( argv[i-1][1]=='c' )
                    {
                        cc = atof(argv[i]);
                        if( cc <0 || cc > 1)
                        {
                            cout << "Error : value of c must be between 0 and 1 :" << endl;
                            return -1;
                        }
                    }
                    else if( argv[i-1][1]=='n' )
                    {
                        nn = atoi(argv[i]);
                    }
                    else if( argv[i-1][1]=='r' )
                    {
                        rr = atoi(argv[i]);
                        i++;
                        pp = atoi(argv[i]);
                    }
                    else
                    {
                        it = atoi(argv[i]);
                        if( it < 0 )
                        {
                            cout << "Error : value of i must be positive" << endl;
                            return -1;
                        }
                    }
                    i++;
                }
                else
                {
                    cout << "Error in argument : " << argv[i] << endl;
                    return -1;
                }
            } //end while
            cout << "================================" << endl;
            cout << "Parameters:" << endl;
            cout << "a  = " << aa << endl;
            cout << "b  = " << bb << endl;
            cout << "c  = " << cc << endl;
            cout << "it = " << it << endl;
            cout << "--------------------------------" <<endl;
            
            
            cout << "Constructing the 1st network..." << endl;
            cout << name1 << ":"<< endl;
            //construct the first network
            Network network1(name1);
            cout << "Nodes= " << network1.size << endl;
            cout << "Edges= " << network1.numOfEdge << endl;
            cout << "--------------------------------" <<endl;
            cout << "Constructing the 2nd network..." << endl;
            cout << name2 << ":"<< endl;
            //construct the first network
            Network network2(name2);
            cout << "Nodes= " << network2.size << endl;
            cout << "Edges= " << network2.numOfEdge << endl;
            cout << "================================" << endl;
            
            bool reverse = false; //it means size of first input network is bigger than second one
            
            if(network1.size > network2.size)
                reverse = true;
            
            //Initializes the alignment class
            Alignment alignment( network1, network2 );
            
            //Calculates the similarity values
            cout << "Computing the similarity values..." << endl;
            alignment.setSimilarities(it, bb, cc);
            
            //Aligns two networks
            cout << "--------------------------------" <<endl;
            cout << "Alignment..." << endl;
            alignment.align(aa);
            
            //calculation of evaluating measures
            int CCCV = 0, CCCE = 0; //LCCS based on vertex and edge
            float EC = 0, NC = 0, IC = 0;
            CCCV += alignment.CCCV;
            CCCE += alignment.CCCE;
            EC += alignment.EC;
            NC += alignment.NC;
            IC += alignment.IC;
            
            //making the name for output file
            stringstream strm;
            strm << "(" << name1 << "-" << name2;
            if( rr > 0 )
                strm << rr << "r" << pp;
            strm << ")" << "-a" << aa << "-b" << bb << "-c" << cc << "-i" << it;
            
            //no random alignment
            if(rr == 0)
            {
                alignment.outputEvaluation(strm.str());
                alignment.outputAlignment(strm.str());
            }
            //random alignment
            else
            {
                int edgeNum; //percent of randome edges should be removed
                
                for( int i = 1; i < nn; i++ )
                {
                    
                    cout << "-------------------------------> Random Netowrk:  " << i << endl; //print the iteration number
                    Network network2(name2);
                    edgeNum = pp * network2.numOfEdge / 100;
                    
                    if( rr == 1) //construct the random graph with removing edges randomly
                    {
                        network2.randomEdgeRemoval( edgeNum );
                    }
                    else if(rr == 2) //construct the random graph with adding edges randomly
                    {
                        network2.randomEdgeAddition( edgeNum );
                    }
                    else if(rr == 3) //construct the random graph with removing and adding edges randomly
                    {
                        network2.randomEdgeRemoval( edgeNum );
                        network2.randomEdgeAddition( edgeNum );
                    }
                    
                    //find the alignment
                    Alignment alignment( network1, network2 );
                    alignment.setSimilarities(it, bb, cc);
                    alignment.align(aa);
                    
                    //calculate the eavluation measurements
                    CCCV += alignment.CCCV;
                    CCCE += alignment.CCCE;
                    EC += alignment.EC;
                    NC += alignment.NC;
                    IC += alignment.IC;
                    
                    //print the results in files
                    alignment.outputEvaluation(strm.str());
                    alignment.outputAlignment(strm.str());
                }
                
                //print the output of total random alignments
                if( nn > 1)
                {
                    // print in consule
                    cout << "================================AVERAGE VALUES================================" << endl;
                    cout << "--> Largest Connected Component:   Nodes= " << CCCV <<"    Edges= " << CCCE << endl;
                    cout << "--> Edge Correctness           :   " << EC << endl;
                    cout << "--> Node Correctness           :   " << NC << endl;
                    cout << "--> Interaction Correctness    :   " << IC << endl;
                    cout << "==============================================================================" << endl;
                    
                    
                    string outFile = strm.str();
                    outFile.append(".eval");
                    ofstream outputFile( outFile.c_str());
                    
                    //print in file
                    outputFile << "==============================================================================" << endl;
                    outputFile << "--> Largest Connected Component  :   Nodes= " << CCCV <<"    Edges= " << CCCE << endl;
                    outputFile << "--> Edge Correctness             :   " << EC << endl;
                    outputFile << "--> Node Correctness             :   " << NC << endl;
                    outputFile << "--> Interaction Correctness      :   " << IC << endl;
                    outputFile << "==============================================================================" << endl;
                }
            }
        }
    }
    catch(exception &e)
    {
        cout << "Error in arguments or input files!" << endl;
        e.what();
        return -1;
    }
        
    return 0;
}




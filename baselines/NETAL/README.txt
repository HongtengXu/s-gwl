=============================NETAL===============================

This software takes two networks as input and returns a global alignment of them.

If you want to compile the code, simply run the following command:

g++ NETAL.cpp Network.cpp Alignment.cpp -o NETAL

If you want to run the executable file, you just need to run the following command:

./NETAL net1.tab net2.tab [-a aa] [-b bb] [-c cc] [-i it]

where the arguments are:

------------------------------

net1.tab: The name of the first network file
net2.tab: The name of the second network file

------------------------------

-a aa:  an option for defining value of parameter a (see the paper for more information)
aa is a float number between 0 and 1.It is a value for parameter a that controls the weight of similarity and interaction scores. If aa=1 then the program considers only similarity scores and if aa=0 then it considers only interaction scores. The default value is aa=0.0001.

------------------------------

-b bb: an option for defining value of parameter b (see the paper for more information)
bb is a float number between 0 and 1.It is a value for parameter b that controls the weight of biological and topological similarities. If bb=1 then the program considers only biological similarity and if bb=0 then it considers only topological similarity. If you don't want to use any biological data, you should set bb=0 and you don't need to provide any file for biological similarities. Default value is bb=0.

------------------------------

-c cc: an option for defining value of parameter c (see the paper for more information)
cc is a float number between 0 and 1. The more the cc, the more the contribution of neighbors of two points in calculating the similarity between them. If cc=0 then the program considers only similarity of two nodes and if cc=1 then it considers only similarities of their neighbors. If you don't want to use any biological data, set cc=1. Default value is cc=1.

------------------------------

-i it: an option for defining value of parameter it (see the paper for more information)
it is a non-negative integer number .It defines the number of iterations for computing similarities. The default value is it=2;

------------------------------

A simple example:
./NETAL yeast.tab human.tab

To generate different random instances of a network, please see the code for the required parameters.

==================================================================
Input files :

You should provide two file with following format:

Each line corresponds to an interaction and contains the name of two proteins(separated by a tab) in that interaction.

Here is a sample for "net1.tab" :

aa	ab
bb	acd
a1	a3d
bd	re5

If you want to use biological similarity data for proteins, you should provide files "net1-net1.val","net2-net2.val" and "net1-net2.val" with the following format:

Each line corresponds to the biological similarity of two proteins and contains the name of the first and second proteins followed by the similarity value while all of them are separated by tabs.

Here is a sample for "net1-net2.val"

aa	cc	65.0
bb	dd	32.4
ab	cd	23.6


Here is a sample for "net1-net1.val"

aa	aa	128.0
bb	bb	94.7
ab	ab	30.5

==================================================================
Output files :

The program outputs two files "net1-net2.algn" and "net1-net2.eval". The first one contains global alignment of two networks, each matched pair in a line, and the second one contains some statistics about the alignment.

==================================================================

If you have any questions or if you have found any problem in the software, please email bneyshabur@ttic.edu. For the latest updates, please visit http://ttic.uchicago.edu/~bneyshabur or if you want to use the NETAL server, visit http://www.bioinf.cs.ipm.ir/software/netal

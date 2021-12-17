
Data File Format:
----------------------------------------------------
line - <details> - <example>
----------------------------------------------------
1 - int numInputs                                  e.g. 1
2 - int numOutputs                                 e.g. 1
3 - int numHiddenLayers                            e.g. 1
4 - int numberOfNodesInHiddenLayers []             e.g. 2  or   2 3
5 - weights <multiple formats: basic, random, inline>
  0 -> basic                                   e.g. 0
  1 -> random minVal maxVal                    e.g. 1 -2 2
  2 -> inline w[n][k][j]                       e.g  2 <w000> <w001> <w002> <w003> <w010> <w011> <w012> <w013> <w020> .. <w133>
                                               e.g. 2 0.25073206193168807 0.185397360929079 0.2806875990501302 0.6248532619795124 0.13402270892734014 0.10788549586548069

6 - lambda <multiple fromats: fixed, adaptive>
  0 -> fixed <learning rate of change>         e.g. 0 0.2
  1 -> adaptive <initial learning factor>       e.g. 1 0.3
7 - error threshold end training                   e.g. .01 or .001
8 - max number iterations                          e.g. 5
9 - "Network Run Type <evaluate_and_train, evaluation>"   "Num Cases"
  0  ->  evaluate and train the network using the training  e.g.  0 4
  1  ->  evaluation only                                    e.g.  1 2
10 - double inputActivations []                     e.g. 0.55 0.77
11 - expected output(s)                             e.g. 0 1
double inputActivations []                     e.g. 0.65 0.8
expected output                                e.g. 12
double inputActivations []                     e.g. 0.59 0.97
expected output                                e.g. 13
double inputActivations []                     e.g. 0.55 0.77
expected output                                e.g. 14
12 - Activation Function

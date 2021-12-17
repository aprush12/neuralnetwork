/**
 * @author Arusha Patil (p4)
 * @version 4-26-2020
 * The Perceptron class creates a neural network when given configuration information such as the number of input nodes,
 * number of nodes per hidden layer, and number of output nodes. The network receives multiple training data sets and is trained
 * using the back-propagation algorithm.
 */

import java.io.*;
import java.util.*;

public class Perceptron
{
   // Perceptron configuration, parameters specified by user
   int numInputNodes;         // number of input nodes
   int[] hiddenLayerNodes;    // number of nodes per hidden layer
   int numOutputNodes;        // number of output nodes
   int numHiddenLayers;       // number of hidden layers
   double lambda;             // initial learning factor used in calculating gradient descent
   double errorThreshold;     // error threshold for function
   int maxIterations;         // max number of iterations
   boolean adaptiveLambda;
   String activationFunction;
   String weightsSummary;
   int[] numNodesInLayer;

   // Perceptron variables that are calculated in constructor
   int maxActivationNodes;              // maximum of numInputNodes, numNodesPerHiddenLayer, numOutputNodes
   int activationMatrixLength;          // number of layers (columns) of activation Matrix
   double[] calculatedOutputs;          // output array
   double[][] psi;                        // array for psi values
   double[][] omegaJ;                        // array for psi values
   double[] omegaJJ;                        // array for psi values

   // Perceptron weights
   int weightsN, weightsJ, weightsK;    // Dimensions of Weights Array
   double[][][] weights;                // Weights Array

   // Perceptron activations matrix
   double[][] activations;              // Activations Array
   double[][] thetaJs;                  // thetaJ Array

   /**
    * Creates Perceptron object
    *
    * @param numInputs   total number of inputs
    * @param hLayerNodes array containing number of hidden nodes per hidden layer
    * @param numOutputs  total number of outputs
    */
   public Perceptron(int numInputs, int[] hLayerNodes, int numOutputs)
   {
      numInputNodes = numInputs;            // Number of total input nodes
      numHiddenLayers = hLayerNodes.length; // Number of hidden layers in network
      numOutputNodes = numOutputs;          // Number of total outputs, only 1 output for basic spreadsheet perceptron

      activationMatrixLength = numHiddenLayers + 2;  // The number of columns in the network
      hiddenLayerNodes = hLayerNodes;                // Array of number of nodes for each hidden layer

      // Calculates the tallest layer to find the most nodes in a single column,
      // using hiddenLayerNodes[0] since only considering a one hidden-layer network
      int maxHiddenLayerNodes = hiddenLayerNodes[0];
      for (int n: hiddenLayerNodes)
         maxHiddenLayerNodes = Math.max(maxHiddenLayerNodes, n);
      maxActivationNodes = Math.max(numOutputNodes, Math.max(numInputNodes, maxHiddenLayerNodes));

      // Creates the activation matrix for the network
      activations = new double[activationMatrixLength][maxActivationNodes];
      thetaJs = new double[activationMatrixLength][maxActivationNodes];   // We don't need the input layer col in this case

      // Creates an array to store the calculated outputs
      calculatedOutputs = new double[numOutputNodes];

      // Creates an array to store OmegaJ values for back propagation
      omegaJ = new double[activationMatrixLength][maxActivationNodes];
      omegaJJ = new double[maxActivationNodes];

      //Creates an array to store psi values
      psi = new double[activationMatrixLength][maxActivationNodes];

      weightsN = activationMatrixLength - 1;   // Initializes connectivity-layer, or nth dimension of weights
      // Not including output layer in the activations matrix, therefore - 1
      weightsK = maxActivationNodes;           // Initializes the input-activation layer, or kth dimension of weights
      weightsJ = maxActivationNodes;           // Initializes the right-most hidden-layer, or jth dimension of weights

      weights = new double[weightsN][weightsK][weightsJ];      // Creates the weight array using above dimensions
      adaptiveLambda = false; // set adaptive Lambda false by default

      // Create a generic array to hold number of nodes in each layer, with leftmost column as num Input activations
      // and the right most entry as num Output nodes
      numNodesInLayer = new int[numHiddenLayers+2];
      numNodesInLayer[0] = numInputs;
      for (int i = 0; i < numHiddenLayers; i++)
      {
         numNodesInLayer[i + 1] = hiddenLayerNodes[i];
      }
      numNodesInLayer[numHiddenLayers + 1] = numOutputs;
   } // constructor public Perceptron(int numInputs, int[]hLayerNodes, int numOutputs)

   /**
    * Creates random weights for network
    * @param min minimum value of range
    * @param max maximum value of range
    * @return random double value
    */
   public double randomWeights(double min, double max)
   {
      return ((Math.random() * (max - min)) + min);           // Calculates random double value
   }

   /**
    * Uses feed-forward algorithm to calculate hidden nodes and output values of network
    * @return calculated output values array
    */
   public double[] forwardPropagate(double[] expected)
   {
      for(int alpha = 1; alpha < activationMatrixLength; alpha++)
      {
         for (int beta = 0; beta < numNodesInLayer[alpha]; beta++)
         {
            thetaJs[alpha][beta] = 0.0;
            for (int gamma = 0; gamma < numNodesInLayer[alpha - 1]; gamma++)
            {
               thetaJs[alpha][beta] += activations[alpha - 1][gamma] * weights[alpha - 1][gamma][beta];
            }
            activations[alpha][beta] = applyActivationFunction(thetaJs[alpha][beta]);
         }
      }
      calculatedOutputs = activations[activationMatrixLength - 1];
      return calculatedOutputs;
   } // public double[] forwardPropagate()

   /**
    * Prints activation matrix into console
    */
   public void printActivations()
   {
      System.out.println("Activation Matrix:");
      for (double[] arr : activations)
      {
         for (double val : arr)
         {
            System.out.printf("%10.4f", val);
         }
         System.out.println();
      }
   } // public void printActivations()

   /**
    * Applies threshold function to hidden node value
    * @param theta theta value
    * @return updated hidden layer node value
    */
   public double applyActivationFunction(double theta)
   {
      // default
      // return 0.0;

      // sigmoid
      return (1.0 / (1.0 + Math.exp(-theta)));
   } // public double applyActivationFunction(double theta)


   /**
    * Applies derivative of activation function
    * @param theta value of node
    * @return double value of derivative
    */
   public double thresholdFunctionDerivative(double theta)
   {
      double f0 = applyActivationFunction(theta);

      // sigmoid derivative
      return f0 * (1 - f0);
   }


   /**
    * Sets activation function type
    * @param activationFunction determines value of activation function
    */
   public void setActivationFunction(String activationFunction)
   {
      this.activationFunction = activationFunction;
   }

   /**
    * Determines error for individual training sets
    * @param expected   correct calculated output values
    * @param calculated outputs returned by network
    * @return error value of training set
    */
   public double calculateError(double[] expected, double[] calculated)
   {
      double sum = 0.0;
      double diff = 0.0;
      for(int i = 0; i < numOutputNodes; i++)
      {
         diff = expected[i] - calculated[i];
         sum += diff * diff;
      }
      return 0.5 * sum;
   } // public double calculateError(double[] expected, double[] calculated)

   /**
    * Prints values of input nodes to console
    */
   public void printInputActivations()
   {
      double[] inp = getInputActivationsArray();
      System.out.print("Inputs:  ");
      for (double val : inp)
      {
         System.out.printf("%6.2f", val);
      }
      System.out.println();
   } // public void printInputActivations()

   /**
    * Returns the pointer to the allocated space for input activations within the activations matrix
    * @return array of numInputs length
    */
   public double[] getInputActivationsArray()
   {
      return activations[0];     // Input activations array is 0th column of activations matrix; returns
      // array to be populated
   }

   /**
    * Fills the weight values based on the user's decision to use the default settings or randomized weight values
    */
   public void fillBasicWeights()
   {
      weights[0][0][0] = -2.5;
      weights[0][0][1] = -2.0;
      weights[0][1][0] = -1.5;
      weights[0][1][1] = -1.0;
      weights[1][0][0] = -0.5;
      weights[1][1][0] = 0.25;
   }

   /**
    * Fills the weights randomly
    * @param min min value in range of random weights
    * @param max maximum value in range of random weights
    */
   public void fillRandomWeights(Double min, Double max)
   {
      for (int n = 0; n < weightsN; n++)                      // Iterates over nth-dimension of weights
      {
         for (int j = 0; j < weightsJ; j++)                   // Iterates over jth-dimension of weights
         {
            for (int k = 0; k < weightsK; k++)                // Iterates over kth-dimension of weights
            {
               weights[n][k][j] = randomWeights(min, max);    // Assigns random weight values
            }
         }
      } // for (int n = 0; n < weightsN; n++)
   } // public void fillRandomWeights(Double min, Double max)


   /**
    * Sets weight to specific value specified by user
    * @param n   nth layer of weight
    * @param k   kth left node of weight
    * @param j   jth right node of weight
    * @param val value to set weight to
    * @return value that weight is set to
    */
   public double setWeightAtIndex(int n, int k, int j, double val)
   {
      weights[n][k][j] = val;
      return val;
   }

   /**
    * Back-propagates network in order to train and develop weights
    * @param expectedOutputs array of expectedOutput values
    */
   public void calculateDeltaWeights(double[] expectedOutputs)
   {
      int lastHiddenLayerIndexInActivationMatrix = activationMatrixLength - 2;

      for (int i = 0; i < numOutputNodes; i++) {
         double derivative = thresholdFunctionDerivative(thetaJs[activationMatrixLength - 1][i]);
         psi[activationMatrixLength - 1][i] = calculateLowerOmega(expectedOutputs, i) * derivative;
      }

      for(int alpha = lastHiddenLayerIndexInActivationMatrix; alpha >= 0; alpha--)
      {
         int thetaJIndex = alpha - 1;
         for(int beta = 0; beta < numNodesInLayer[alpha]; beta++)
         {
            double hJ = activations[alpha][beta];

            double omegaJ = 0.0;
            int numNodesInLayerToTheRight = numNodesInLayer[alpha + 1];
            for (int gamma = 0; gamma < numNodesInLayerToTheRight; gamma++) {

               double psiI = psi[alpha + 1][gamma];

               // calculate omegaJ for current layer using psi values from layer to the right
               omegaJ += (psiI * weights[alpha][beta][gamma]);

               // update weights between current layer and layer to the right
               weights[alpha][beta][gamma] += (lambda * hJ * psiI);
            }

            // pre-calculate psi values for current layer - to be used for layer to the right
            psi[alpha][beta] = omegaJ * thresholdFunctionDerivative(thetaJs[alpha][beta]);
         }
      }
   } // public void calculateDeltaWeights(double expectedOutput)


   /**
    * Calculates lower omega value for back-propagation purposes
    * @param expected expected value array
    * @param i index of calculated and expected outputs to calculate omega from
    * @return lower omega value
    */
   public double calculateLowerOmega(double[] expected, int i)
   {
      return expected[i] - calculatedOutputs[i];
   }

   /**
    * Prints the weight values to console for reference
    */
   public void printWeights()
   {
      System.out.println("Weights: ");
      for (int n = 0; n < weightsN; n++)
      {
         for (int k = 0; k < weightsK; k++)
         {
            for (int j = 0; j < weightsJ; j++)
            {
               System.out.printf("%6.2f ", weights[n][k][j]);
            }
         }
      }
      System.out.println();
   } // public void printWeights()

   /**
    * Sets lambda to an inputted value
    * @param val value to set lambda to
    */
   public void setLambda(double val)
   {
      lambda = val;
   }

   /**
    * Gets lambda value
    * @return lambda value
    */
   public double getLambda()
   {
      return lambda;
   }

   /**
    * Sets the error threshold to value given by user
    * @param val value of error threshold
    */
   public void setErrorThreshold(double val)
   {
      errorThreshold = val;
   }

   /**
    * Gets the error threshold value from user
    * @return errorThresholdValue
    */
   public double getErrorThreshold()
   {
      return errorThreshold;
   }

   /**
    * Sets maximum number of iterations for looping over training set
    * @param num maximum number of iterations
    */
   public void setMaxIterations(int num)
   {
      maxIterations = num;
   }

   /**
    * Gets maximum number of iterations
    * @return maximum number of iterations
    */
   public int getMaxIterations()
   {
      return maxIterations;
   }

   /**
    * Sets the adaptive lambda value for future implementation
    */
   public void setAdaptiveLambda()
   {
      adaptiveLambda = true;
   }

   /**
    * Returns the number of hidden layers
    * @return number of hidden layers in network
    */
   public int getNumHiddenLayers()
   {
      return numHiddenLayers;
   }

   /**
    * Returns number of inputs
    * @return number of inputs values in network
    */
   public int getNumInputNodes()
   {
      return numInputNodes;
   }

   /**
    * Returns number of output in network
    * @return number of outputs
    */
   public int getNumOutputNodes()
   {
      return numOutputNodes;
   }

   /**
    * Checks whether adaptive learning is on
    * @return true if adaptive learning should run; otherwise, false
    */
   public boolean isAdaptiveLambdaSet()
   {
      return adaptiveLambda == true;
   }

   /**
    * Prints the parsed training cases for reference
    * @param iO input output training case
    * @param numInputs number of inputs in training case
    */
   public static void printTrainingCase(double[] iO, int numInputs) {
      System.out.print("Input(s): [ ");

      int outIndex = 0;
      boolean printOutputHeader = false;
      for (double v : iO) {
         if (outIndex == numInputs) {
            printOutputHeader = true;
         }
         outIndex++;

         if (printOutputHeader) {
            System.out.print("]   Output(s): [ ");
            printOutputHeader = false;
         }

         System.out.print(v + " ");
      }
      System.out.print("]\n");
   } //public static void printTrainingCase(double[] iO, int numInputs)

   /**
    * Parses training sets from user data
    * @param br Buffered Reader to parse data
    * @return double[][] training set from data
    */
   public double[][] parseTrainingSets(BufferedReader br)
   {
      int numTrainingSets = 0;
      StringTokenizer st;
      try
      {
         st = new StringTokenizer(br.readLine());
         numTrainingSets = Integer.parseInt(st.nextToken());
      } catch (IOException ex)
      {
         System.out.println(ex);
      }

      double[][] trainingCases = new double[numTrainingSets][getNumInputNodes() + getNumOutputNodes()];
      try
      {
         int setCount = 0;
         while (setCount < numTrainingSets)                // Iterates over training data sets
         {
            st = new StringTokenizer(br.readLine());       // line10 - array of inputs
            int count = 0;
            while (count < getNumInputNodes())             // Iterates over all inputs and stores them in trainingCases
            {
               trainingCases[setCount][count] = Double.parseDouble(st.nextToken());
               count++;

            }

            st = new StringTokenizer(br.readLine());       // line10 - array of inputs
            count = 0;
            while (count < getNumOutputNodes())
            {
               trainingCases[setCount][count + getNumInputNodes()] = Double.parseDouble(st.nextToken());
               count++;
            }
            setCount++;
         } // while (setCount < numTrainingSets)
      } catch (IOException ex)
      {
         System.out.println(ex);
      }
      return trainingCases;
   } // public double[][] parseTrainingSets(BufferedReader br)

   /**
    * Set weights summary from config data
    * @param msg String Message for weights
    */
   public void setWeightsSummary(String msg) {
      weightsSummary = msg;
   }

   /**
    * Parses weights summary info
    * @return String summary of weights
    */
   public String getWeightsSummary() {
      return weightsSummary;
   }

   /**
    * Parses weights from user data
    * @param st String Tokenizer API for parsing input document
    */
   public void parseWeights(StringTokenizer st)
   {
      int weightType = Integer.parseInt(st.nextToken());
      if (weightType == 0)       // Basic weights
      {
         fillBasicWeights();
         setWeightsSummary("Basic prefilled weights");
      }
      else if (weightType == 1) // Random weights
      {
         double min = Double.parseDouble(st.nextToken());
         double max = Double.parseDouble(st.nextToken());
         fillRandomWeights(min, max);
         setWeightsSummary("Random weights - range " + min + ":" + max);
      }
      else if (weightType == 2) // User-specified weights
      {
         setWeightsSummary("User Specified weights.");

         for (int n = 0; n < weightsN; n++)
         {
            for (int k = 0; k < weightsK; k++)
            {
               for (int j = 0; j < weightsJ; j++)
               {
                  setWeightAtIndex(n, k, j, Double.parseDouble(st.nextToken()));
               }
            }
         } // for (int n = 0; n < weightsN; n++)
         System.out.println("Read user defined weights:\n");
         printWeights();
      }
   } // public void parseWeights(StringTokenizer st, PerceptronMultiOut perceptron)

   /**
    * Takes in file containing user's inputs, creates network using user-specified parameters, prints weights, input
    * activations, activation matrix, output value, and activation type
    * @param args information from command line
    */
   public static void main(String[] args)
   {
      try
      {
         Scanner sc = new Scanner(System.in);
//         System.out.println("Enter a filename");
//         String configFile = sc.nextLine();
         String configFile = "2-4-3.OR-AND-XOR";
//         String configFile = "2-4-1.OR";
//         String configFile = "2-4-1.XOR";
//         String configFile = "2-4-4-3.AND-OR-XOR";
         BufferedReader br = new BufferedReader(new FileReader(configFile));

         StringTokenizer st = new StringTokenizer(br.readLine()); // line1 - number of inputs
         int numInputs = Integer.parseInt(st.nextToken());

         st = new StringTokenizer(br.readLine()); // line2 - number of outputs
         int numOutputs = Integer.parseInt(st.nextToken());

         st = new StringTokenizer(br.readLine()); // line3 - number of hidden layers
         int numHiddenLayers = Integer.parseInt(st.nextToken());

         st = new StringTokenizer(br.readLine()); // line4 - array of hidden layers nodes
         int[] hiddenLayerLengths = new int[numHiddenLayers];

         int count = 0;
         while (count < numHiddenLayers)          // Iterates over file values to populate hiddenLayers array
         {
            hiddenLayerLengths[count] = Integer.parseInt(st.nextToken());
            count++;
         }

         Perceptron perceptron = new Perceptron(numInputs, hiddenLayerLengths, numOutputs); // Initializes perceptron obj

         // Read in weights
         st = new StringTokenizer(br.readLine()); // in-line reading of weights, random, basic, user-input
         perceptron.parseWeights(st);

         // debugging purposes: perceptron.printWeights();
         //  System.out.println("–––––––––––––––––––––––––––––––––––––––––––––––––––––– ");

         st = new StringTokenizer(br.readLine()); // line 6 - initial learning factor
         int lambdaType = Integer.parseInt(st.nextToken());
         if (lambdaType == 0)
         {
            perceptron.setLambda(Double.parseDouble(st.nextToken()));
         }
         else if (lambdaType == 1)
         {
            perceptron.setLambda(Double.parseDouble(st.nextToken()));
            perceptron.setAdaptiveLambda();
         }

         st = new StringTokenizer(br.readLine()); // line7 - error threshold
         perceptron.setErrorThreshold(Double.parseDouble(st.nextToken()));

         st = new StringTokenizer(br.readLine()); // line8 - maximum iterations
         perceptron.setMaxIterations(Integer.parseInt(st.nextToken()));

         // Read in the training cases
         double[][] trainingCases = perceptron.parseTrainingSets(br);  // line9 - number of training data sets

         System.out.println("\nRead training set [...inputs, expected_output]:");
         for (double iO[] : trainingCases)
            perceptron.printTrainingCase(iO, numInputs);

         System.out.println("\n\nRunning training:");
         int numLoops = 0;
         boolean errorThresholdReached = false;

         st = new StringTokenizer(br.readLine());  // line18 - type of activation function
         perceptron.setActivationFunction(st.nextToken());

         double[] expectedOutputs = new double[numOutputs];
         double iOLen = numInputs + numOutputs;
         while (!errorThresholdReached && numLoops < perceptron.getMaxIterations())
         {
            System.out.printf("\n\n----------------------------\nLoop #%d\n", numLoops++);

            double maxErrorInSet = 0.0;
            for (double iO[] : trainingCases)
            {
               perceptron.printTrainingCase(iO, numInputs);

               int k = 0;
               for (int i = numInputs; i < iOLen; i++, k++)
               {
                  expectedOutputs[k] = iO[i];
               }

               double[] inputActivations = perceptron.getInputActivationsArray();
               for (int c = 0; c < numInputs; c++)
               {
                  inputActivations[c] = iO[c];
               }

               System.out.println("Forward propagating");
               double[] calculatedOutputs  = perceptron.forwardPropagate(expectedOutputs);  // Calculated output value of perceptron network
               double error = perceptron.calculateError(expectedOutputs, calculatedOutputs);

               System.out.println("Calculated Outputs:");
               for (int i = 0; i < numOutputs; i++) {
                  System.out.print(calculatedOutputs[i] + " ");
               }
               System.out.printf("\nError: %6.4f\n", error);

               if (maxErrorInSet < error)
               {
                  maxErrorInSet = error;
               }
               // Steepest descent
               perceptron.calculateDeltaWeights(expectedOutputs); // Back propagation algorithm

               System.out.println("––––––––––––––––––––––––––––––––––––––––");
            } // for (double iO[] : trainingCases) - end of looping through each training set

            System.out.printf("Max Error in set: %6.4f\n", maxErrorInSet);
            if (maxErrorInSet <= perceptron.getErrorThreshold())
            {
               errorThresholdReached = true;
            }
         } // while(!errorThresholdReached && numLoops < perceptron.getMaxIterations()) training process

         System.out.println("---------- done ------------");
         if (numLoops >= perceptron.getMaxIterations())
            System.out.printf("Max iterations %d exceeded. Stopping the loop.\n", numLoops);
         else
            System.out.printf("Error threshold reached. Stopping the training. Num loops: %d\n", numLoops);


         System.out.println("\n\n\n--------------------- SUMMARY ------------------");
         System.out.printf("PerceptronMultiOut Config File: %s\n\n", configFile);
         System.out.printf("Num Inputs: %d\n", numInputs);
         System.out.printf("Num Outputs: %d\n", numOutputs);
         System.out.printf("Num HiddenLayers: %d\n", numHiddenLayers);
         System.out.printf("Num Nodes in Hidden Layers: [ ");
         for (int nodes: hiddenLayerLengths )
            System.out.print(nodes + " ");
         System.out.println("]");
         System.out.printf("Weights: %s\n", perceptron.getWeightsSummary());

         if (perceptron.isAdaptiveLambdaSet())
            System.out.printf("Adaptive Lambda enabled. Setting Initial learning factor (lambda): %6.2f\n", perceptron.getLambda());
         else
            System.out.printf("Setting Rate of learning (lambda): %6.2f\n", perceptron.getLambda());

         System.out.printf("Max iterations: %d\n", perceptron.getMaxIterations());
         System.out.printf("Error threshold: %6.4f\n\n", perceptron.getErrorThreshold());

         System.out.println("Inputs and Expected Outputs:");
         int numTrainingCases = 0;
         for (double iO[] : trainingCases)
         {
            perceptron.printTrainingCase(iO, numInputs);
            numTrainingCases ++;
         }
         System.out.printf("\nNum Training Cases: %d\n", numTrainingCases);


         System.out.printf("\nResults:\n");
         if (numLoops >= perceptron.getMaxIterations())
         {
            System.out.println("Error: Trained failed!");
            System.out.println("Max iterations exceed. Looping stopped.");
         }
         else
         {
            System.out.println("––––––––––––––––––––––––––––––––––––––––");
            System.out.println("Success: Training completed successfully!");
            System.out.printf("Error threshold achieved: %6.4f\n", perceptron.getErrorThreshold());
            System.out.printf("Num Iterations taken: %d\n", numLoops);
            System.out.println("––––––––––––––––––––––––––––––––––––––––");
            //perceptron.printWeights();
         }
      } catch (IOException ex)
      {
         System.out.println(ex);
      }
   } // public static void main(String[] args)
} // public class Perceptron
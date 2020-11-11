
// Name: Marco Flores
// Student Number: 10259733
// Date: October 29, 2020
// Computer Science 475 – Assignment 2
// MNIST Handwritten Digit Recognizer Neural Network
// 
// Neural Network class handling calculations and network algorithms

package network;
import java.util.*;

public class NeuralNetwork {
    public int[] layerSizes;
    public List<List<Neuron> > neurons = new ArrayList<List<Neuron> >();
    public double learningRate = 3.0;
    public int correctNum;
    public NeuralNetwork network;

    // Constructor to establish base fully connected neural network with random weights and biases
    public NeuralNetwork(int[] layers) {
        layerSizes = layers;
        generateLayers();
        generateConnections();
    }

    // Constructor to establsh fully connected neural network with specified weights and biases
    public NeuralNetwork(int[] layers, List<List <List<Double> > > weights, List<List <Double> > biases, int correctNumber){
        layerSizes = layers;
        generateLayers();
        generateConnections();
        setWeights(weights);
        setBiases(biases);
        correctNum = correctNumber;
    }
    // generate neuron layers 
    public void generateLayers(){
        for (int i = 0; i < layerSizes.length; i++) {
            neurons.add(new ArrayList<Neuron>());
            generateNeurons(layerSizes[i], neurons.get(i));
        }
    }

    // generate neurons in each layer
    public void generateNeurons(int layerSize, List<Neuron> layerNeurons) {
        for (int i = 0; i < layerSize; i++) {
            layerNeurons.add(new Neuron());
        }
    }

    // establish connections between neurons in a fully connected network
    public void generateConnections(){
        for(int i = 0; i < layerSizes.length - 1; i++){
            for(int j = 0; j < layerSizes[i]; j++){
                neurons.get(i).get(j).generateWeights(layerSizes[i+1]);
            }
        }
    }

    // set weights from a specified list
    public void setWeights(List<List <List<Double> > > weights){
        for(int i = 0; i < weights.size(); i++){
            for(int j = 0; j < weights.get(i).size(); j++){
                neurons.get(i).get(j).setWeights(weights.get(i).get(j));
            }
        }
    }

    // set biases from a specified list
    public void setBiases(List<List<Double> > biases){
        for(int i = 0; i < biases.size(); i++){
            for(int j = 0; j < biases.get(i).size(); j++){
                neurons.get(i).get(j).bias = biases.get(i).get(j);
            }
        }
    }

    // feedforward through network for each training pair
    public void parse(List<List<Double> > trainingData, int epochs, int miniBatchSize){
        // ArrayList to hold correct output values for each pair
        List<Integer> correctVals = new ArrayList<Integer>(); 
        // ArrayList to hold number of occurances of each digit
        List<Integer> numberOfOccurances = new ArrayList<Integer>();
        // ArrayList to hold number of occurances for each digit in guesses for each digit
        List<Integer> guesses = new ArrayList<Integer>();
        // digit output for network
        int output = 0;

        // add 10 zeroes to the ArrayLists for digits
        for(int indx = 0; indx < 10; indx++){
            numberOfOccurances.add(0);
            guesses.add(0);
        }

        network = new NeuralNetwork(layerSizes); // represents a mini batch network

        // define lists to maintain history of gradients as well as weights and biases for network
        List< List< List< Double > > > weights = new ArrayList< List< List<Double > > >();
        List< List< Double > > biases = new ArrayList< List<Double > >();
        List<List< List< List< Double > > > > weightGradients = new ArrayList<List < List< List<Double > > > >();
        List<List< List< Double> > > biasGradients = new ArrayList<List < List<Double > > >();

        // for loop for epochs
        for(int i = 0; i < epochs; i++){
            // changing learning rate to compensate for overshooting values
            if(i >= 20)
                network.learningRate = 0.1;
            else if(i >= 10)
                network.learningRate = 1;

            network.correctNum = 0;
            
            //Collections.shuffle(trainingData);

            // data input loop
            for(int j = 0; j < trainingData.size(); j++){

                // populate tally of each number of the digits
                if(i == 0){
                    correctVals.add(trainingData.get(j).get(0).intValue());
                }

                // at the beginning of a mini batch, reset the gradients history
                if(((j % miniBatchSize)) == 0){
                    weightGradients = new ArrayList<List < List< List<Double > > > >();
                    biasGradients = new ArrayList<List < List<Double > > >();
                }

                // populate our mini batch
                weightGradients.add(new ArrayList< List< List<Double > > >());
                biasGradients.add(new ArrayList< List<Double > >());

                // populate network with input and feedforward
                network.addNeuronInputs(trainingData.get(j), network.neurons.get(0));
                network.parseLayers(layerSizes.length - 1);

                // determine output and if it is correct
                output = network.parseOutputLayer(correctVals.get(j));

                // store weight gradients
                for(int k = 0; k < network.neurons.size(); k++){
                    weightGradients.get(j%miniBatchSize).add(new ArrayList<List<Double> >());
                    for(int l = 0; l < network.neurons.get(k).size(); l++){
                        weightGradients.get(j%miniBatchSize).get(k).add(network.neurons.get(k).get(l).weightGradients);
                    }
                }
                
                // store bias gradients
                for(int k = 0; k < network.neurons.size(); k++){
                    biasGradients.get(j%miniBatchSize).add(new ArrayList<Double>());
                    for(int l = 0; l < network.neurons.get(k).size(); l++){
                        biasGradients.get(j%miniBatchSize).get(k).add(network.neurons.get(k).get(l).biasGradient);
                    }
                }

                // if start of a mini batch
                if(j % miniBatchSize == 0){
                    // if not the first iteration assign weights from prior network
                    if(!(i == 0 && j == 0)){
                        network = new NeuralNetwork(layerSizes, weights, biases, network.correctNum);
                    }
                    // update weights + set lists to proper weights
                    if(i > 0){
                        weights = new ArrayList<List<List<Double> > >();
                        biases = new ArrayList<List<Double> >();
                        network.updateWeights(miniBatchSize, weightGradients);
                        network.updateBias(miniBatchSize, biasGradients);
                        for(int l = 0; l < network.neurons.size(); l++){
                            weights.add(new ArrayList< List<Double > >());
                            biases.add(new ArrayList<Double>());
                            for(int m = 0; m < network.neurons.get(l).size(); m++){
                                weights.get(l).add(network.neurons.get(l).get(m).weights);
                                biases.get(l).add(network.neurons.get(l).get(m).bias);
                            }
                        }

                    }
            
                }

                // count number of guesses of each digit
                if(output == correctVals.get(j)){
                    guesses.set(output, guesses.get(output)+1);
                }

                //calculate gradients + store values for each input
                network.calculateBiasGradients(correctVals.get(j));
                network.calculateWeightGradients();
            }

            // populate number of occurances for each correct value
            if(i == 0){
                for(int k = 0; k < correctVals.size(); k++){
                    numberOfOccurances.set(correctVals.get(k), numberOfOccurances.get(correctVals.get(k))+1);
                }
            }

            // print output after each epoch
            System.out.println("Epoch " + i + ":");
            for(int printLoop = 0; printLoop < guesses.size(); printLoop++){
                System.out.print(printLoop + " = " + guesses.get(printLoop) + "/" + numberOfOccurances.get(printLoop) + " ");
                if(printLoop % 5 == 0 && printLoop > 0){
                    System.out.println();
                }
            }
    
            double acc = (double)network.correctNum/trainingData.size()*100;
            System.out.print(" Accuracy = " + network.correctNum + "/" + trainingData.size() + " = " + acc + "%\n");
            // reset guesses for each epoch
            for(int z = 0; z < guesses.size(); z++){
                guesses.set(z, 0);
            }
        }


    }


    // parse each layer after the first and calculate activation values
    public void parseLayers(int numLayers){
        if(numLayers > 0){
            int layerIndx = layerSizes.length - numLayers;
            for(int i = 0; i < neurons.get(layerIndx).size(); i++){
                neurons.get(layerIndx).get(i).activationVal = sigmoid(generateZValue(layerIndx-1, i));
            }
            parseLayers(numLayers - 1);
        }
    }

    // determine most likely output value
    public int parseOutputLayer(int correctVal){
        // softmax threshold function
        List<Double> outputProbabilities = softmax(neurons.get(neurons.size() - 1));

        // generic threshold function
        //List<Double> outputProbabilities = new ArrayList<Double>();
        //for(int j = 0; j < neurons.get(neurons.size() - 1).size(); j++){
        //    outputProbabilities.add(neurons.get(neurons.size() - 1).get(j).activationVal);
        //}
        
        int maxProb = 0;
        for(int i = 0; i < outputProbabilities.size(); i++){
            maxProb = outputProbabilities.get(maxProb) > outputProbabilities.get(i) ? maxProb: i;
        }

        // if the most likely output is correct, return that value and iterate the total number of correct guesses
        if(maxProb == correctVal){
            correctNum+=1;
        }
        
        return maxProb;
    }

    // populate first layer with inputs
    public void addNeuronInputs(List<Double> inputList, List<Neuron> layerNeurons) {
        for (int i = 1; i < inputList.size(); i++){
            layerNeurons.get(i-1).setInputVal(inputList.get(i));
        }
    }

    // Activation functions
    public double generateZValue(int layerIndx, int neuronIndx) {
        double z = 0;
        for (int i = 0; i < neurons.get(layerIndx).size(); i++) {
            z += neurons.get(layerIndx).get(i).activationVal * neurons.get(layerIndx).get(i).weights.get(neuronIndx);
        }
        z += neurons.get(layerIndx+1).get(neuronIndx).bias;
        return z;
    }

    public double sigmoid(double z) {
        return (double)1/(1 + Math.exp(-z));
    }

    // softmax thresholding function
    public List<Double> softmax(List<Neuron> layerNeurons){
        List<Double> probabilities = new ArrayList<Double>();
        double sum = 0;
        for(int j = 0; j < layerNeurons.size(); j++){
            sum += Math.exp(layerNeurons.get(j).activationVal);
        }
        for(int i = 0; i < layerNeurons.size(); i++){
            probabilities.add(Math.exp(layerNeurons.get(i).activationVal) / sum);
        }
        return probabilities;
    }

    // update weights
    public void updateWeights(int trainingDataSize, List<List< List< List< Double > > > > weightGrads){
        double newWeight = 0;
        for(int i = 0; i < neurons.size(); i++){
            for(int j = 0; j < neurons.get(i).size(); j++){
                for(int k = 0; k < neurons.get(i).get(j).weights.size(); k++){
                    double sum = 0;
                    for(int l = 0; l < weightGrads.size(); l++){
                        sum += weightGrads.get(l).get(i).get(j).get(k);
                    }
                    newWeight = neurons.get(i).get(j).weights.get(k) - (learningRate/trainingDataSize) * sum;
                    neurons.get(i).get(j).weights.set(k, newWeight);
                }
            }
        }
    }

    // update biases
    public void updateBias(int trainingDataSize, List<List< List< Double > > > biasGrads){
        for(int i = 1; i < neurons.size(); i++){
            for(int j = 0; j < neurons.get(i).size(); j++){
                double sum = 0;
                for(int k = 0; k < biasGrads.size(); k++){
                    sum += biasGrads.get(k).get(i).get(j);
                }
                neurons.get(i).get(j).bias = neurons.get(i).get(j).bias - (learningRate/trainingDataSize) * sum;
            }
        }
    }

    public void calculateBiasGradients(int correctVal){
        // bias gradient for final layer
        // for each neuron in the final layer
        for(int i = 0; i < neurons.get(neurons.size() - 1).size(); i++){
            // 1 if value is correct else 0
            int correctYVal = correctVal == i ? 1 : 0;
            // update bias gradient for each neuron in the final layer
            // ( activation value - correct output value(1 or 0) ) * activation value * ( 1 - activation value )
            neurons.get(neurons.size() - 1).get(i).biasGradient =
                (neurons.get(neurons.size() - 1).get(i).activationVal - correctYVal) * 
                neurons.get(neurons.size() - 1).get(i).activationVal * 
                (1 - neurons.get(neurons.size() - 1).get(i).activationVal);
        }
        
        // bias gradient for each layer before except for the first
        // start from second to last layer
        for(int i = neurons.size() - 2; i > 0; i--){
            double sum = 0;
            // for each neuron
            for(int j = 0; j < neurons.get(i).size(); j++){
                // find sum of bias gradient * weight for each neuron connection
                for(int k = 0; k < neurons.get(i).get(j).weights.size(); k++){
                    sum += neurons.get(i).get(j).weights.get(k) * neurons.get(i+1).get(k).biasGradient;
                }
                // sum * activation value * ( 1 - activation val )
                neurons.get(i).get(j).biasGradient = sum * 
                    neurons.get(i).get(j).activationVal * 
                    (1 - neurons.get(i).get(j).activationVal);
            }
        }
    }

    public void calculateWeightGradients(){
        // go through each layer except the final layer
        for(int i = 0; i < neurons.size() - 1; i++){
            // go through each neuron
            for(int j = 0; j < neurons.get(i).size(); j++){
                // go through each weight gradient of each neuron
                for(int k = 0; k < neurons.get(i).get(j).weightGradients.size(); k++){
                    // activation value * bias gradient
                    double weightGradient = neurons.get(i).get(j).activationVal * neurons.get(i + 1).get(k).biasGradient;
                    neurons.get(i).get(j).weightGradients.set(k, weightGradient);
                }
            }
        }
    }

    public void runNetwork(List<List<Double> > testingData, boolean isTesting){
        // ArrayList to hold correct output values for each pair
        List<Integer> correctVals = new ArrayList<Integer>(); 
        // ArrayList to hold number of occurances of each digit
        List<Integer> numberOfOccurances = new ArrayList<Integer>();
        // ArrayList to hold number of occurances for each digit in guesses for each digit correctly guessed
        List<Integer> guesses = new ArrayList<Integer>();
        // ArrayList to hold incorrect guesses
        List<Integer> incorrectGuesses = new ArrayList<Integer>();
        // ArrayList to hold incorrect guesses indicies
        List<Integer> incorrectGuessIndx = new ArrayList<Integer>();
        // digit output for network
        int output = 0;

        // add 10 zeroes to the ArrayLists for digits
        for(int indx = 0; indx < 10; indx++){
            numberOfOccurances.add(0);
            guesses.add(0);
        }

        network.correctNum = 0;
        for(int j = 0; j < testingData.size(); j++){
            // populate number of occurances for each correct value
            correctVals.add(testingData.get(j).get(0).intValue());

            network.addNeuronInputs(testingData.get(j), network.neurons.get(0));
            network.parseLayers(layerSizes.length - 1);
            output = network.parseOutputLayer(correctVals.get(j));
            // count number of guesses of each digit
            if(output == correctVals.get(j)){
                guesses.set(output, guesses.get(output)+1);
            }
            else{
                incorrectGuessIndx.add(j);
                incorrectGuesses.add(output);
            }
        }

        for(int k = 0; k < correctVals.size(); k++){
            numberOfOccurances.set(correctVals.get(k), numberOfOccurances.get(correctVals.get(k))+1);
        }



        System.out.println("Testing Output:");
        for(int i = 0; i < guesses.size(); i++){
            System.out.print(i + " = " + guesses.get(i) + "/" + numberOfOccurances.get(i) + " ");
            if(i % 5 == 0 && i > 0){
                System.out.println();
            }
        }

        double acc = (double)network.correctNum/testingData.size()*100;
        System.out.print("Accuracy = " + network.correctNum + "/" + testingData.size() + " = " + acc + "%\n");

        if(isTesting){
            try{
                System.out.println("Show incorrect guesses? Press 1 to show, any other key returns to menu");
                Scanner s = new Scanner(System.in);
                String input = s.nextLine();
                int in = Integer.valueOf(input);
                System.out.println(input);
                if(in == 1){
                    for(int i = 0; i < incorrectGuesses.size(); i++){
                        if(in == 1){
                            displayImage(testingData, incorrectGuesses.get(i), incorrectGuessIndx.get(i), correctVals.get(incorrectGuessIndx.get(i)));
                            System.out.println("Press 1 to continue, any other key returns to the menu");
                            input = s.nextLine();
                            in = Integer.valueOf(input);
    
                        }    
                    }
                    
                }

            }catch(Exception e){
            }

        }
    }

    public void displayImage(List<List<Double>> pixelBuff, int guessedVal, int testCase, int correctVal){
        System.out.print("\033[H\033[2J");   
        System.out.flush();

        System.out.println("Testing case #" + testCase + ": Correct Value: " + correctVal + " Network Value: " + guessedVal + "\n");
        
        for(int i = 1; i < pixelBuff.get(testCase).size(); i++){
            if(i > 0 && i % 28 == 0){
                System.out.println();
            }
            if(0 <= pixelBuff.get(testCase).get(i)*255 && pixelBuff.get(testCase).get(i)*255 <= 25){
                System.out.print(' ');
            }
            else if(26 <= pixelBuff.get(testCase).get(i)*255 && pixelBuff.get(testCase).get(i)*255 <= 50){
                System.out.print('.');
            }
            else if(51 <= pixelBuff.get(testCase).get(i)*255 && pixelBuff.get(testCase).get(i)*255 <= 76){
                System.out.print('\'');
            }
            else if(77 <= pixelBuff.get(testCase).get(i)*255 && pixelBuff.get(testCase).get(i)*255 <= 102){
                System.out.print('=');
            }
            else if(103 <= pixelBuff.get(testCase).get(i)*255 && pixelBuff.get(testCase).get(i)*255 <= 127){
                System.out.print(':');
            }
            else if(128 <= pixelBuff.get(testCase).get(i)*255 && pixelBuff.get(testCase).get(i)*255 <= 152){
                System.out.print('*');
            }
            else if(153 <= pixelBuff.get(testCase).get(i)*255 && pixelBuff.get(testCase).get(i)*255 <= 178){
                System.out.print('|');
            }
            else if(179 <= pixelBuff.get(testCase).get(i)*255 && pixelBuff.get(testCase).get(i)*255 <= 204){
                System.out.print('o');
            }
            else if(205 <= pixelBuff.get(testCase).get(i)*255 && pixelBuff.get(testCase).get(i)*255 <= 230){
                System.out.print('$');
            }
            else{
                System.out.print('M');
                
            }
        }
    
    }
}

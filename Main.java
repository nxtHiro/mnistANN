
// Name: Marco Flores
// Student Number: 10259733
// Date: October 29, 2020
// Computer Science 475 – Assignment 2
// MNIST Handwritten Digit Recognizer Neural Network
//
// Main class to handle user input and basic control flow

import java.io.*;
import java.util.*;
import network.*;

public class Main {
    public static boolean hasTrained = false; // is the network trained
    public static NeuralNetwork network;
    public static List<Integer> layers;

    // main UI
    public static void main(String[] args) throws Exception{
        Scanner in = new Scanner(System.in);
        while(true){

            System.out.println("\nPlease select an operating mode:\n" +
            "[1] Train the network\n" + 
                "[2] Load a pre-trained network\n" + 
                "[3] Display network accuracy on TRAINING data\n" + 
                "[4] Display network accuracy on TESTING data\n" + 
                "[5] Save the network state to file\n" +
                "[0] Exit");
            try{
                String input = in.nextLine();
                int option = Integer.valueOf(input);
                switch(option){
                    case 1: 
                    int epochs = 30;
                    int miniBatchSize = 10;
                    train(epochs, miniBatchSize);
                    break;
                    case 2: load();
                    break;
                    case 3: displayAccuracyOnTraining();
                    break;
                    case 4: displayAccuracyOnTesting();
                    break;
                    case 5: saveStateToFile();
                    break;
                    case 0: System.exit(0);
                }
            }catch(Exception e){
            }
        }
    }

    // main training functionality
    public static void train(int epochs, int miniBatchSize){

        // parse training file
        List<List<Double> > pixelBuff = readInputFile("mnist_train.csv");

        int layerOne = pixelBuff.get(0).size() - 1;
        layers = new ArrayList<Integer>(Arrays.asList(layerOne, 15, 10));
        int[] layersArray = new int[layers.size()];
        for(int i = 0; i < layersArray.length; i++){
            layersArray[i] = layers.get(i);
        }

        network = new NeuralNetwork(layersArray);
        network.parse(pixelBuff, epochs, miniBatchSize);

        hasTrained = true;
    }

    // load nn file
    public static void load(){
        List<List<String> > buffer = new ArrayList<List<String> >();
        List<List<Double> > doubleBuffer = new ArrayList<List<Double > >();
        String currLine = "";
        String delimiter = ",";
        try{
            BufferedReader br = new BufferedReader(new FileReader("weights_and_biases.nn"));  
            while ((currLine = br.readLine()) != null) {
                buffer.add(Arrays.asList(currLine.split(delimiter)));
            }
            br.close();
            // convert String elements to Double elements
            for (int i = 0; i < buffer.size(); i++){
                doubleBuffer.add(new ArrayList<Double>());
                for (int j = 0; j < buffer.get(i).size(); j++) {
                    try{
                        doubleBuffer.get(i).add(Double.parseDouble(buffer.get(i).get(j)));
                    }catch(Exception e){
                        doubleBuffer.get(i).add(0.0);
                    }
                }
            }
            List<List<List<Double> > > weights = new ArrayList<List<List<Double> > >();
            List<List<Double> > biases = new ArrayList<List<Double > >();
            int parseIndx = 0;
            // layer sizes list
                for(int i = 0; i < buffer.get(0).size() - 1; i++){
                    weights.add(new ArrayList< List<Double> >());
                    // layer size
                    for(int j = 0; j < Integer.parseInt(buffer.get(0).get(i)); j++){
                        parseIndx++;
                        weights.get(i).add(doubleBuffer.get(parseIndx));
                    }
                }
                for(int i = 0; i < buffer.get(0).size(); i++){
                    parseIndx++;
                    biases.add(doubleBuffer.get(parseIndx));
                    }

            int[] layerSizes = new int[buffer.get(0).size()];
            layers = new ArrayList<Integer>();
            for(int i = 0; i < buffer.get(0).size(); i++){
                layerSizes[i] = Integer.parseInt(buffer.get(0).get(i));
                layers.add(Integer.parseInt(buffer.get(0).get(i)));
            }
            buffer = null;
            network = new NeuralNetwork(layerSizes, weights, biases, 0);
            network.network = network;

        } catch(Exception e) {
            System.out.println("File weights_and_biases.nn is not found.");
            e.printStackTrace();
        }
        hasTrained = true;

    }

    // display accuracy after one pass of the training data
    public static void displayAccuracyOnTraining(){
        if(hasTrained){
            List<List<Double> > pixelBuff = readInputFile("mnist_train.csv");
            network.runNetwork(pixelBuff, false);
        }
    }

    // display accuracy after one pass of the testing data
    public static void displayAccuracyOnTesting(){
        if(hasTrained){
            List<List<Double> > pixelBuff = readInputFile("mnist_test.csv");
            network.runNetwork(pixelBuff, true);
        }
    }

    // save current weights and biases to a nn file
    public static void saveStateToFile(){
        if(hasTrained){
            try{
                String writeToFile = "";
    
                for(int i = 0; i < layers.size(); i++){
                    writeToFile += Integer.toString(layers.get(i));
                    if(i < layers.size() - 1)
                        writeToFile += ",";
                }
                writeToFile += "\n";
                FileWriter writer = new FileWriter("weights_and_biases.nn");
                writer.write(writeToFile);
                
                writeToFile = "";
                for(int j = 0; j < network.network.neurons.size(); j++){
                    for(int k = 0; k < network.network.neurons.get(j).size(); k++){
                        for(int l = 0; l < network.network.neurons.get(j).get(k).weights.size(); l++){
                            writeToFile += network.network.neurons.get(j).get(k).weights.get(l);
                            writeToFile += ",";
                        }
                        writeToFile += "\n";
                    }
                }
                writer.write(writeToFile);
                
                writeToFile = "";
                for(int j = 0; j < network.network.neurons.size(); j++){
                    for(int k = 0; k < network.network.neurons.get(j).size(); k++){
                        writeToFile += network.network.neurons.get(j).get(k).bias;
                        writeToFile += ",";
                    }
                    writeToFile += "\n";
                }
                writer.write(writeToFile);
                writer.close();
            } catch(Exception e){
                e.printStackTrace();
            }
        }
    }

    // read training and testing data
    public static List<List<Double> > readInputFile(String fileName) {
        List<List<String> > tmpPixelBuff = new ArrayList<List<String> >();
        List<List<Double> > pixelBuff = new ArrayList<List<Double> >();
        String currLine = "";
        String delimiter = ",";
        try{
            BufferedReader br = new BufferedReader(new FileReader(fileName));  
            while ((currLine = br.readLine()) != null) {
                tmpPixelBuff.add(Arrays.asList(currLine.split(delimiter)));
            }
            br.close();
            // convert String elements to Integer elements
            for (int i = 0; i < tmpPixelBuff.size(); i++){
                pixelBuff.add(new ArrayList<Double>());
                for (int j = 0; j < tmpPixelBuff.get(i).size(); j++) {
                    if(j != 0){
                        pixelBuff.get(i).add(Double.parseDouble(tmpPixelBuff.get(i).get(j))/255);
                    }
                    else{
                        pixelBuff.get(i).add(Double.parseDouble(tmpPixelBuff.get(i).get(j)));
                    }
                }
            }
            // tmpPixelBuff no longer needed
            tmpPixelBuff = null;

        } catch(Exception fileNotFoundException) {
            System.out.println("File " + fileName + " is not found.");
        }
        return pixelBuff;
    }

}

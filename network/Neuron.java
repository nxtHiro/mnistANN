
// Name: Marco Flores
// Student Number: 10259733
// Date: October 29, 2020
// Computer Science 475 – Assignment 2
// MNIST Handwritten Digit Recognizer Neural Network
//
// Neuron class representing a neuron for the NeuralNetwork class

package network;
import java.util.*;

public class Neuron{
    public double activationVal;
    public List<Double> weights = new ArrayList<Double>();
    public List<Double> weightGradients = new ArrayList<Double>();
    public double biasGradient;
    public double bias;

    public Neuron(){
        activationVal = 0;
        bias = Math.random() * 2 - 1;
        biasGradient = 0.0;
    }

    // generates random weights and populates weight gradient by default with zeroes
    public void generateWeights(int connections){
        for(int i = 0; i < connections; i++){
            weights.add(Math.random() * 2 - 1);
            weightGradients.add(0.0);
        }
    }

    public void setInputVal(double inputValue){
        activationVal = inputValue;
    }

    // assigns weights for predefined values
    public void setWeights(List<Double> inWeights){
        weights = inWeights;
    }

    public String toString(){
        return weights.toString();
    }
}
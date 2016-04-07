package homework2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Enumeration;

import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class TreeDriver {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static void main(String[] args) throws Exception {
		BufferedReader readTraining = readDataFile("/Users/arnonnir/Documents/workspace/HomeWork2/src/homework2/mushrooms_training.txt");
		BufferedReader readTesting = readDataFile("/Users/arnonnir/Documents/workspace/HomeWork2/src/homework2/mushrooms_training.txt");
		
		Instances instancesTraining = new Instances(readTraining);
		instancesTraining.setClassIndex(instancesTraining.numAttributes() - 1);
		
		Instances instancesTesting = new Instances(readTesting);
		instancesTesting.setClassIndex(instancesTesting.numAttributes() - 1);
		
		DecisionTree decisionTree = new DecisionTree();
		decisionTree.setPruningMode(false);
		decisionTree.buildClassifier(instancesTraining);
		
		//System.out.println(decisionTree.Classify(instancesTesting.firstInstance()));
		double error = decisionTree.CalcAvgError(instancesTesting);
		System.out.println(error);
	}
	
	
}

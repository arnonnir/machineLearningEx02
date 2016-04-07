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
		BufferedReader reader = readDataFile("/Users/arnonnir/Documents/workspace/HomeWork2/src/homework2/cancer_train.txt");
		Instances instances = new Instances(reader);
		DecisionTree tree = new DecisionTree();
		tree.setPruningMode(false);
		tree.buildClassifier(instances);
	}
}

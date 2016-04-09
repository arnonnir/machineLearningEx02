package homework2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
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
		BufferedReader readTraining = readDataFile("src/homework2/cancer_train.txt");
		BufferedReader readTesting = readDataFile("src/homework2/cancer_test.txt");
		
		Instances instancesTraining = new Instances(readTraining);
		instancesTraining.setClassIndex(instancesTraining.numAttributes() - 1);
		
		Instances instancesTesting = new Instances(readTesting);
		instancesTesting.setClassIndex(instancesTesting.numAttributes() - 1);
		
		DecisionTree decisionTreeWithoutPruning = new DecisionTree();
		decisionTreeWithoutPruning.setPruningMode(false);
		decisionTreeWithoutPruning.buildClassifier(instancesTraining);
		
		DecisionTree decisionTreeWithPruning = new DecisionTree();
		decisionTreeWithPruning.setPruningMode(true);
		decisionTreeWithPruning.setChartValue(2.733);
		decisionTreeWithPruning.buildClassifier(instancesTraining);
		
		double trainAverageErrorWithoutPruning = decisionTreeWithoutPruning.CalcAvgError(instancesTraining);
		System.out.println("The average train error of the decision tree is " + trainAverageErrorWithoutPruning);
		
		double testAverageErrorWithoutPruning = decisionTreeWithoutPruning.CalcAvgError(instancesTesting);
		System.out.println("The average test error of the decision tree is " + testAverageErrorWithoutPruning);
		
		double trainAverageErrorWithPruning = decisionTreeWithPruning.CalcAvgError(instancesTraining);
		System.out.println("The average train error of the decision tree with pruning is " + trainAverageErrorWithPruning);
		
		double testAverageErrorWithPruning = decisionTreeWithPruning.CalcAvgError(instancesTesting);
		System.out.println("The average test error of the decision tree with pruning is " + testAverageErrorWithPruning);
	}
	
	
}

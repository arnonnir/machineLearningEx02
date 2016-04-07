package homework2;

import java.text.AttributedCharacterIterator.Attribute;
import java.util.ArrayList;
import java.util.Properties;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class DecisionTree extends Classifier{
	
	private boolean m_pruningMode = false;
	private Instances trainingData;
	private int numberOfAttributes;
	public Node tree;
	
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		trainingData = instances;
		double d = trainingData.instance(0).classValue();
		numberOfAttributes = trainingData.numAttributes() - 1;
		ArrayList<Integer> instanceIndexs = convertInstacesToList(instances);
		tree = buildTree(instanceIndexs, null);
		
	}
	
	private ArrayList<Integer> convertInstacesToList(Instances instances) {
		ArrayList<Integer> convertedList = new ArrayList<Integer>();
		
		for (int i = 0; i < instances.numInstances(); i++) {
			convertedList.add(i);
		}
		
		return convertedList;
	}

	public void setPruningMode(boolean pruningMode){
		m_pruningMode = pruningMode;
	}
	
	private Node buildTree(ArrayList<Integer> instanceIndexes, Node parent) {
		if(haveSameClassifaction(instanceIndexes)) {
			Node leaf = new Node(0);
			leaf.isLeaf = true;
			leaf.setParent(parent);
			leaf.setInstanceIndexes(instanceIndexes);
			
			return leaf;
		}
		
		double maxGain = 0;
		int attributeMaxGainIndex = 0;
		int attributeNumOfValues = 0;
		
		for(int attributeIndex = 0; attributeIndex < numberOfAttributes; attributeIndex++) {
			attributeNumOfValues = trainingData.attribute(attributeIndex).numValues();
			double attributeGain = calcInfoGain(instanceIndexes, attributeNumOfValues, attributeIndex);
			
			if(maxGain < attributeGain) {
				maxGain = attributeGain;
				attributeMaxGainIndex = attributeIndex;
			}
		}
		
		Node root = new Node(attributeNumOfValues);
		root.atributeIndex = attributeMaxGainIndex;
		root.setParent(parent);
		
		int attributeMaxGainNumOfValues = trainingData.attribute(attributeMaxGainIndex).numValues();
		ArrayList<ArrayList<Integer>> arrayOfFilterdInstanceIndexes = dividesInstances(instanceIndexes, attributeMaxGainNumOfValues, attributeMaxGainIndex);
		
		for(int i = 0; i < arrayOfFilterdInstanceIndexes.size(); i++) {
			root.children[i] = buildTree(arrayOfFilterdInstanceIndexes.get(i), root);
		}
		
		return root;
	}
	
	private ArrayList<ArrayList<Integer>> dividesInstances(ArrayList<Integer> instanceIndexes, int numOfChildren, int attributeIndex) {
		ArrayList<ArrayList<Integer>> arrayOfFilterdInstanceIndexes = new ArrayList<ArrayList<Integer>>(numOfChildren);
		
		for (int i = 0; i < numOfChildren; i++) {
			arrayOfFilterdInstanceIndexes.add(new ArrayList<Integer>());
		}
		
		for (Integer instanceIndex : instanceIndexes) {
			Instance currentInstance = trainingData.instance(instanceIndex);
			int attributeValueIndex = (int)currentInstance.value(attributeIndex);
			arrayOfFilterdInstanceIndexes.get(attributeValueIndex).add(instanceIndex);
		}
		
		return arrayOfFilterdInstanceIndexes;
	}

	private boolean haveSameClassifaction(ArrayList<Integer> instanceIndexs) {
		boolean isEquals = true;
		
		for (int i = 0; i < instanceIndexs.size() - 1; i++) {
			double firstInstanceClassValue = trainingData.instance(instanceIndexs.get(i)).classValue();
			double secondInstanceClassValue = trainingData.instance(instanceIndexs.get(i + 1)).classValue();
			if (firstInstanceClassValue != secondInstanceClassValue) {
				isEquals = false;
				break;
			}
		}
		
		return isEquals;
	}

	private double calcInfoGain(ArrayList<Integer> instanceIndexs, int attributeNumOfValues, int attributeIndex) {
		ArrayList<ArrayList<Integer>> arrayOfFilterdInstanceIndexes = dividesInstances(instanceIndexs, attributeNumOfValues, attributeIndex);
		double[] prob = getProbabilities(instanceIndexs);
		// H(p/p+n, n/p+n)
		double nodeEntropy = calcEntropy(prob);
		// Expected Entropy for all children... 2/12 * H(0,1) + ...
		double expectedEntropy = calcExpectedEntropy(instanceIndexs, arrayOfFilterdInstanceIndexes);
		
		return nodeEntropy - expectedEntropy;
	}
	
	private double[] getProbabilities(ArrayList<Integer> instanceIndexes) {
		int numOfProbabilities = trainingData.numClasses();
		double[] prob = new double[numOfProbabilities];
		int numOfInstances = instanceIndexes.size();
		for (int i = 0; i < numOfInstances; i++) {
			int classValue = (int)trainingData.instance(instanceIndexes.get(i)).classValue();
			prob[classValue]++;
		}
		
		for (int i = 0; i < numOfProbabilities; i++) {
			prob[i] /= (double)numOfInstances;
		}
		
		return prob;
	}
	
	private double calcEntropy(double[] probabilities) {
		double entropy = 0;
		
		for(double probability : probabilities) {
			double log2BaseValue = Math.log10(probability) / Math.log10(2);
			entropy -= probability * log2BaseValue;
		}
		
		return entropy;
	}
	
	private double calcExpectedEntropy(ArrayList<Integer> instanceIndexs, ArrayList<ArrayList<Integer>> arrayOfFilterdInstanceIndexes) {
//		double expectedEntropy = 0;
//		int numOfInstances = arrayOfFilterdInstanceIndexes.size();
//		
//		for(int i = 0; i < numOfDistinctValues; i++) {
//			int numOfInstancesOfDistinctValue = arrayOfFilterdInstanceIndexes.get(numOfDistinctValues).size();
//			double prob = (double)numOfInstancesOfDistinctValue / (double)numOfInstances;
//			double currentEntropy = calcEntropy(allDistinctValuesProbabilties[i]);
//			expectedEntropy += prob * currentEntropy;
//		}
		
		double expectedEntropy = 0;
		int numOfChildren = arrayOfFilterdInstanceIndexes.size();
		double numOfInstances = instanceIndexs.size();
		double childEntropy = 0;
		double speceificChildProb = 0;
		double[] childProbForEntropy;
		
		for (int i = 0; i < numOfChildren; i++) {
			double numSpecificChildInstances = arrayOfFilterdInstanceIndexes.get(i).size();
			// (ni + pi) / (n + p)
			speceificChildProb = numSpecificChildInstances / numOfInstances;;
			childProbForEntropy = getProbabilities(arrayOfFilterdInstanceIndexes.get(i)); 
			childEntropy = calcEntropy(childProbForEntropy);
			expectedEntropy += speceificChildProb * childEntropy;
		}
		
		return expectedEntropy;
	}
	
	public double Classify(Instance testInstance, Node tree) {
		while (!tree.isLeaf) {
			double result = testInstance.value(tree.atributeIndex);
			tree = tree.children[(int)result];
		}
		
		return trainingData.instance(tree.instanceIndexes.get(0)).classValue();
	}

	public double CalcAvgError(Instances testingData, Node tree) {
		double errorCounter = 0;
		for (int i = 0; i < testingData.numInstances(); i++) {
			double classValue = testingData.instance(i).classValue();
			double predictValue = Classify(testingData.instance(i), tree);
			errorCounter = (classValue != predictValue) ? errorCounter++ : errorCounter; 
		}
		
		return errorCounter / (double)testingData.numInstances();
	}
}


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
		numberOfAttributes = trainingData.numAttributes() - 1;
		ArrayList<Integer> instanceIndexs = convertInstacesToList(instances);
		ArrayList<Integer> attributeIndexes = initializeAttributeIndexes();
		tree = buildTree(instanceIndexs, null, attributeIndexes);
		
	}
	
	private ArrayList<Integer> initializeAttributeIndexes() {
		ArrayList<Integer> attributeIndexes = new ArrayList<Integer>();
		
		for(int i = 0; i < numberOfAttributes; i ++) {
			attributeIndexes.add(i);
		}
		
		return attributeIndexes;
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
	
	private Node buildTree(ArrayList<Integer> instanceIndexes, Node parent, ArrayList<Integer> attributeIndexes) {
			
		boolean isSame = haveSameClassifaction(instanceIndexes);
		if(isSame || attributeIndexes.size() == 0) {
			Leaf leaf = new Leaf(0);
			leaf.setParent(parent);
			leaf.setInstanceIndexes(instanceIndexes);
			leaf.classValue = getMajorClassValue(instanceIndexes); 
			
			return leaf;
		}
		
		double maxGain = -1; // initialze to number that the gain cannot reach
		int attributeMaxGainIndex = 0;
		int maxAttributeNumOfValues = 0;
		int currentAttributeNumOfValues = 0;
		
		for(int attributeIndex : attributeIndexes) {
			currentAttributeNumOfValues = trainingData.attribute(attributeIndex).numValues();
			double attributeGain = calcInfoGain(instanceIndexes, currentAttributeNumOfValues, attributeIndex);
			
			if(maxGain < attributeGain) {
				maxGain = attributeGain;
				attributeMaxGainIndex = attributeIndex;
			}
		}
		
		attributeIndexes.remove((Object)attributeMaxGainIndex);
		
		maxAttributeNumOfValues = trainingData.attribute(attributeMaxGainIndex).numValues();
		
		Root root = new Root(maxAttributeNumOfValues);
		root.atributeIndex = attributeMaxGainIndex;
		root.setParent(parent);
		
		ArrayList<ArrayList<Integer>> arrayOfFilterdInstanceIndexes = dividesInstances(instanceIndexes, maxAttributeNumOfValues, attributeMaxGainIndex);
		
		for(int i = 0; i < arrayOfFilterdInstanceIndexes.size(); i++) {
			ArrayList<Integer> currentInstanceIndexes = arrayOfFilterdInstanceIndexes.get(i);
			
			if(currentInstanceIndexes.size() == 0) {
				Leaf leaf = new Leaf(0);
				leaf.setParent(parent);
				leaf.classValue = getMajorClassValue(instanceIndexes); // assign arbitrary value
				
				root.children[i] = leaf;
			}else {
				ArrayList<Integer> attributeIndexesCopy = copyAttributeIndexes(attributeIndexes);
				root.children[i] = buildTree(currentInstanceIndexes, root, attributeIndexesCopy);
			}
		}
		
		return root;
	}
	
	private double getMajorClassValue(ArrayList<Integer> instanceIndexes) {
		int[] classValueCounters = new int[trainingData.numClasses()];
		double majorClassValueIndex = 0;
		
		for (int instanceIndex : instanceIndexes) {
			classValueCounters[(int)trainingData.instance(instanceIndex).classValue()]++;
			
			if (classValueCounters[(int)trainingData.instance(instanceIndex).classValue()] > majorClassValueIndex) {
				majorClassValueIndex = trainingData.instance(instanceIndex).classValue();
			}
		}
		
		return majorClassValueIndex;
		
	}

	private ArrayList<Integer> copyAttributeIndexes(ArrayList<Integer> attributeIndexes) {
		ArrayList<Integer> attributeIndexesCopy = new ArrayList<Integer>();
		
		for(int attributeIndex : attributeIndexes) {
			attributeIndexesCopy.add(attributeIndex);
		}
		
		return attributeIndexesCopy;
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
		double nodeEntropy = calcEntropy(prob);
		double expectedEntropy = calcExpectedEntropy(instanceIndexs, arrayOfFilterdInstanceIndexes);
		
		return nodeEntropy - expectedEntropy;
	}
	
	private double[] getProbabilities(ArrayList<Integer> instanceIndexes) {
		int numOfInstances = instanceIndexes.size();
		int numOfProbabilities = trainingData.numClasses();
		double[] prob = new double[numOfProbabilities];
		
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
			double log2BaseValue = 0;
			if(probability != 0) {
				log2BaseValue = Math.log10(probability) / Math.log10(2);
			}
			
			entropy -= probability * log2BaseValue;
		}
		
		return entropy;
	}
	
	private double calcExpectedEntropy(ArrayList<Integer> instanceIndexs, ArrayList<ArrayList<Integer>> arrayOfFilterdInstanceIndexes) {		
		double expectedEntropy = 0;
		int numOfChildren = arrayOfFilterdInstanceIndexes.size();
		double numOfInstances = instanceIndexs.size();
		double childEntropy = 0;
		double speceificChildProb = 0;
		double[] childProbForEntropy;
		
		for (int i = 0; i < numOfChildren; i++) {
			double numSpecificChildInstances = arrayOfFilterdInstanceIndexes.get(i).size();
			speceificChildProb = numSpecificChildInstances / numOfInstances;
			
			if(numSpecificChildInstances == 0) {
				childEntropy = 0;
			}else {
				childProbForEntropy = getProbabilities(arrayOfFilterdInstanceIndexes.get(i)); 
				childEntropy = calcEntropy(childProbForEntropy);
			}
			
			expectedEntropy += speceificChildProb * childEntropy;
		}
		
		return expectedEntropy;
	}
	
	public double Classify(Instance testInstance) {
		Node temp = tree;
		while (temp.children.length != 0) { 
			double attributeValue = testInstance.value(((Root)temp).atributeIndex);
			temp = temp.children[(int)attributeValue];
		}
		
		return ((Leaf)temp).classValue;
	}

	public double CalcAvgError(Instances testingData) {
		double errorCounter = 0;
		System.out.println("number instances: " + testingData.numInstances());
		for (int i = 0; i < testingData.numInstances(); i++) {
			double classValue = testingData.instance(i).classValue();
			double predictValue = Classify(testingData.instance(i));
			boolean equals = (classValue == predictValue);
			errorCounter += !equals ? 1 : 0; 
		}
		
		return errorCounter / (double)testingData.numInstances();
	}
}


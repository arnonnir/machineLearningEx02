package homework2;

import java.util.ArrayList;
import java.util.List;

import weka.core.Instances;

public class Node {
	public Node[] children;
	public Node parent;
	public ArrayList<Integer> instanceIndexes = new ArrayList<Integer>();
	
	
	public Node(int numOfChildren) {
		instanceIndexes = new ArrayList<Integer>();
		children = new Node[numOfChildren]; 
	}
	
	public void setInstanceIndexes(ArrayList<Integer> instanceIndexes) {
		this.instanceIndexes = instanceIndexes;
	}
	
	public void setParent(Node parent) {
		this.parent = parent;
	}
}



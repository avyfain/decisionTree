from __future__ import division
import math
import operator
import time
import random
import copy
from collections import Counter

############################################################################
class TreeNode():
	def predict(self, e):
		if isinstance(self, TreeLeaf):
			return self.result
		else:
			if self.numeric:
				if e[self.attr] <= self.splitval and '<=' in self.branches:
					return self.branches['<='].predict(e)
				elif '>' in self.branches:
					return self.branches['>'].predict(e)
				else:
					return '0'
			else:
				try:    
					out = self.branches[e[self.attr]].predict(e)
				except:
					return '0'
				return out

	def disjunctive_nf(self, path):
		if isinstance(self, TreeLeaf):
			if self.result == '1':
				print "("+ str(path) + ") OR"
				return path
			else:
				return False

		else:
			for branch_label, branch in self.branches.iteritems():
				if self.numeric:
					clause = self.attr_name + " " + branch_label + " "+ str(self.splitval)
				else: 
					clause = self.attr_name + " is " + branch_label
				new_path = path + [clause]
				branch.disjunctive_nf(new_path)
		return path

	def list_nodes(self, nodes):
		if isinstance(self, TreeLeaf):
			nodes.append(self)
			return nodes
		nodes.append(self)
		for branch_label, branch in self.branches.iteritems():
			nodes = branch.list_nodes(nodes)
		return nodes

############################################################################

class TreeLeaf(TreeNode):
	def __init__(self, target_class):
		self.result = target_class

	def __repr__(self):
		return "This is a TreeLeaf with result: {0}".format(self.result)

	def toFork(self):
		self.__class__ = TreeFork
		self.result = None
############################################################################

class TreeFork(TreeNode):
	def __init__(self, attr_arr):
		self.attr =  attr_arr[0]
		self.splitval =  attr_arr[1]
		self.numeric = attr_arr[2]
		self.attr_name = attr_arr[3]
		self.mode = attr_arr[4]
		self.branches = {}

	def toLeaf(self, target):
		self.__class__ = TreeLeaf
		self.result = target

	def add_branch(self, val, subtree, default):
		self.branches[val] = subtree

	def __repr__(self):
		return "\nThis node is a fork on {0}, with {1} branches.\nMost instances in it are {2}.".format(self.attr_name,len(self.branches),self.mode)
############################################################################
class Dataset:
	def __init__(self, filename, test=False):
		self.filename = filename
		self.arrange_data()
		if not test:
			self.fill_missing()

	def arrange_data(self):
		with open(self.filename) as f:
			original_text = f.read()
			self.numeric_attrs = [True,True,True,False,False,False,False,False,True,True,True,True,False]

			# split data by instances... if last line is empty, delete it
			self.instances = [line.split(',') for line in original_text.split("\n")]

			if self.instances[-1]==['']:
				del self.instances[-1]

			# set header row as attributes, remove it from dataset
			self.attr_names = self.instances.pop(0)

			# Change strings to int where necessary
			for instance in self.instances:
				for a in range(len(self.numeric_attrs)):
					if instance[a] == '?':
						continue
					if self.numeric_attrs[a]:
						instance[a] = int(instance[a])

			# self.values = [[e[attr] for e in self.examples] for attr in self.attributes]

	def fill_missing(self):
		groups = [0, 1]
		fill_values = 2*[[None] * len(self.attr_names)]
		for g in groups:
			group_instances = [e for e in self.instances if e[-1] == str(groups[g])]
			for attr in range(len(self.attr_names[:-1])):
				if self.numeric_attrs[attr]:
					not_missing = [e[attr] for e in group_instances if e[attr] != '?']      
					fill_values[g][attr] = int(sum(not_missing) / len(not_missing))
				else:
					nominal_vals = [e[attr] for e in group_instances]
					fill_values[g][attr] = Counter(nominal_vals).most_common()[0][0]

		for instance in self.instances:
			for attr in range(len(self.attr_names)):
				if instance[attr]=='?' and instance[-1]=='0':
					instance[attr] = fill_values[0][attr]
				elif instance[attr]=='?' and instance[-1]=='1':
					instance[attr] = fill_values[1][attr]
		return True

############################################################################

def getFrequencies(examples, attributes, target):
	frequencies = {}
	#find target in data
	a = attributes.index(target)
	#calculate frequency of values in target attr
	for row in examples:
		if row[a] in frequencies:
			frequencies[row[a]] += 1 
		else:
			frequencies[row[a]] = 1
	return frequencies

############################################################################

def entropy(examples, attributes, target):

	dataEntropy = 0.0
	frequencies = getFrequencies(examples, attributes, target)
	# Calculate the entropy of the data for the target attr
	for freq in frequencies.values():
		dataEntropy += (-freq/len(examples)) * math.log(freq/len(examples), 2) 
	return dataEntropy

############################################################################

def gain(examples, attributes, attr, targetAttr):
	numeric_attrs = [True,True,True,False,False,False,False,False,True,True,True,True,False]

	currentEntropy = entropy(examples, attributes, targetAttr)
	subsetEntropy = 0.0
	i = attributes.index(attr)
	best = 0

	if numeric_attrs[i]:
		order = sorted(examples, key=operator.itemgetter(i))
		subsetEntropy = currentEntropy
		for j in range(len(order)):
			if j==0 or j == (len(order)-1) or order[j][-1]==order[j+1][-1]:
				continue

			currentSplitEntropy = 0.0
			subsets = [order[0:j], order[j+1:]]

			for subset in subsets:
				setProb = len(subset)/len(order)
				currentSplitEntropy += setProb*entropy(subset, attributes, targetAttr)

			if currentSplitEntropy < subsetEntropy:
				best = order[j][i]
				subsetEntropy = currentSplitEntropy
	else:
		valFrequency = getFrequencies(examples, attributes, attr)

		for val, freq in valFrequency.iteritems():
			valProbability =  freq / sum(valFrequency.values())
			dataSubset     = [entry for entry in examples if entry[i] == val]
			subsetEntropy += valProbability * entropy(dataSubset, attributes, targetAttr)
	
	return [(currentEntropy - subsetEntropy),best]

############################################################################
def selectAttr(examples, attributes, target):
	best = False
	bestCut = None
	maxGain = 0
	for a in attributes[:-1]:
		newGain, cut_at = gain(examples, attributes, a, target) 
		if newGain>maxGain:
			maxGain = newGain
			best = attributes.index(a)
			bestCut = cut_at
	return [best, bestCut]
############################################################################

def one_class(examples):
	first_class = examples[0][-1]
	for e in examples:
		if e[-1]!=first_class:
			return False
	return True
############################################################################

def mode(examples, index):  
	L = [e[index] for e in examples]
	return Counter(L).most_common()[0][0]

############################################################################
def splitTree(examples, attr, splitval):
	numeric_attrs = [True,True,True,False,False,False,False,False,True,True,True,True,False]
	isNum = numeric_attrs[attr]
	positive_count = 0

	if isNum:
		subsets = {'<=': [], 
					'>': []}
		for row in examples:
			if row[-1]=='1':
				positive_count += 1
			if row[attr]<=splitval:
				subsets['<='].append(row)
			elif row[attr]>splitval:
				subsets['>'].append(row)
	else:
		subsets = {}
		for row in examples:
			if row[-1]=='1':
				positive_count += 1
			if row[attr] in subsets:
				subsets[row[attr]].append(row)
			else:
				subsets[row[attr]] = [row]
	negative_count = len(examples)-positive_count
	if positive_count > negative_count:
		majority = '1'
	else:
		majority = '0'

	out = {"splitOn": splitval, "branches": subsets, "numeric": isNum, "mode": majority}
	return out

############################################################################

def learn_decision_tree(examples, attributes, default, target, iteration):
	iteration += 1

	if iteration > 10:
		return TreeLeaf(default)
	if not examples:
		tree = TreeLeaf(default)
	elif one_class(examples):
		tree = TreeLeaf(examples[0][-1])
	else:
		best_attr = selectAttr(examples, attributes, target)
		if best_attr is False:
			tree = TreeLeaf(default)

		else:
			split_examples = splitTree(examples, best_attr[0], best_attr[1]) #new decision tree with root test *best_attr*
			best_attr.append(split_examples['numeric'])
			best_attr.append(attributes[best_attr[0]])
			best_attr.append(split_examples["mode"])
			tree = TreeFork(best_attr)
			for branch_lab, branch_examples in split_examples['branches'].iteritems():
				if not branch_examples:
					break
				sub_default = mode(branch_examples, -1)
				subtree = learn_decision_tree(branch_examples, attributes, sub_default, target, iteration)
				tree.add_branch(branch_lab, subtree, sub_default)
	return tree

############################################################################
def tree_accuracy(examples, dt):
	count = 0
	correct_predictions = 0
	for row in examples:
		count += 1
		pred_val = dt.predict(row)
		if row[-1]==pred_val:
			correct_predictions+=1
	accuracy = 100*correct_predictions/len(examples)
	return accuracy

###########################################################################
def test_tree(examples, dt):
	for row in examples:
		row[-1] = dt.predict(row)
	return examples
############################################################################
def prune_tree(tree, nodes, validation_examples, old_acc):
	percent_to_try = 0.2
	nodes = random.sample(nodes, int(percent_to_try*(len(nodes))))
	reduced_by = 1000
	while reduced_by >0:
		reduction = []
		for n in nodes:
			if isinstance(n, TreeLeaf):
				nodes.pop(nodes.index(n))
				continue
			else:
				target_class = n.mode
				n.toLeaf(target_class)
				new_acc = tree_accuracy(validation_examples, tree)
				diff = new_acc - old_acc
				reduction.append(diff)
				n.toFork()
		if reduction != []:
			max_red_at = reduction.index(max(reduction))
			if isinstance(nodes[max_red_at], TreeFork):
				nodes[max_red_at].toLeaf(nodes[max_red_at].mode)
			nodes.pop(max_red_at)
			reduced_by = max(reduction)
			old_acc = tree_accuracy(validation_examples, tree)
		else:
			reduced_by = 0

	print "The new accuracy is: " + str(new_acc) + "%"
	return [tree, new_acc]

############################################################################
def main():
	now = time.time()
	target = "group"
	train_filename = "train.csv"
	validation_filename = "validate.csv"
	test_filename = "test.csv"
	train_data = Dataset(train_filename)
	validation_data = Dataset(validation_filename)
	test_data = Dataset(test_filename, True)
	
	#build tree
	default = mode(train_data.instances, -1)
	print "\nTime to learn this tree..."
	learned_tree = learn_decision_tree(train_data.instances, train_data.attr_names, default, target, 0)
	
	print "Trained on " +str(train_filename)+ ":"
	train_accuracy = tree_accuracy(train_data.instances, learned_tree)
	print "Training Accuracy= " + str(train_accuracy) + "%"

	validation_accuracy = tree_accuracy(validation_data.instances, learned_tree)
	print "Validation Accuracy= " + str(validation_accuracy) + "%"
	print "\nDISJUNCTIVE NORMAL FORM PRE PRUNING"
	dnf = learned_tree.disjunctive_nf([])
	nodes = learned_tree.list_nodes([])
	print "We have " + str(len(nodes)) + " nodes!"

	prePruningTime = time.time() - now	
	print "Pre-pruning Runtime = " + str(prePruningTime) + "\n"
	pruned_learned_tree = prune_tree(learned_tree, nodes, validation_data.instances, validation_accuracy)

	print "\nDISJUNCTIVE NORMAL FORM POST PRUNING"
	dnf_pruned = pruned_learned_tree[0].disjunctive_nf([])
	nodes_pruned = pruned_learned_tree[0].list_nodes([])
	print "We have " + str(len(nodes_pruned)) + " nodes!"
	
	print "Now this is the test set!"
	tested_set = test_tree(test_data.instances, pruned_learned_tree[0])
	print tested_set

	totalTime = time.time() - now
	print "Final Stats:"
	print "Runtime = " + str(totalTime) + "\n"
	print "Train accuracy: " + str(train_accuracy)
	print "Validation pre-pruning accuracy: " + str(validation_accuracy)
	print "Validation post-pruning accuracy: " + str(pruned_learned_tree[1])
	print "Pre-pruning tree size: " + str(len(nodes))
	print "Post-pruning tree size: " + str(len(nodes_pruned))

if __name__ == "__main__":
	main()
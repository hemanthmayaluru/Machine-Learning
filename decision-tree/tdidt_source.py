#!/usr/bin/env python3

from random import choice 
from numpy import array, dot, random, asfarray
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz as gv
import re
import math
import copy
import csv
import pandas as pd
import os.path

# This is a node for building a tree.
class TDIDTNode:

    def __init__(self, parent_id=-1, left_child_id=None,right_child_id=None):
        self.parent_id      = parent_id
        self.is_Left         = False
        #self.direction      = direction
        self.left_child_id  = left_child_id
        self.right_child_id = right_child_id
        self.is_leaf        = False
        self.outcome        = None
        # only needed to fullfill exercise requirements
        self.identifier      = 0
        self.parent_test_outcome = None
        self.pplus = None
        self.pminus = None
        self.label = None
        self.threshold = None

    def setLeftChild(self,id):
        self.left_child_id = id

    def setRightChild(self,id):
        self.right_child_id = id

    def setpplus(self,id):
        self.pplus = id
    
    def setpminus(self,id):
        self.pminus = id
    
    def setthreshold(self,id):
        self.threshold = id

    def setlabel(self,id):
        self.label = id
        
    def setdirection(self, id):
        self.direction = direction
        
    def setidentifier(self, id):
        self.identifier = id
        
    def setis_Left(self, id):
        self.is_Left = id

    def __str__(self):
        return "{} {} {} {} ".format(self.label, self.threshold, self.pplus, self.pminus)

# The function returns the information gain based on the positive and negative side split
def get_information_gain(ppos = 335, pneg = 340, npos = 0, nneg = 8):
    total = float(ppos + pneg + npos + nneg)
    p_total = float(ppos + pneg)
    n_total = float(npos + nneg)
    information_gain = entropy((ppos+npos)/total,(pneg + nneg)/total)
    if p_total > 0:
        information_gain -= p_total/total * entropy(ppos/p_total,pneg/p_total)
    if n_total > 0:
        information_gain -= n_total/total * entropy(npos/n_total,nneg/n_total)
    return information_gain

# This calculates the entropy 
def entropy(p,n):
    if n == 0:
        return p*math.log(1.0/p, 2)
    elif p == 0:
        return n*math.log(1.0/n, 2)
    return p*math.log(1.0/p, 2) + n*math.log(1.0/n, 2)

#Read csv file into a dataframe
def initialize_from_file(filename):
    """
    initialize this example set from a file as specified by the exercise
    """
    df = pd.read_csv(filename)
    return df

#This one calculates the total number of positive and negative outputs
def number_of_positives(dflocal):
    rowaxes, columnaxes = dflocal.axes
    number_of_positives = 0
    number_of_negatives = 0
    for i in range(len(rowaxes)):
        if(dflocal.iat[i,-1] == 1.0):
            number_of_positives += 1
        else :
            number_of_negatives += 1
    return number_of_positives, number_of_negatives 

# Determines the tree recursively by finding the best node heuristically
def Create_tree_TDIDT(node_list, dfa, current_node_id, tree_depth):
    current_node = node_list[current_node_id]
    
    rowaxes, columnaxes = dfa.axes
    pplus, pminus = number_of_positives(dfa)
    
    network_information_gain = 0
    final_mean = 0
    node_attribute = 0
    final_cutpoint = 0
    
    for current_column in range(len(columnaxes) - 1):
        df_temp = dfa.sort_values(by=[columnaxes[current_column]])
        sorted_array = df_temp[:][columnaxes[current_column]]
        result = df_temp[:][columnaxes[-1]]
        #print(result)
        pinnerplus = 0
        pinnerminus = 0
        max_information_gain = 0
        prev_out = 2
        for i in range(len(rowaxes)):
            if(df_temp.iat[i,-1] == 1.0):
                pinnerplus +=1
                information_gain = get_information_gain(ppos = pinnerplus, pneg = pinnerminus, 
                                     npos = (pplus - pinnerplus), nneg = (pminus - pinnerminus))
                if(information_gain > max_information_gain):
                    max_information_gain = information_gain
                    potential_cutpoint = i
                    if i > 0:
                        potential_mean = (df_temp.iat[i, current_column] + 
                        df_temp.iat[i-1, current_column])/2;
                    else:
                        potential_mean =  df_temp.iat[i, current_column];
            else:            
                pinnerminus +=1
        if(max_information_gain > network_information_gain):
            network_information_gain = max_information_gain
            node_attribute = current_column
            final_mean = potential_mean
            final_cutpoint = potential_cutpoint
#    print('network_information_gain',network_information_gain)
#    print('node_attribute',node_attribute)
#    print('final_mean',final_mean)
#    print('final_cutpoint',final_cutpoint)
#    print(columnaxes[node_attribute])
#    print('-----------------------------')

    # Updating the current array
    current_node.threshold = final_mean
    current_node.pplus = pplus
    current_node.pminus = pminus
    current_node.label = columnaxes[node_attribute]
    # The array is sorted and split
    df_temp = dfa.sort_values(by=[columnaxes[node_attribute]])
    df1 = df_temp.iloc[:final_cutpoint, :]
    df2 = df_temp.iloc[final_cutpoint:, :]
    
    if pplus == 0 or pminus == 0 or final_cutpoint == 0 or  tree_depth >= 3:
        current_node.is_leaf = True
        current_node.outcome = (pplus > pminus)
        return
    else:
        current_node.is_leaf = False

    left_node = TDIDTNode(current_node_id)
    right_node = TDIDTNode(current_node_id)

    current_node.left_child_id = len(node_list)
    current_node.right_child_id = len(node_list)+1

    # only needed to fullfill exercise requirements
    left_node.identifier = current_node.left_child_id
    right_node.identifier = current_node.right_child_id
    left_node.parent_test_outcome = "yes"
    right_node.parent_test_outcome = "no"

    node_list.append(left_node)
    node_list.append(right_node)
    node_list[current_node.left_child_id].identifier = current_node.left_child_id;
    node_list[current_node.right_child_id].identifier = current_node.right_child_id;
#    node_list[current_node.left_child_id].is_Left = True
    Create_tree_TDIDT(node_list,df1,current_node.left_child_id, tree_depth+1)
    Create_tree_TDIDT(node_list,df2,current_node.right_child_id, tree_depth+1)

    return df_temp

# Parses through the decision tree to find outcome.
def classify(row, dftest,node_list):

    current_node = node_list[0]

    while not current_node.is_leaf:
        if (dftest.get_value(row,str(current_node.label)) < current_node.threshold):
            current_node = node_list[current_node.left_child_id]
        else:
            current_node = node_list[current_node.right_child_id]
    return current_node.outcome

# Compares the predicted output to actual output and prints the likelihood
def test_data_output(dftest,node_list):
    rowaxes, columnaxes = dftest.axes
    number_of_matches = 0;
    for row in range(len(rowaxes)):
        predict_op = classify(row, dftest, node_list)
        if(dftest.iat[row,-1] == predict_op): 
            number_of_matches += 1
    print('Out of', len(rowaxes),'tests run, ',number_of_matches, 
          'matched the result which is at %',number_of_matches/len(rowaxes))

# To write the node into dot file
def Export_tree_node(node_list , index ): 
    if index == None:
        return
    Update_to_dot_file(node_list,node_list[index])
    Export_tree_node(node_list , node_list[index].left_child_id )
    Export_tree_node(node_list , node_list[index].right_child_id)

# To write the node into dot file
def Update_to_dot_file(node_list, node):
	#create node
	if(node.is_leaf and (node.outcome)):
		node_description=str(node.identifier)+" [ label=\""+node.label+"["+str(node.pplus)+" "+str(node.pminus)+"]"+"\" , fillcolor=\"#99ff99\"] ;\n"
	elif(node.is_leaf and (node.outcome == False)):
		node_description=str(node.identifier)+" [ label=\""+node.label+"["+str(node.pplus)+" "+str(node.pminus)+"]"+"\" , fillcolor=\"#ff9999\"] ;\n"
	else:
		node_description=str(node.identifier)+" [ label=\""+node.label+"["+str(node.pplus)+" "+str(node.pminus)+"]"+"\" , fillcolor=\"#ffffff\"] ;\n"

	fo.write(node_description)

	if(node.parent_id!=-1):
	#create relation
		condition = node.identifier % 2
		if(condition):
			node_relation= str(node.parent_id)+"->"+str(node.identifier) + " [labeldistance=2.5, labelangle=45, headlabel=\"<"+str(node_list[node.parent_id].threshold)+"\"] ;\n"
		else:
			node_relation=str(node.parent_id)+"->"+str(node.identifier) + " [labeldistance=2.5, labelangle=-45, headlabel=\">"+str(node_list[node.parent_id].threshold)+"\"] ;\n"
		fo.write(node_relation)
	return

def read_filename(file_type):
    file_nr = 0
    file_name = ''
    while file_nr > 5 or file_nr < 1:
        print('Please choose the',file_type, ' from options below: ')
        print('  1. gene_expression_training.csv')
        print('  2. gene_expression_test.csv')
        print('  3. gene_expression_training_memantine.csv')
        print('  4. gene_expression_test_memantine.csv')
        print('  5. Type the name of the file manually')
        file_nr = int(input(''))
        if file_nr > 0 and file_nr < 5:
            switch ={ 1:'gene_expression_training.csv' , 2:'gene_expression_test.csv' , 
                     3:'gene_expression_training_memantine.csv' , 4:'gene_expression_test_memantine.csv'}
            file_name = switch[file_nr]
        elif file_nr == 5:
            file_name = raw_input('Write the file path: ')
        else: 
            print('Please choose one of the available options.')
        
        if not os.path.isfile(file_name):
            print("The file \'" , file_name , "\' does not exists.")
            file_nr = 0
            
    return file_name

# Training and test data
training_file = read_filename('training data')
test_file = read_filename('test data')

# run TDIDT
df = initialize_from_file(training_file)
node_list = [TDIDTNode()]
k = Create_tree_TDIDT(node_list,df,0,0)

# For exporting the decision tree
fo=open("decision_tree.dot","w")
print("Name of the dot file: ",fo.name)
fo.write("digraph Tree {\nnode [shape=box, style=\"filled\", color=\"black\"] ;\n")

Export_tree_node(node_list, 0)

fo.write("}")
fo.close()

# print all nodes created by TDIDT
print('The following are the nodes created in the decision tree')
for node in node_list:
    print(node)


# Data set validation
df_validation = initialize_from_file(test_file)
test_data_output(df_validation, node_list)

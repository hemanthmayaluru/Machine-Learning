#!/usr/bin/env python3

from random import choice 
import numpy as np
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

rule_identifier = 0
default_prediction = False

# This is a node for building a tree.
class TDIDTNode:

    def __init__(self, parent_id=-1, left_child_id=None,right_child_id=None):
        self.parent_id      = parent_id
        self.is_Left         = False
        #self.direction      = direction
        self.left_child_id  = left_child_id
        self.right_child_id = right_child_id
        self.is_leaf        = False
        self.is_pruned      = False
        self.outcome        = None
        # only needed to fullfill exercise requirements
        self.identifier      = 0
        self.parent_test_outcome = None
        self.pplus = None
        self.pminus = None
        self.label = None
        self.threshold = None
        self.test_plus = 0
        self.test_minus = 0
        self.node_postives = 0
        self.rule_created = False
        
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

    def settest_plus(self, id):
        self.test_plus = id

    def setis_pruned(self, id):
        self.is_pruned = id

    def settest_minus(self, id):
        self.test_minus = id

    def setnode_positives(self, id):
        self.node_postives = id

    def setrule_created(self, id):
        self.rule_created = id
        
    def __str__(self):
        return "{} {} {} {} {} {}".format(self.label, self.threshold, self.pplus, self.pminus, self.node_postives, self.is_Left)

class ruleProperty:
    def __init__(self, parent_id=-1, child_id=None):
        self.index          = 0
        self.outcome        = None
        self.mismatch       = 0
        self.node_postives  = 0
        self.point_next     = 0
        self.pess_error     = 0
        
    def setoutcome(self,id):
        self.outcome = id

    def setpplus(self,id):
        self.node_postives = id
    
    def setpminus(self,id):
        self.mismatch = id
        
    def __str__(self):
        return "{} {} {} ".format(self.index, self.node_postives, self.mismatch)
    
# This is a node for building a tree.
class ClassificationNode:

    def __init__(self, parent_id=-1, child_id=None):
        self.parent_id      = parent_id
        self.is_Left        = False
        self.child_id       = child_id
        self.is_leaf        = False
        self.is_pruned      = False
        self.identifier      = 0
        self.label = None
        self.threshold = None
        self.greater_than = False
        self.less_than = False
        self.ignore_node = False

    def setchild_id(self,id):
        self.child_id = id
    
    def setthreshold(self,id):
        self.threshold = id

    def setlabel(self,id):
        self.label = id
        
    def setidentifier(self, id):
        self.identifier = id
        
    def setis_pruned(self, id):
        self.is_pruned = id

    def setnode_positives(self, id):
        self.node_postives = id

    def setrule_created(self, id):
        self.rule_created = id
        
    def setignore_node(self, id):
        self.ignore_node = id

    def __str__(self):
        return "{} {} {} {}".format(self.label, self.threshold, self.greater_than, self.less_than)

def create_decision(node_list, identifier, classification_rules_list, rule_property):
    save_left_state = False
    current_node = node_list[identifier]
    new_rule = []
    i = 0
    rule_property.outcome = current_node.outcome
    while current_node.parent_id != -1:
        rule_node = ClassificationNode()
        rule_node.label = current_node.label
        rule_node.identifier = i
        rule_node.is_leaf = current_node.is_leaf
        if(rule_node.is_leaf == True) :    
            rule_node.greater_than = not current_node.is_Left
            rule_node.less_than = current_node.is_Left
            save_left_state = current_node.is_Left
        else :
            rule_node.greater_than = not save_left_state
            rule_node.less_than = save_left_state
            save_left_state = current_node.is_Left
        rule_node.parent_id = i+1
        rule_node.threshold = current_node.threshold
        new_rule.append(rule_node)
        current_node = node_list[current_node.parent_id]
        i += 1
    rule_node = ClassificationNode()
    rule_node.label = current_node.label
    rule_node.identifier = i
    rule_node.outcome = current_node.outcome
    rule_node.is_leaf = current_node.is_leaf
    rule_node.greater_than = not save_left_state
    rule_node.less_than = save_left_state
    rule_node.parent_id = -1
    rule_node.threshold = current_node.threshold
    new_rule.append(rule_node)
    new_rule.reverse()
    rule_property.point_next = rule_node.identifier + 1
    new_rule.insert(0, rule_property)
    classification_rules_list.append(new_rule)
    return

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
    node_list[current_node.left_child_id].is_Left = True
    Create_tree_TDIDT(node_list,df1,current_node.left_child_id, tree_depth+1)
    Create_tree_TDIDT(node_list,df2,current_node.right_child_id, tree_depth+1)

    return df_temp

def classify_rules(row, df_test_data, node_list, expected_output, default):
    total_match = 0
    for rules in range(1, len(node_list)):
        rule_match = True
        for condition in range(1, len(node_list[rules])):
            if(node_list[rules][condition].is_leaf == False):
                if(node_list[rules][condition].greater_than == True):
                    if ((df_test_data.get_value(row,str(node_list[rules][condition].label))) < node_list[rules][condition].threshold):
                        rule_match = False
                else :
                    if ((df_test_data.get_value(row,str(node_list[rules][condition].label))) > node_list[rules][condition].threshold):
                        rule_match = False
            else:
                break
        if(rule_match == True):
            if(node_list[rules][0].outcome == expected_output):
                node_list[rules][0].node_postives += 1
                total_match = 1
            else:
                node_list[rules][0].mismatch += 1
            return total_match
    if(expected_output == default):
        total_match = 1
    
    return total_match

# Parses through the decision tree to find outcome.
def classify(row, dftest,node_list, expected_output):

    current_node = node_list[0]

    while not current_node.is_leaf:
        # This part is counting the number of times a particular output is parsing through a 
        # specific node - Same logic 8 lines down in case something here changes
        if(expected_output == 1):
            current_node.test_plus += 1
        else:
            current_node.test_minus += 1
        #Traverses through the node based on the best fit condition
        rowaxis, colaxis = dftest.axes

        if ((dftest.get_value(row,str(current_node.label))) < current_node.threshold):
            current_node = node_list[current_node.left_child_id]
        else:
            current_node = node_list[current_node.right_child_id]
    # This part is counting the number of times a particular output is parsing through a 
    # specific node - Logic repeated
    if(expected_output == 1):
        current_node.test_plus += 1
    else:
        current_node.test_minus += 1
    # This data will be used for pruning
    current_node.is_pruned = True
    
    return current_node.outcome

# Compares the predicted output to actual output and prints the likelihood
def test_data_output(dftest,node_list):
    rowaxes, columnaxes = dftest.axes
    number_of_matches = 0;
    for rowip in range(len(rowaxes) - 1):
        row = rowip + 1# this is added to avoid processing 0th row
        predict_op = classify(row, dftest, node_list, dftest.iat[row,-1])
        if(dftest.iat[row,-1] == predict_op): 
            number_of_matches += 1
    print('Out of', len(rowaxes),'tests run, ',number_of_matches, 
          'matched the result which is at %',number_of_matches/len(rowaxes))

def parse_classificatoin_data(df_test_data, node_list):
    global     default_prediction
    rowaxes, columnaxes = df_test_data.axes
    number_of_matches = 0
    number_of_mismatches = 0
    default_true_condition = 0
    for row in range(1, len(rowaxes) - 1):
        if(df_test_data.iat[row,-1]) :
            default_true_condition += 1
    if(default_true_condition > (len(rowaxes) - default_true_condition)):
        default_prediction = True
    for row in range(1, len(rowaxes) - 1):
        positive_match = classify_rules(row, df_test_data, node_list, df_test_data.iat[row,-1], default_prediction)
        number_of_matches += positive_match
    for rules in range(1, len(node_list)):
        node_list[rules][0].pess_error = pessimistic_error(node_list[rules][0].node_postives,(node_list[rules][0].node_postives + node_list[rules][0].mismatch))
    print('Classification rule: Out of classification rules', len(rowaxes),
          'tests run, ',number_of_matches,'matched the result which is at %',number_of_matches/len(rowaxes))

def find_positives_in_subtree(node_list, current_node_id):
    
    parent_node = node_list[current_node_id]
    if(node_list[parent_node.left_child_id].is_leaf == True):
        if(node_list[parent_node.left_child_id].outcome == 1):
            parent_node.node_postives += node_list[parent_node.left_child_id].test_plus
        else :
            parent_node.node_postives += node_list[parent_node.left_child_id].test_minus
    else :
        parent_node.node_postives += node_list[parent_node.left_child_id].node_postives
        
    if(node_list[parent_node.right_child_id].is_leaf == True):
        if(node_list[parent_node.right_child_id].outcome == 1):
            parent_node.node_postives += node_list[parent_node.right_child_id].test_plus
        else :
            parent_node.node_postives += node_list[parent_node.right_child_id].test_minus
    else :
        parent_node.node_postives += node_list[parent_node.right_child_id].node_postives
        
    return (parent_node.node_postives/(parent_node.test_plus + parent_node.test_minus))

def pessimistic_error(e, n):
    z = 0.674
    observed_error = 1 - e
    pess_error = 0
    if(n > 0) :
        pess_error = observed_error + np.power(z,2)/(2*n) + z*np.sqrt(observed_error/n -np.power(observed_error,2)/n+ np.power(z/(2*n),2))
        pess_error = pess_error/ (1+np.power(z,2)/n)
    return pess_error

def classify_for_updated_rules(row, df_test_data, node_list, expected_output, default):
    total_match = 0
    for rules in range(1, len(node_list)):
        rule_match = True
        for condition in range(1, len(node_list[rules])):
            if(node_list[rules][condition].is_leaf == False):
                if(node_list[rules][condition].greater_than == True):
                    if ((df_test_data.get_value(row,str(node_list[rules][condition].label))) < node_list[rules][condition].threshold):
                        rule_match = False
                else :
                    if ((df_test_data.get_value(row,str(node_list[rules][condition].label))) > node_list[rules][condition].threshold):
                        rule_match = False
            else:
                break
        if(rule_match == True):
            if(node_list[rules][0].outcome == expected_output):
                node_list[rules][0].node_postives += 1
                total_match = 1
            else:
                node_list[rules][0].mismatch += 1
            return total_match
    if(expected_output == default):
        total_match = 1
    
    return total_match

def parse_for_updated_rules(df_test_data, node_list):
    global     default_prediction
    rowaxes, columnaxes = df_test_data.axes
    number_of_matches = 0
    number_of_mismatches = 0
    default_true_condition = 0
    save_condition = 0
    # For pruning the data a node is ignored and the testing is performed again.
    for rules in range(1, len(node_list)):
        for condition in range(1, len(node_list[rules])):
            if(node_list[rules][condition].is_leaf == True):
                node_list[rules][save_condition].is_leaf = True
            
                for row in range(1, len(rowaxes) - 1):
                    positive_match = classify_for_updated_rules(row, df_test_data, node_list, df_test_data.iat[row,-1], default_prediction)
                    number_of_matches += positive_match
                    previous_error = node_list[rules][0].pess_error

                node_list[rules][0].pess_error = pessimistic_error(node_list[rules][0].node_postives,(node_list[rules][0].node_postives + node_list[rules][0].mismatch))
                if(previous_error < node_list[rules][0].pess_error):
                    node_list[rules][save_condition].is_leaf = False
            else:
                save_condition = condition
                    
            
    print('Classification rule: Out of classification rules', len(rowaxes),
          'tests run, ',number_of_matches,'matched the result which is at %',number_of_matches/len(rowaxes))


# Parses through the decision tree to find outcome.
def prune_the_tree(node_list, current_node_id):

    current_node = node_list[current_node_id]
    if((current_node.is_pruned == True) or (current_node.is_leaf == True)):
        return
#    if(current_node.left_child_id == None) or (current_node.left_child_id == None):
#        current_node.is_leaf = True
#        return
    while not ((node_list[int(current_node.left_child_id)].is_pruned == True) and 
                (node_list[int(current_node.right_child_id)].is_pruned == True)):

        if not (node_list[current_node.left_child_id].is_pruned):
            current_node = node_list[current_node.left_child_id]
        else:
            current_node = node_list[current_node.right_child_id]
    current_node.is_pruned = True
    subtree_expectation = find_positives_in_subtree(node_list, current_node.identifier)
    
    if(current_node.test_plus > current_node.test_minus):
        node_expectation = current_node.test_plus/(current_node.test_plus + current_node.test_minus)
    else :
        node_expectation = current_node.test_minus/(current_node.test_plus + current_node.test_minus)
    
    node_pessimistic_error = pessimistic_error(node_expectation,(current_node.test_plus + current_node.test_minus))
    subtree_pessimistic_error = pessimistic_error(subtree_expectation,(current_node.test_plus + current_node.test_minus))
#    print('PE:', node_pessimistic_error)
#    print('PE:', subtree_pessimistic_error)
    if((subtree_pessimistic_error >= node_pessimistic_error) and (current_node.identifier != 0)):
        current_node.outcome = (current_node.test_plus > current_node.test_minus)
        current_node.is_leaf = True
        current_node.is_pruned = True
        print('Node', current_node,' is deleted as a part of pruning')

    prune_the_tree(node_list, current_node.parent_id)
    return
                
# To write the node into dot file
def Export_tree_node(node_list , index ): 
    if node_list[index].is_leaf:
        Update_to_dot_file(node_list,node_list[index])
        return
    Update_to_dot_file(node_list,node_list[index])
    Export_tree_node(node_list , node_list[index].left_child_id )
    Export_tree_node(node_list , node_list[index].right_child_id)

# To write the node into dot file
def Update_to_dot_file(node_list, node):
    #create node
    if(node.is_leaf and (node.outcome)):
        node_description=str(node.identifier)+" [ label=\""+node.label+"["+str(node.test_plus)+" "+str(node.test_minus)+"]"+"\" , fillcolor=\"#99ff99\"] ;\n"
    elif(node.is_leaf and (node.outcome == False)):
        node_description=str(node.identifier)+" [ label=\""+node.label+"["+str(node.test_plus)+" "+str(node.test_minus)+"]"+"\" , fillcolor=\"#ff9999\"] ;\n"
    else:
        node_description=str(node.identifier)+" [ label=\""+node.label+"["+str(node.test_plus)+" "+str(node.test_minus)+"]"+"\" , fillcolor=\"#ffffff\"] ;\n"

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

def create_classification_rule(node_list, classification_rules_list):
    global rule_identifier
    current_node = node_list[0]
    if((node_list[current_node.left_child_id].rule_created) and 
       (node_list[current_node.right_child_id].rule_created)):
        current_node.rule_created = True
        if(current_node.identifier == 0):
            return
        else :
            create_classification_rule(node_list, classification_rules_list)
            return
    else:
        while not current_node.is_leaf:
            if(node_list[current_node.left_child_id].rule_created == False):
                current_node = node_list[current_node.left_child_id]
            elif(node_list[current_node.right_child_id].rule_created == False):
                current_node = node_list[current_node.right_child_id]
            else:
                current_node.rule_created = True
                create_classification_rule(node_list, classification_rules_list)
                return
        rule_property = ruleProperty()
        rule_identifier += 1
        rule_property.index = rule_identifier
        rule_property.point_next = rule_identifier + 1
        create_decision(node_list,current_node.identifier, classification_rules_list, rule_property)
        current_node.rule_created = True
        create_classification_rule(node_list, classification_rules_list)
    return classification_rules_list
                
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
            switch ={ 1:'gene_expression_training.csv' ,
                     2:'gene_expression_test.csv' , 
                     3:'gene_expression_training_memantine.csv' ,
                     4:'gene_expression_test_memantine.csv'}
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

# Split the data into two parts with the ratio 2:1
rowaxes, columnaxes = df.axes
split_point = int(0.65*len(rowaxes))

training_set = df.iloc[:split_point, :]
pruning_set = df.iloc[split_point:, :]
pruning_set.columns = columnaxes
pruning_set = pruning_set.reset_index(drop = True)

print('The training data is split at row', split_point, 'into training set and pruning set')

node_list = [TDIDTNode()]
    
k = Create_tree_TDIDT(node_list,training_set,0,0)
print('The tree has been created..starting pruning')

prepruned_node = copy.deepcopy(node_list)
classification_rules_list = [[]]
classification_rules_list = create_classification_rule(prepruned_node, classification_rules_list)

parse_classificatoin_data(pruning_set, classification_rules_list)

#for rows in range(len(classification_rules_list)):
#    print('The new rule class created is below', rows)
#    for nodes in range(len(classification_rules_list[rows])):
#        print(classification_rules_list[rows][nodes])

# Data set validation
df_validation = initialize_from_file(test_file)
test_data_output(pruning_set, node_list)

print('The test data is used for determining the accuracy using classification rules')
print('-------------------------------------------------------------------')
parse_classificatoin_data(df_validation, classification_rules_list)
print('-------------------------------------------------------------------')
#rules_without_pruning = copy.deepcopy(classification_rules_list)

# parse_for_updated_rules(pruning_set,rules_without_pruning)

# print all nodes created by TDIDT
#print('----------------------------------------------------------------------')
#print('Before Pruning: The following are the nodes created in the decision tree')
#for node in prepruned_node:
#    print(node)

# For exporting the decision tree
fo=open("decision_tree_prepruning.dot","w")
print("Name of the dot file: ",fo.name)
fo.write("digraph Tree {\nnode [shape=box, style=\"filled\", color=\"black\"] ;\n")

Export_tree_node(node_list, 0)

fo.write("}")
fo.close()

print('----------------------------------------------------------------------')
prune_the_tree(node_list, 0)

# print all nodes created by TDIDT
#print('----------------------------------------------------------------------')
#print('Post pruning: The following are the nodes created in the decision tree')
#for node in node_list:
#    print(node)
#print('----------------------------------------------------------------------')

# For exporting the decision tree
fo=open("decision_tree.dot","w")
print("Name of the dot file: ",fo.name)
fo.write("digraph Tree {\nnode [shape=box, style=\"filled\", color=\"black\"] ;\n")

Export_tree_node(node_list, 0)

fo.write("}")
fo.close()

print('The test data is used for determining the accuracy using pessimistic_error based pruning')
print('-------------------------------------------------------------------')
test_data_output(df_validation, node_list)
print('-------------------------------------------------------------------')

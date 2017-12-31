"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import numpy as np

FLAGS = None

numNetworks = 100
numNodesPerNetwork = 100
inputFeatures = 10
outputFeatures = 5

globalInnovationNumber = 0

assert inputFeatures < numNodesPerNetwork
assert outputFeatures < numNodesPerNetwork

class Node:
    """
    This represents a neural network node with an integer label nodeId, 
    and a special type, one of ["input", "output", "hidden"]
    """
    def __init__(self, nodeId, nodeType):
        self.nodeId = nodeId
        assert nodeType == "input" or nodeType == "output" or nodeType == "hidden"
        self.nodeType = nodeType
        
    def copy(self):
        return Node(self.nodeId, self.nodeType)    
        
    def __str__(self):
        return "Node: " + str(self.nodeId) + ", " + self.nodeType
    
        
class Edge:
    """
    This class represent a connection (axon) in a neuroevolution network with its data.
    nodeIn and nodeOut are the ids of the input and output nodes,
    enabled is a boolean representing its ability to propogate into future neural networks,
    innovId is the labeled innovation number which corresponds to the same edge in other NNs,
    and weight is the normal edge weight.
    """
    def __init__(self, nodeIn, nodeOut, enabled, innovId, weight):
        self.nodeInId = nodeIn
        self.nodeOutId = nodeOut
        self.enabled = enabled
        self.innovId = innovId
        self.weight = weight
        
    def copy(self):
        return Edge(self.nodeInId, self.nodeOutId, self.enabled, self.innovId, self.weight)
        
    def __str__(self):
        return "Edge: " + str(self.nodeInId) + " -> " + str(self.nodeOutId) + ", Enabled: " + \
            str(self.enabled) + ", InnovID: " + str(self.innovId) + ", Weight: " + str(self.weight)
    
class NeuroevolutionNetwork:
    
    # self.nodes is a dictionary indexed by (node id -> node object)
    # self.edges is indexed by (in node id -> edge object, from in node to out node)
    def __init__(self, numNodes, numInput, numOutput):
        assert numInput + numOutput < numNodes
        
        self.nodes = dict() 
        self.edges = dict()
        self.listEdges = []
        self.numNodesCounter = 0
        
        self.numInput = numInput
        self.numOutput = numOutput
        self.initNodes = numNodes
        
        for _ in range(numInput):
            self.addNode("input")
        for _ in range(numOutput):
            self.addNode("output")
        for _ in range(numNodes - numInput - numOutput):
            self.addNode("hidden")
        
    # Special copy method to handle it ourselves, so it's not just a shallow/deep clone
    def copy(self):
        copyNetwork = NeuroevolutionNetwork(self.initNodes, self.numInput, self.numOutput)
        
        for nodeId, node in self.nodes.items():
            # For every node id, copy its node to the new network
            copyNetwork.nodes[nodeId] = node.copy()
            
        for nodeId, edgeList in self.edges.items():
            # For every node in id, copy the list of outgoing edges to the new network
            copyEdgeArr = []
            for edge in edgeList:
                copyEdgeArr.append(edge.copy())
            copyNetwork.edges[nodeId] = copyEdgeArr  
            
        for pair in self.listEdges:
            copyNetwork.listEdges.append(tuple(pair))
            
        return copyNetwork
    
    def printAllEdges(self):
        for nodeInId, nodeArr in self.edges.items():
            print(nodeInId)
            for node in nodeArr:
                print(str(node))
                
    def numNodes(self):
        return len(self.nodes)
    
    def numEdges(self):
        return len(self.listEdges)
        
    def addNode(self, nodeType):
        nodeId = self.numNodesCounter
        self.nodes[nodeId] = Node(nodeId, nodeType)
        self.numNodesCounter += 1
        return nodeId
        
    def hasNode(self, nodeId):
        return nodeId in self.nodes    
        
    # Return true if a connection exists between nodes with these ids,
    # a connection from in -> out
    def isConnected(self, nodeInId, nodeOutId):    
        """if hasNode(nodeInId) and hasNode(nodeOutId):
            if nodeInId in self.edges:
                connectionsOut = self.edges[nodeInId]
                for nodeOut in connectionsOut:
                    if nodeOut.nodeId == nodeOutId:
                        return True
        return False"""
        return (nodeInId, nodeOutId) in self.listEdges
    
    # Return true if there is a valid connection from in -> out
    # Invariant to whether or not the nodes are connected or not
    def allowedConnection(self, nodeInId, nodeOutId):
        if self.hasNode(nodeInId) and self.hasNode(nodeOutId):
            inNode = self.nodes[nodeInId]
            outNode = self.nodes[nodeOutId]
            doubleInput = inNode.nodeType == "input" and outNode.nodeType == "output"
            doubleOutput = inNode.nodeType == "input" and outNode.nodeType == "output"
            return not doubleInput and not doubleOutput
        else:
            return False
        
    # Variant to whether nodes are connected. To be able to add a new connection, the nodes
    # must follow these conditions:
    """
    Nodes exist
    Nodes are not connected in either direction
    The end node is not an input
    The start node is not an output
    """
    def canAddConnection(self, nodeInId, nodeOutId):
        assert nodeInId != nodeOutId
        if not self.allowedConnection(nodeInId, nodeOutId):
            return False
        inToOutC = self.isConnected(nodeInId, nodeOutId)
        outToInC = self.isConnected(nodeOutId, nodeInId)
        endInput = self.nodes[nodeOutId].nodeType == "input"
        startOutput = self.nodes[nodeInId].nodeType == "output"
        return not (inToOutC or outToInC or endInput or startOutput)
        
    # Select a random pair of nodes, that have no connection either way,
    # and are not both input and not both output,
    # and add a connection between them, from in -> out.
    def mutateAddConnection(self):
        assert len(self.nodes) >= 2
        
        # Generate a shuffled list of possible future connections
        availIds = [node.nodeId for _,node in self.nodes.items()]
        lenIds = len(availIds)
        availablePairs = [(availIds[i], availIds[j]) for i in range(lenIds) for j in range(lenIds)]
        random.shuffle(availablePairs)
        
        # Find two nodes that have no connection
        # Once found, add the one new connection, and return
        while True: 
            if len(availablePairs) == 0: # No more possible connections to check
                return
            pair = availablePairs.pop() 
            if pair[0] == pair[1]:
                continue
            # Select a random connection, that is neither a self connection (does not exist),
            # nor an existing connection in either direction.
            # inToOutC = self.isConnected(pair[0], pair[1])
            # outToInC = self.isConnected(pair[1], pair[0])
            if self.canAddConnection(pair[0], pair[1]):
                global globalInnovationNumber
                self.addEdgeWithInnovId(pair[0], pair[1], globalInnovationNumber, edgeWeight=0.5)
                globalInnovationNumber += 1
                return
        
    def removeEdge(self, nodeInId, nodeOutId):
        assert nodeInId in self.edges
        assert (nodeInId, nodeOutId) in self.listEdges
        
        self.listEdges.remove((nodeInId, nodeOutId))
        
        # Look for the correct node within the outgoing connections
        # i.e. find the connection in -> out
        for nodeOutIndex in range(len(self.edges[nodeInId])):
            nodeOut = self.edges[nodeInId][nodeOutIndex]
            if nodeOut.nodeOutId == nodeOutId:
                self.edges[nodeInId].pop(nodeOutIndex)
                return    
            
    def addEdgeWithInnovId(self, nodeInId, nodeOutId, innovId, edgeWeight):
        assert self.canAddConnection(nodeInId, nodeOutId)
        
        newEdge = Edge(nodeInId, nodeOutId, True, globalInnovationNumber, edgeWeight)
        
        self.listEdges.append((nodeInId, nodeOutId))
        if nodeInId not in self.edges:
            self.edges[nodeInId] = []
        self.edges[nodeInId].append(newEdge)
            
    """
    This mutates the graph by finding a random existing edge,
    and then splitting into two edges with a new connection in the middle,
    as per Stanley & Miikkulainen.
    """    
    def mutateSplitConnection(self):
        # Find a random edge to mutate, assuming one exists
        assert len(self.edges) > 0
        randomEdgeIndex = int(len(self.listEdges) * random.random())
        oldEdge = tuple(self.listEdges[randomEdgeIndex])
        nodeInId, nodeOutId = oldEdge[0], oldEdge[1]
        
        # TEMP:
        self.removeEdge(nodeInId, nodeOutId)
        # No, do not remove the edge. It must be preserved to play a part in later
        # NN "reproduction" and similarity measures
        
        newMiddleNode = self.addNode("hidden")
        
        global globalInnovationNumber
        self.addEdgeWithInnovId(nodeInId, newMiddleNode, globalInnovationNumber, edgeWeight=0.5)
        globalInnovationNumber += 1
        self.addEdgeWithInnovId(newMiddleNode, nodeOutId, globalInnovationNumber, edgeWeight=0.5)
        globalInnovationNumber += 1
        
        
    def forwardStep(self, inputData):
        inputData = np.array(inputData)
        assert inputData.shape[0] == self.numInput
        
        for i in range(len(inputData)):
            nodeData[i] = inputData[i]
            
        # Use Dijkstra's algorithm to update the NN as we go, 
        # using the input data as the starting fringe.
        
        # Ideally, the NN is a DAG, but this will work even if not.
        
        nodeData = dict()
        prev = dict()
        marked = dict()
        
        queue = [i for i in range(len(self.numInput))]
        
        for nodeId, _ in self.nodes.items():
            nodeData[nodeId] = -9999
            prev[nodeId] = -1
            marked[nodeId] = False
        
        while len(queue) > 0:
            v = queue.pop() # A node id
            marked[v] = True
            for edge in self.edges[v]:
                w = edge.nodeOutId
                if :
                    nodeData[]
                    
                    
"""
Calculate the distance delta between two neuroevolution networks.
This heuristic is important in determining a normalized fitness score,
which rewards successful organisms but ensures that one species does not take over.

In Stanley, Miikkulainen, this is defined as

delta = c_1 * E / N + c_2 * D / N + c_3 * W
where E is the number of excess genes,
D is the number of disjoint genes, 
N is the normalization factor (number of genes total),
and W is the average weight distance between shared genes.

Ideally, the weights have constraint 0 <= c_1, c_2, c_3 <= 1, 
and should also be consistent.
"""
def neuroNetworkDiff(nn1, nn2, c1, c2, c3):
    # Calculate excess number of genes by count
    excess = abs(nn1.numEdges() - nn2.numEdges())
    
    # Disjoint genes just need to be counted,
    # shared genes must have their weight differences averaged
    disjoint = 0
    sharedGenesDiff = []
    
    numGenes = max(nn1.numEdges(), nn2.numEdges())
    
    firstEdgeInnovIds = set()
    weightsByInnovId = dict() # Guaranteed to be surjective, stores weights of the first NN
    
    for _,outEdges in nn1.edges.items():
        for edge in outEdges:
            firstEdgeInnovIds.add(edge.innovId)
            weightsByInnovId[edge.innovId] = edge.weight
    
    for _,outEdges in nn2.edges.items():
        for secondEdge in outEdges:
            if secondEdge.innovId in firstEdgeInnovIds:
                firstEdgeWeight = weightsByInnovId[secondEdge.innovId]
                secondEdgeWeight = secondEdge.weight
                # secondEdge = nn2.edges[node.nodeId]
                sharedGenesDiff.append(abs(firstEdgeWeight - secondEdgeWeight))
            else:
                disjoint += 1

    avgDiffW = np.sum(sharedGenesDiff) / len(sharedGenesDiff)
    
    print(excess, disjoint, numGenes, avgDiffW)
    
    return c1 * excess / numGenes + \
        c2 * disjoint / numGenes + \
        c3 * avgDiffW
        
        
def main(_):
  pass

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
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
    
class Edge:
  
  """
  This class represent a connection (axon) in a neuroevolution network with its data.
  nodeIn and nodeOut are the ids of the input and output nodes,
  enabled is a boolean representing its ability to propogate into future neural networks,
  innovId is the labeled innovation number which corresponds to the same edge in other NNs,
  and weight is the normal edge weight.
  """
  def __init__(self, nodeIn, nodeOut, enabled, innovId, weight):
    self.nodeIn = nodeIn
    self.nodeOut = nodeOut
    self.enabled = enabled
    self.innovId = innovId
    self.weight = weight
  
class NeuroevolutionNetwork:
  
  # self.nodes is a dictionary indexed by (node id -> node object)
  # self.edges is indexed by (in node id -> edge object, from in node to out node)
  def __init__(self, numNodes, numInput, numOutput):
    self.nodes = dict() 
    self.edges = dict()
    self.listEdges = []
    self.numNodesCounter = 0
    
  def addNode(nodeType):
    nodeId = self.numNodesCounter
    self.nodes.append(Node(nodeId, nodeType))
    self.numNodesCounter += 1
    return nodeId
      
  def hasNode(nodeId):
    return nodeId in self.nodes
      
  # Return true if a connection exists between nodes with these ids,
  # a connection from in -> out
  def isConnected(nodeInId, nodeOutId):    
    """if hasNode(nodeInId) and hasNode(nodeOutId):
        if nodeInId in self.edges:
            connectionsOut = self.edges[nodeInId]
            for nodeOut in connectionsOut:
                if nodeOut.nodeId == nodeOutId:
                    return True
    return False"""
    return (nodeInId, nodeOutId) in listEdges

  # Return true if there is a valid connection from in -> out
  # Invariant to whether or not the nodes are connected or not
  def allowedConnection(nodeInId, nodeOutId):
    if hasNode(nodeInId) and hasNode(nodeOutId):
      inNode = self.nodes[nodeInId]
      outNode = self.nodes[nodeOutId]
      doubleInput = inNode.nodeType == "input" and outNode.nodeType == "output"
      doubleOutput = inNode.nodeType == "input" and outNode.nodeType == "output"
      return not doubleInput and not doubleOutput
    else:
      return False
      
  def mutateAddConnection():
    assert len(self.nodes) >= 2
    availableIds = [node.nodeId for node in self.nodes]
    availablePairs = [(availableIds[i], availableIds[j]) for i in range(availableIds) for j in range(availableIds) ]
    while True: # Find two nodes that have no connection
      if len(availablePairs) == 0:
        return
      shuffle(availablePairs)
      pair = availablePairs.pop()
      if pair[0] == pair[1]:
        continue
      if allowedConnection(pair[0], pair[1]) and not isConnected(pair[0], pair[1]):
        newEdge = Edge(pair[0], pair[1], True, globalInnovationNumber, 0.5)
        self.listEdges.append(tuple(pair))
        globalInnovationNumber += 1
        self.edges[pair[0]].append(newEdge)
        return
         
  def removeEdge(nodeInId, nodeOutId):
    self.listEdges.remove((nodeInId, nodeOutId))
    
    # Look for the correct node within the outgoing connections
    for nodeOutIndex in range(len(self.edges[nodeInId])):
      nodeOut = self.edges[nodeInId][nodeOutIndex]
      if nodeOut.nodeId == nodeOutId:
        self.edges[nodeInId].pop(nodeOutIndex)
        return    
      
  def mutateSplitEdge():
    # Find a random edge to mutate, assuming one exists
    assert len(self.edges) > 0
    randomEdgeIndex = int(len(self.listEdges) * random.random())
    oldEdge = tuple(self.listEdges[randomEdgeIndex])
    nodeInId, nodeOutId = oldEdge[0], oldEdge[1]
    
    removeEdge(nodeInId, nodeOutId)
    
    newMiddleNode = self.addNode("hidden")
    
    newEdge1 = Edge(nodeInId, newMiddleNode, True, globalInnovationNumber, 0.5)
    self.listEdges.append(tuple(pair))
    globalInnovationNumber += 1
    self.edges[nodeInId].append(newEdge1)
    
    newEdge2 = Edge(newMiddleNode, nodeOutId, True, globalInnovationNumber, 0.5)
    self.listEdges.append(tuple(pair))
    globalInnovationNumber += 1
    self.edges[newMiddleNode].append(newEdge2)
        
def main(_):
  pass

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neuroevolution

import tensorflow as tf
import numpy as np

FLAGS = None

numNetworks = 100
numNodesPerNetwork = 100
inputFeatures = 10
outputFeatures = 5

# Unit tests for neuroevolution network

testNet = NeuroevolutionNetwork(numNodesPerNetwork, inputFeatures, outputFeatures)

# Nodes are initialized correctly
"""
for k,v in testNet.nodes.items():
    print(v)
"""
    
# Edges are initialized correctly
print("-----------------------------------")
print("Test #2")
print(testNet.edges, testNet.listEdges)

print("-----------------------------------")
print("Test #3")
for _ in range(5):
    testNet.mutateAddConnection()

testNet.printAllEdges()
    
print(testNet.listEdges)

print("-----------------------------------")
print("Test #4")

testEdge = testNet.listEdges[0]
testNet.removeEdge(testEdge[0], testEdge[1])

testNet.printAllEdges()

print(testNet.listEdges)

print("-----------------------------------")
print("Test #5")

testNet.mutateSplitConnection()

testNet.printAllEdges()

print(testNet.listEdges)

print("-----------------------------------")
print("Test #6")

testNetCopy = testNet.copy()
testNetCopy.mutateAddConnection()

print("Original network: ")
testNet.printAllEdges()
print("Copy network: ")
testNetCopy.printAllEdges()
        
print("-----------------------------------")
print("Test #7")

testNet = NeuroevolutionNetwork(numNodesPerNetwork, inputFeatures, outputFeatures)
for _ in range(10):
    testNet.mutateAddConnection()
    
testNetMutate = testNet.copy()

for _ in range(2):
    testNetMutate.mutateAddConnection()
for _ in range(5):
    testNetMutate.mutateSplitConnection()
    
testNet.printAllEdges()
testNetMutate.printAllEdges()

print(neuroNetworkDiff(testNet, testNetMutate, 0.5, 0.25, 0.2))

def main(_):
  pass

"""
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
"""
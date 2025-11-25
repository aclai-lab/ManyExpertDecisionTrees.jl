module ManyExpertDecisionTrees

using SoleLogics
using SoleLogics.ManyValuedLogics
using DecisionTree
using FuzzyLogic

export Leaf, Node, ManyExpertDecisionTree

# Include submodules/files
include("many-expert-decision-tree.jl")

end
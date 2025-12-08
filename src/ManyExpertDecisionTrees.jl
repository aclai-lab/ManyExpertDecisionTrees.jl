module ManyExpertDecisionTrees

using SoleLogics
using DecisionTree
using FuzzyLogic
using Statistics

export ManyExpertDecisionTree, depth

include("many-expert-decision-tree.jl")

export manify, addexperts!

include("manify.jl")
include("utils.jl")

end
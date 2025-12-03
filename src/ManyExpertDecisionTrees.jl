module ManyExpertDecisionTrees

using SoleLogics
using SoleLogics.ManyValuedLogics
using DecisionTree
using FuzzyLogic
using Statistics

export ManyExpertDecisionTree, depth

include("many-expert-decision-tree.jl")

export manify

include("manify.jl")
include("utils.jl")

end
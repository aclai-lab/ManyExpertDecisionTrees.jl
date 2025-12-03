module ManyExpertDecisionTrees

using SoleLogics
using SoleLogics.ManyValuedLogics
using DecisionTree
using FuzzyLogic
using Statistics

export ManyExpertDecisionTree

include("many-expert-decision-tree.jl")

export manify

include("manify.jl")

end
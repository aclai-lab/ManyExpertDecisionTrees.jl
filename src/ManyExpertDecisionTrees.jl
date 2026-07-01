module ManyExpertDecisionTrees

using SoleLogics
using DecisionTree
using Statistics

export SigmoidMF
include("membership_functions.jl")

export ManyExpertDecisionTree, depth
include("many-expert-decision-tree.jl")

export updateExpertHyperParameters
include("tuning.jl")

export manify, fuzzify

include("manify.jl")
include("utils.jl")

export apply

include("apply.jl")

export ConfusionMatrix, confusionmatrix, accuracy, recall, precision, vagueness
include("metrics.jl")

export montecarlocv 
include("cross_validation.jl")
end
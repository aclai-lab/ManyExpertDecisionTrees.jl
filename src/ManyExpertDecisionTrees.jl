module ManyExpertDecisionTrees

using SoleLogics
using DecisionTree
using Statistics

import FuzzyLogic as FL

export ManyExpertDecisionTree, depth

include("many-expert-decision-tree.jl")

export manify, fuzzify

include("manify.jl")
include("utils.jl")

export apply 

include("apply.jl")

export predict, ConfusionMatrix, confusion_matrix
include("metrics.jl")

end
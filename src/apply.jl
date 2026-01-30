using SoleLogics.ManyValuedLogics

"""
    function predict(
        X::AbstractMatrix{S},
        medt::ManyExpertDecisionTree{T, S},
        experts::FuzzyLogic...
    ) where {T, S}

Given a set of test instances, evaluate the subset of classes they could belong to
using the tnorms defined by the fuzzy logics
"""
function predict(
    X::AbstractMatrix{S},
    medt::ManyExpertDecisionTree{T, S},
    experts::FuzzyLogic...
) where {T, S}

    MXA = ManyExpertAlgebra(experts...)
    return [apply(medt, MXA, row) for row in eachrow(X)]
end

"""
    function predict(
        X::AbstractMatrix{S},
        medt::ManyExpertDecisionTree{T, S},
        algebra::ManyExpertAlgebra
    ) where {T, S}

Given a set of test instances, evaluate the subset of classes they could belong to
using the tnorms defined by the ManyExpertAlgebra
"""
function predict(
    X::AbstractMatrix{S},
    medt::ManyExpertDecisionTree{T, S},
    algebra::ManyExpertAlgebra
) where {T, S}

    return [apply(medt, algebra, row) for row in eachrow(X)]
end

"""
   function apply(
    tree::ManyExpertDecisionTree{T}, 
    MXA::ManyExpertAlgebra, 
    instance::AbstractVector{S}
) where {T, S}

Given an instance, evaluate its membership degree to each class using the tnorms 
defined by the ManyExpertAlgebra.  
"""
function apply(
    tree::ManyExpertDecisionTree{T}, 
    MXA::ManyExpertAlgebra, 
    instance::AbstractVector{S}
) where {T, S}
    length(tree.mftypes) == length(MXA.experts) || 
        error("Expert mismatch: expected $(length(tree.mftypes)) experts, got $(length(MXA.experts))")
    
    length(instance) == tree.nfeats ||
        error("Instance dimension mismatch: expected $(tree.nfeats) features, got $(length(instance))")
    
    candidates = Vector{Pair{T, NTuple{length(MXA.experts), ContinuousTruth}}}()
    evalsubtree(candidates, tree.root, MXA, instance, top(MXA))
        
    return unique(first.(candidates))
end

# Internal function used to evaluate a subtree recursively 
function evalsubtree(
    candidates::Vector{Pair{T, NTuple{N, ContinuousTruth}}}, 
    node::MEDTLeaf{T},
    MXA::ManyExpertAlgebra, 
    instance::AbstractVector{S}, 
    mmdg::NTuple{N, ContinuousTruth}
) where {T, N, S}
    
    # Check if the new candidate is dominated by any existing candidate
    @inbounds for i in 1:length(candidates)
        SoleLogics.precedes(MXA, mmdg, candidates[i][2]) && return
    end

    # Remove any existing candidates that are strictly dominated by the current one
    i = 1
    @inbounds while i <= length(candidates)
        if SoleLogics.precedes(MXA, candidates[i][2], mmdg)
            deleteat!(candidates, i)
        else
            i += 1
        end
    end

    push!(candidates, node.label => mmdg)
end

function evalsubtree(
    candidates::Vector{Pair{T, NTuple{N, ContinuousTruth}}}, 
    node::MEDTNode{T, S},
    MXA::ManyExpertAlgebra, 
    instance::AbstractVector{S}, 
    mmdg::NTuple{N, ContinuousTruth}
) where {T, N, S}

    feat_val = instance[node.featid]
    
    mmdgleft = ntuple(i -> ContinuousTruth(node.mfleft[i](feat_val)), N)
    evalsubtree(candidates, node.left, MXA, instance, SoleLogics.collatetruth(∧, (mmdg, mmdgleft), MXA))
    
    mmdgright = ntuple(i -> ContinuousTruth(node.mfright[i](feat_val)), N)
    evalsubtree(candidates, node.right, MXA, instance, SoleLogics.collatetruth(∧, (mmdg, mmdgright), MXA))
end
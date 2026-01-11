using SoleLogics.ManyValuedLogics

"""
    apply(tree::ManyExpertDecisionTree{T}, MXA::ManyExpertAlgebra, instance::AbstractVector{Float64}) where {T}

Given an instance, evaluate its membership degree to each class using the tnorms defined by the ManyExpertAlgebra.  
"""
function apply(tree::ManyExpertDecisionTree{T}, MXA::ManyExpertAlgebra, instance::AbstractVector{Float64}) where {T}
    length(tree.mftypes) == length(MXA.experts) || 
        error("Expert mismatch: expected $(length(tree.mftypes)) experts, got $(length(MXA.experts))")
    
    length(instance) == length(tree.featnames) ||
        error("Instance dimension mismatch: expected $(length(tree.featnames)) features, got $(length(instance))")
    
    candidates = Vector{Pair{T, NTuple{length(MXA.experts), ContinuousTruth}}}()
    evalsubtree(candidates, tree.root, MXA, instance, top(MXA))
    
    # As of this moment apply returns a specific candidate if he's the maximal value in the poset 
    # otherwise the function returns a subset of entries that are not "dominated" by other evaluations.
    # I have doubts about this and would like a second opinion. 
    
    return unique(first.(candidates))
end

# Internal function used to evaluate a subtree recursively 
function evalsubtree(candidates::Vector{Pair{T, NTuple{N, ContinuousTruth}}}, 
                      node::MEDTLeaf{T},
                      MXA::ManyExpertAlgebra, 
                      instance::AbstractVector{Float64}, 
                      mmdg::NTuple{N, ContinuousTruth}) where {T, N}
    
    # Check if the new candidate 'mmdg' is dominated by any existing candidate
    @inbounds for i in 1:length(candidates)
        SoleLogics.precedes(MXA, mmdg, candidates[i][2]) && return
    end

    # Remove any existing candidates that are strictly dominated by 'mmdg'
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

function evalsubtree(candidates::Vector{Pair{T, NTuple{N, ContinuousTruth}}}, 
                      node::MEDTNode{T}, 
                      MXA::ManyExpertAlgebra, 
                      instance::AbstractVector{Float64}, 
                      mmdg::NTuple{N, ContinuousTruth}) where {T, N}

    feat_val = instance[node.featid]
    
    # Evaluate left branch
    mmdgleft = ntuple(i -> ContinuousTruth(node.mfleft[i](feat_val)), N)
    evalsubtree(candidates, node.left, MXA, instance, SoleLogics.collatetruth(∧, (mmdg, mmdgleft), MXA))
    
    mmdgright = ntuple(i -> ContinuousTruth(node.mfright[i](feat_val)), N)
    evalsubtree(candidates, node.right, MXA, instance, SoleLogics.collatetruth(∧, (mmdg, mmdgright), MXA))
end
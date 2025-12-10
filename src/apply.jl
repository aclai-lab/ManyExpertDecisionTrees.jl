using SoleLogics.ManyValuedLogics

function apply(tree::ManyExpertDecisionTree{T}, MXA::ManyExpertAlgebra, instance::AbstractVector{Float64}) where {T}
    length(tree.mftypes) == length(MXA.experts) || 
        error("Expert mismatch: the number of experts in the Algebra doesn't match the expected number of experts")
   
    results = Dict{T, NTuple{length(MXA.experts), ContinuousTruth}}()
    evalsubtree!(results, tree.root, MXA, instance, top(MXA))
    return results
end

function evalsubtree!(results::Dict{T, NTuple{N, ContinuousTruth}}, 
                      node::Union{MEDTNode{T}, MEDTLeaf{T}}, 
                      MXA::ManyExpertAlgebra, 
                      instance::AbstractVector{Float64}, 
                      mmdg::NTuple{N, ContinuousTruth}) where {T, N}
    if node isa MEDTLeaf
        if haskey(results, node.label)
            results[node.label] = SoleLogics.collatetruth(∨, (results[node.label], mmdg), MXA)
        else
            results[node.label] = mmdg
        end
        return nothing
    end

    mmdgleft = ntuple(i -> ContinuousTruth(node.mfleft[i](instance[node.featid])), N)
    evalsubtree!(results, node.left, MXA, instance, SoleLogics.collatetruth(∧, (mmdg, mmdgleft), MXA))
    
    mmdgright = ntuple(i -> ContinuousTruth(node.mfright[i](instance[node.featid])), N)
    evalsubtree!(results, node.right, MXA, instance, SoleLogics.collatetruth(∧, (mmdg, mmdgright), MXA))
end
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
    
    results = Dict{T, NTuple{length(MXA.experts), ContinuousTruth}}()
    evalsubtree(results, tree.root, MXA, instance, top(MXA))
    
    candidates = Vector{T}()

    # As of this moment apply returns a specific candidate if he's the maximal value in the poset 
    # otherwise the function returns a subset of entries that are not "dominated" by other evaluations.
    # I have doubts about this and would like a second opinion. 

    for i in results
        is_dominated = false
        for j in results
            if i != j && SoleLogics.precedes(MXA, i[2], j[2]) && !SoleLogics.precedes(MXA, j[2], i[2])
                is_dominated = true
                break  
            end
        end

        if(!is_dominated)
            push!(candidates, i[1])
        end
    end
    
    if length(candidates) == 1
        return candidates[1]
    else
        return candidates
    end
end

# Internal function used to evaluate a subtree recursively 
function evalsubtree(results::Dict{T, NTuple{N, ContinuousTruth}}, 
                      node::Union{MEDTNode{T}, MEDTLeaf{T}}, 
                      MXA::ManyExpertAlgebra, 
                      instance::AbstractVector{Float64}, 
                      mmdg::NTuple{N, ContinuousTruth}) where {T, N}
    if node isa MEDTLeaf
        t = get!(results, node.label, mmdg)
        if (t != mmdg) results[node.label] = SoleLogics.collatetruth(∨, (results[node.label], mmdg), MXA) end
        return nothing
    end

    mmdgleft = ntuple(i -> ContinuousTruth(node.mfleft[i](instance[node.featid])), N)
    evalsubtree(results, node.left, MXA, instance, SoleLogics.collatetruth(∧, (mmdg, mmdgleft), MXA))
    
    mmdgright = ntuple(i -> ContinuousTruth(node.mfright[i](instance[node.featid])), N)
    evalsubtree(results, node.right, MXA, instance, SoleLogics.collatetruth(∧, (mmdg, mmdgright), MXA))
end
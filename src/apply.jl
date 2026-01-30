using SoleLogics.ManyValuedLogics

function predict(
    X::AbstractMatrix{S},
    medt::ManyExpertDecisionTree{T, S},
    experts::FuzzyLogic...;
    depth=-1
) where {S}

    MXA = ManyExpertAlgebra(experts...)
    return [apply(medt, MXA, row; depth) for row in eachrow(X)]
end

function predict(
    X::AbstractMatrix{S},
    medt::ManyExpertDecisionTree{T, S},
    algebra::ManyExpertAlgebra;
    depth=-1
) where {S}

    return [apply(medt, algebra, row; depth) for row in eachrow(X)]
end

function apply(
    tree::ManyExpertDecisionTree{T, S}, 
    expert::FuzzyLogic, 
    instance::AbstractVector; 
    depth=-1
) where {T, S}
    MXA = ManyExpertAlgebra(expert)
    apply(tree, MXA, instance; depth=depth)
end

function apply(
    tree::ManyExpertDecisionTree{T, S}, 
    experts::NTuple{N, FuzzyLogic}, 
    instance::AbstractVector; 
    depth=-1
) where {T, N, S}
    MXA = ManyExpertAlgebra(experts...)
    apply(tree, MXA, instance; depth=depth)
end

function apply(
    tree::ManyExpertDecisionTree{T, S}, 
    MXA::ManyExpertAlgebra, 
    instance::AbstractVector; 
    depth=-1
) where {T, S}
    
    (depth == -1 || depth > 0) ||
    error("Invalid depth: invalid depth value")

    length(tree.mftypes) == length(MXA.experts) || 
    error("Expert mismatch: expected $(length(tree.mftypes)) experts, got $(length(MXA.experts))")
    
    length(instance) == tree.nfeats ||
    error("Instance dimension mismatch: expected $(tree.nfeats) features, got $(length(instance))")

    N = length(MXA.experts)

    solutions = Vector{Pair{MEDTLeaf{T}, NTuple{N, ContinuousTruth}}}()
    queue = Vector{Pair{MEDTLeafOrNode{T, S}, NTuple{N, ContinuousTruth}}}()

    # Add the root to the queue
    pushfirst!(queue, tree.root => top(MXA))

    lvl = 0
    while !isempty(queue)
        
        # If i reached the desired depth, prune out the dominated branches
        if lvl == depth
            local_maxima = empty(queue)

            # I have to find a way to prune out undesired values
            for (node, mmdg) in queue 
                if node isa MEDTLeaf
                    pushpareto!(solutions, node => mmdg, MXA)
                    continue
                end

                pushpareto!(local_maxima, node => mmdg, MXA)
            end

            # Reset the queue
            queue = local_maxima

            # Reset the depth, so this step can be repeated
            lvl = 0
        end
        
        # All nodes in the queue while inside the outer loop are at the same level
        level_size = length(queue)

        # Repeat inner loop until all nodes at depth k have been explored
        while level_size != 0
            node, mmdg = pop!(queue)
            
            # If node is a leaf, add it to the solutions and skip to next node at that level
            if node isa MEDTLeaf
                pushpareto!(solutions, node => mmdg, MXA)

                level_size -= 1
                continue 
            end

            # If i've reached this point, the node isn't a leaf. 
            feat_val = instance[node.featid]

            # Compute local membership degrees for right and left children
            mmdgleft = ntuple(i -> ContinuousTruth(node.mfleft[i](feat_val)), N)
            mmdgright = ntuple(i -> ContinuousTruth(node.mfright[i](feat_val)), N)

            # Global membership degrees are the conjuction of father and children mmdgs
            mmdgleft = SoleLogics.collatetruth(∧, (mmdg, mmdgleft), MXA)
            mmdgright = SoleLogics.collatetruth(∧, (mmdg, mmdgright), MXA)

            # Add children nodes and their global mmdgs to queue
            pushfirst!(queue, node.left => mmdgleft, node.right => mmdgright)
            
            level_size -= 1
        end
        
        # If inner loop has finished, i went down 1 level
        lvl += 1
    end

    return unique([leaf.label for (leaf, _) in solutions])
end

function pushpareto!(
    solutions::Vector{Pair{MEDTLeaf{T}, NTuple{N, ContinuousTruth}}},
    node_mmdg::Pair{MEDTLeaf{T}, NTuple{N, ContinuousTruth}},
    MXA
) where {T, N}
    
    node, mmdg = node_mmdg
    
    # Check if the new solution is dominated by any existing solution
    @inbounds for i in 1:length(solutions)
        SoleLogics.precedes(MXA, mmdg, solutions[i][2]) && return
    end

    # Remove any existing solutions that are strictly dominated by the current one
    i = 1
    @inbounds while i <= length(solutions)
        if SoleLogics.precedes(MXA, solutions[i][2], mmdg)
            deleteat!(solutions, i)
        else
            i += 1
        end
    end

    push!(solutions, node_mmdg)
end

function pushpareto!(
    solutions::Vector{Pair{MEDTLeafOrNode{T, S}, NTuple{N, ContinuousTruth}}},
    node_mmdg::Pair{<:MEDTLeafOrNode{T, S}, NTuple{N, ContinuousTruth}},
    MXA
) where {T, S, N}
    
    node, mmdg = node_mmdg
    
    # Check if the new solution is dominated by any existing solution
    @inbounds for i in 1:length(solutions)
        SoleLogics.precedes(MXA, mmdg, solutions[i][2]) && return
    end

    # Remove any existing solutions that are strictly dominated by the current one
    i = 1
    @inbounds while i <= length(solutions)
        if SoleLogics.precedes(MXA, solutions[i][2], mmdg)
            deleteat!(solutions, i)
        else
            i += 1
        end
    end

    push!(solutions, node_mmdg)
end
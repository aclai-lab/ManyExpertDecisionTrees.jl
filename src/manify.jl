using DecisionTree
using SoleLogics
using SoleLogics.ManyValuedLogics
using DataFrames

function manify(dt::DecisionTree.Root, X::DataFrame, experts::NTuple{N, UnionAll}) where {N}
    expert_sets = subdivide(N, X)
    root = build_metree(dt.node, experts, expert_sets)

    return ManyExpertDecisionTree{N}(root, names(X), experts) 
end

function build_metree(node::Union{DecisionTree.Node, DecisionTree.Leaf}, experts::NTuple{N, UnionAll}, expert_sets::NTuple{N, SubDataFrame}) where {N}
    if(node isa DecisionTree.Leaf)
        return ME_Leaf(node.majority)
    end
    
    params = ntuple(N) do i 
        get_params(node.featval, node.featid, expert_sets[i], experts[i])
    end

    mem_l = ntuple(i -> experts[i](params[i][1]...), N)
    mem_r = ntuple(i -> experts[i](params[i][2]...), N)

    ME_Node(
        node.featval,
        node.featid,
        mem_l,
        mem_r,
        build_metree(node.left, experts, expert_sets),
        build_metree(node.right, experts, expert_sets)
    )
end
using DecisionTree
using SoleLogics
using SoleLogics.ManyValuedLogics
using DataFrames


"""
    manify(dt::DecisionTree.Root, X::DataFrame, experts::NTuple{N, UnionAll})

Convert a DecisionTree.jl decision tree into a ManyExpertDecisionTree by attaching N membership 
functions per node, parametrized from subdivisions of X. 
"""
function manify(dt::DecisionTree.Root, X::DataFrame, experts::NTuple{N, UnionAll})::ManyExpertDecisionTree{N} where {N}
    expertsdata = subdivide(N, X)
    root = build_medt(dt.node, experts, expertsdata)

    return ManyExpertDecisionTree{N}(root, names(X), experts) 
end

function build_medt(node::Union{DecisionTree.Node, DecisionTree.Leaf}, experts::NTuple{N, UnionAll}, expertsdata::NTuple{N, SubDataFrame}) where {N}
    if(node isa DecisionTree.Leaf)
        return MEDTLeaf(node.majority)
    end
    
    params = ntuple(N) do i 
        get_params(node.featval, node.featid, expertsdata[i], experts[i])
    end

    mem_l = ntuple(i -> experts[i](params[i][1]...), N)
    mem_r = ntuple(i -> experts[i](params[i][2]...), N)

    MEDTNode(
        node.featval,
        node.featid,
        mem_l,
        mem_r,
        build_medt(node.left, experts, expertsdata),
        build_medt(node.right, experts, expertsdata)
    )
end
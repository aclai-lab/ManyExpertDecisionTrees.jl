using DecisionTree
using DataFrames
import FuzzyLogic as FL

"""
    manify(dt::DecisionTree.Root, X::DataFrame, experts::NTuple{N, UnionAll})

Convert a DecisionTree.jl decision tree into a ManyExpertDecisionTree by attaching N membership 
functions per node, parametrized from subdivisions of X. 
"""
function manify(dt::DecisionTree.Root, X::DataFrame, experts::UnionAll...)::ManyExpertDecisionTree
    expertsdata = subdivide(length(experts), X)
    root = build_medt(dt.node, experts, expertsdata)

    return ManyExpertDecisionTree(root, names(X), experts...) 
end

function build_medt(node::Union{DecisionTree.Node, DecisionTree.Leaf}, experts::NTuple{N, UnionAll}, expertsdata::NTuple{N, SubDataFrame}) where {N}    
    if(node isa DecisionTree.Leaf)
        return MEDTLeaf(node.majority)
    end
    
    params = ntuple(N) do i 
        get_params(node.featval, node.featid, expertsdata[i], experts[i])
    end

    mfleft = FL.AbstractMembershipFunction[experts[i](params[i][1]...) for i in 1:N]
    mfright = FL.AbstractMembershipFunction[experts[i](params[i][2]...) for i in 1:N]
   
    MEDTNode(
        node.featval,
        node.featid,
        mfleft,
        mfright,
        build_medt(node.left, experts, expertsdata),
        build_medt(node.right, experts, expertsdata)
    )
end

function addexperts!(medt::ManyExpertDecisionTree, X::DataFrame, experts::UnionAll...)
    N = length(experts)
    expertsdata = subdivide(N, X)

    addmfs!(medt.root, experts, expertsdata)
    append!(medt.mftypes, [mf{Float64} for mf in experts])
end

function addmfs!(node::Union{MEDTLeaf, MEDTNode}, experts::NTuple{N, UnionAll}, expertsdata::NTuple{N, SubDataFrame}) where {N}
    if (node isa MEDTLeaf)
        return nothing
    end

    params = ntuple(N) do i 
        get_params(node.featval, node.featid, expertsdata[i], experts[i])
    end

    for i in 1:N
        push!(node.mfleft, experts[i](params[i][1]...))
        push!(node.mfright, experts[i](params[i][2]...))
    end

    addmfs!(node.left, experts, expertsdata)
    addmfs!(node.right, experts, expertsdata)
end
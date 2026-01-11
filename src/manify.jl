using DecisionTree
using DataFrames
import FuzzyLogic as FL

const CONSTANT_MF = FL.PiecewiseLinearMF([(0, 1)])

"""
    manify(dt::DecisionTree.Root, X::DataFrame, experts::UnionAll...)

Convert a DecisionTree.jl decision tree into a ManyExpertDecisionTree by attaching N membership 
functions per node, parameterized from subdivisions of X. 
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

    # If any of the parameters are NaN, default to the monoid's identity element
    # it doesn't add any information, but it doesn't take away any information either
    mfleft = Vector{FL.AbstractMembershipFunction}(undef, N)
    mfright = Vector{FL.AbstractMembershipFunction}(undef, N)
    
    @inbounds for i in 1:N
        mfleft[i] = any(isnan, params[i][1]) ? CONSTANT_MF : experts[i](params[i][1]...)
        mfright[i] = any(isnan, params[i][2]) ? CONSTANT_MF : experts[i](params[i][2]...)
    end
   
    MEDTNode(
        node.featval,
        node.featid,
        mfleft,
        mfright,
        build_medt(node.left, experts, expertsdata),
        build_medt(node.right, experts, expertsdata)
    )
end

"""
    addexperts!(medt::ManyExpertDecisionTree, X::DataFrame, experts::UnionAll...)

Add an arbitrary number of experts to the MEDT. Each expert's corresponding membership function will
be parameterized from a different subdivision of X. 
"""
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
        push!(node.mfleft, any(isnan, params[i][1]) ? CONSTANT_MF : experts[i](params[i][1]...))
        push!(node.mfright, any(isnan, params[i][2]) ? CONSTANT_MF : experts[i](params[i][2]...))
    end

    addmfs!(node.left, experts, expertsdata)
    addmfs!(node.right, experts, expertsdata)
end
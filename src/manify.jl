using DecisionTree
using DataFrames
import FuzzyLogic as FL

const CONSTANT_MF = FL.PiecewiseLinearMF([(0, 1)])

"""
    manify(dt::DecisionTree.Root, X::AbstractMatrix, experts::UnionAll...)

Convert a DecisionTree.jl decision tree into a ManyExpertDecisionTree by attaching N membership 
functions per node, parameterized from subdivisions of X. 
"""
function manify(dt::DecisionTree.Root, X::AbstractMatrix{S}, experts::UnionAll...)::ManyExpertDecisionTree where {S}
    size(X, 1) > 0 || throw(ArgumentError("X must have at least one row"))
    length(experts) > 0 || throw(ArgumentError("At least one expert must be provided"))
    
    expertsdata = subdivide(length(experts), X)
    root = build_medt(dt.node, experts, expertsdata)

    return ManyExpertDecisionTree(root, size(X, 2), experts...) 
end

"""
    fuzzify(dt::DecisionTree.Root, X::AbstractMatrix{S}, expert::UnionAll) 

Convert a DecisionTree.jl decision tree into a FuzzyTree, which is a ManyExpertDecisionTree with 
a single expert. 
"""
function fuzzify(dt::DecisionTree.Root, X::AbstractMatrix{S}, expert::UnionAll) where {S}
    return manify(dt, X, expert)
end


function build_medt(node::DecisionTree.Leaf, experts::NTuple{N, UnionAll}, expertsdata::NTuple{N, AbstractMatrix{S}}) where {N, S}
        return MEDTLeaf(node.majority)
end


function build_medt(node::DecisionTree.Node, experts::NTuple{N, UnionAll}, expertsdata::NTuple{N, AbstractMatrix{S}}) where {N, S}        
    
    expert_sets = ntuple(N) do i 
        split_set(node.featval, node.featid, expertsdata[i])
    end

    params = ntuple(N) do i
        l = get_params(node.featid, expert_sets[i][1], experts[i])
        r = get_params(node.featid, expert_sets[i][2], experts[i])
        l, r
    end

    mfleft = Vector{FL.AbstractMembershipFunction}(undef, N)
    mfright = Vector{FL.AbstractMembershipFunction}(undef, N)
    
    @inbounds for i in 1:N
        mfleft[i] = any(isnan, params[i][1]) ? CONSTANT_MF : experts[i](params[i][1]...)
        mfright[i] = any(isnan, params[i][2]) ? CONSTANT_MF : experts[i](params[i][2]...)
    end
   
    # Prepare left and right expert data for recursive calls
    left_expertsdata = ntuple(N) do i
        expert_sets[i][1]
    end

    right_expertsdata = ntuple(N) do i
        expert_sets[i][2]
    end
   
    MEDTNode(
        node.featval,
        node.featid,
        mfleft,
        mfright,
        build_medt(node.left, experts, left_expertsdata),
        build_medt(node.right, experts, right_expertsdata)
    )
end


# TODO: adapt add experts to the modifications, kinda low priority rn 

#= """
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
end =#
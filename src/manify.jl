using DecisionTree
using DataFrames
import FuzzyLogic as FL

"""
    manify(dt::DecisionTree.Root, X::AbstractMatrix, experts::UnionAll...)

Convert a DecisionTree.jl decision tree into a ManyExpertDecisionTree by attaching 
N membership functions per node, parameterized from subdivisions of X. 
"""
function manify(
    dt::DecisionTree.Root, 
    X::AbstractMatrix{S}, 
    experts::UnionAll...;
    kwargs...
)::ManyExpertDecisionTree where {S}

    size(X, 1) > 0 || throw(ArgumentError("X must have at least one row"))
    length(experts) > 0 || throw(ArgumentError("At least one expert must be provided"))

    expertsdata = subdivide(length(experts), X)
    root = build_medt(dt.node, experts, expertsdata; kwargs...)

    if root isa MEDTLeaf
        return ManyExpertDecisionTree{typeof(root.label), S}(root, size(X, 2), experts...)
    end

    return ManyExpertDecisionTree(root, size(X, 2), experts...)
end

"""
    fuzzify(dt::DecisionTree.Root, X::AbstractMatrix{S}, expert::UnionAll; kwargs...) 

Convert a DecisionTree.jl decision tree into a FuzzyTree, which is a ManyExpertDecisionTree with 
a single expert. 
"""
function fuzzify(
    dt::DecisionTree.Root,
    X::AbstractMatrix{S}, 
    expert::UnionAll; 
    kwargs...
) where {S}
    return manify(dt, X, expert; kwargs...)
end


function build_medt(
    node::DecisionTree.Leaf,
    experts::NTuple{N,UnionAll}, 
    expertsdata::NTuple{N,AbstractMatrix{S}};
    kwargs...
) where {N,S}

    return MEDTLeaf(node.majority)
end

function build_medt(
    node::DecisionTree.Node, 
    experts::NTuple{N,UnionAll}, 
    expertsdata::NTuple{N,AbstractMatrix{S}};
    kwargs...
) where {N,S}
    
    split_val = convert(Float64, node.featval)

    # Calculate MFs and get split datasets for each expert
    results = ntuple(i -> build_mfs(experts[i], node.featid, split_val, expertsdata[i]; kwargs...), N)

    # Unpack results
    mfleft = FL.AbstractMembershipFunction[r[1] for r in results]
    mfright = FL.AbstractMembershipFunction[r[2] for r in results]
    left_expertsdata = ntuple(i -> results[i][3], N)
    right_expertsdata = ntuple(i -> results[i][4], N)

    return MEDTNode(
        split_val, 
        node.featid, 
        mfleft, 
        mfright,
        build_medt(node.left, experts, left_expertsdata; kwargs...),
        build_medt(node.right, experts, right_expertsdata; kwargs...)
    )
end

# function build_medt(
#     node::DecisionTree.Node, 
#     experts::NTuple{N,UnionAll}, 
#     expertsdata::NTuple{N,AbstractMatrix{S}}
# ) where {N,S}

#     expert_sets = ntuple(N) do i
#         split_set(node.featval, node.featid, expertsdata[i])
#     end

#     mfleft = Vector{FL.AbstractMembershipFunction}(undef, N)
#     mfright = Vector{FL.AbstractMembershipFunction}(undef, N)

#     @inbounds for i in 1:N
#         split_val = convert(Float64, node.featval)
        
#         # Left branch
#         set_l = expert_sets[i][1]
#         if size(set_l, 1) < 15
#             mfleft[i] = FL.PiecewiseLinearMF([(split_val, 1.0), (split_val + 1e-5, 0.0)])
#         else
#             params_l = get_params(node.featid, set_l, experts[i])
#             mfleft[i] = any(isnan, params_l) ? CONSTANT_MF : experts[i](params_l...)
#         end

#         # Right branch
#         set_r = expert_sets[i][2]
#         if size(set_r, 1) < 15
#             mfright[i] = FL.PiecewiseLinearMF([(split_val, 0.0), (split_val + 1e-5, 1.0)])
#         else
#             params_r = get_params(node.featid, set_r, experts[i])
#             mfright[i] = any(isnan, params_r) ? CONSTANT_MF : experts[i](params_r...)
#         end
#     end

#     left_expertsdata = ntuple(N) do i
#         expert_sets[i][1]
#     end

#     right_expertsdata = ntuple(N) do i
#         expert_sets[i][2]
#     end

#     MEDTNode(
#         node.featval,
#         node.featid,
#         mfleft,
#         mfright,
#         build_medt(node.left, experts, left_expertsdata),
#         build_medt(node.right, experts, right_expertsdata)
#     )
# end


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
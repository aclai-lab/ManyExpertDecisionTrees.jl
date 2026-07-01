"""
    struct MEDTLeaf{T}
        label::T
    end

A simple leaf structure that stores just the label associated with itself.
"""
struct MEDTLeaf{T}
    label::T
end

"""
    struct MEDTNode{S, T}
        featval::S
        featid::Int
        left::Union{MEDTNode{S, T}, MEDTLeaf{T}}
        right::Union{MEDTNode{S, T}, MEDTLeaf{T}}
        mfleft::Vector{<:AbstractMembershipFunction}
        mfright::Vector{<:AbstractMembershipFunction}
    end

A node structure that stores information about the corresponding split, as well as references
to its child nodes and the N membership functions associated with its branches.
"""
struct MEDTNode{S, T}
    featval::S
    featid::Int
    left::Union{MEDTNode{S, T}, MEDTLeaf{T}}
    right::Union{MEDTNode{S, T}, MEDTLeaf{T}}
    mfleft::Vector{<:AbstractMembershipFunction}
    mfright::Vector{<:AbstractMembershipFunction}
end

# Union type for nodes 
const MEDTLeafOrNode{S, T} = Union{MEDTLeaf{T}, MEDTNode{S, T}}

"""
    struct ManyExpertDecisionTree{S, T}
        root::Union{MEDTNode{S, T}, MEDTLeaf{T}}
        nfeats::Int
        mftypes::Vector{UnionAll}
        mfparams::Vector{<:Base.RefValue{<:AbstractHyperParameters}}
    end

A MEDT is a DecisionTree-like structure that implements concepts from Many-Valued 
and Fuzzy Logics, such as membership functions and partial ordering of truth 
values. In a MEDT, classical crisp splitsare replaced by fuzzy splits, allowing 
partial membership of instances to multiple branches. The degree of membership of 
an instance to a branch is defined by the corresponding membership functions, each 
of which is associated with a different expert and parameterized on a different
subset of data.
"""
struct ManyExpertDecisionTree{S, T}
    root::MEDTLeafOrNode{S, T}
    nfeats::Int
    mftypes::Vector{UnionAll}
    mfparams::Vector{<:Base.RefValue{<:AbstractHyperParameters}}
end

Base.length(leaf::MEDTLeaf) = 1
Base.length(node::MEDTNode) = length(node.left) + length(node.right)
Base.length(tree::ManyExpertDecisionTree) = length(tree.root)

depth(leaf::MEDTLeaf) = 0
depth(node::MEDTNode) = 1 + max(depth(node.left), depth(node.right))
depth(tree::ManyExpertDecisionTree) = depth(tree.root)
    
function Base.show(io::IO, leaf::MEDTLeaf)
    print(io, "MEDTLeaf(label=$(leaf.label))")
end

function Base.show(io::IO, node::MEDTNode)
    print(io, "MEDTNode(featid=$(node.featid), featval=$(node.featval))")
end

function Base.show(io::IO, tree::ManyExpertDecisionTree)
    print(io, "ManyExpertDecisionTree(nfeats=$(tree.nfeats), experts=$(tree.mftypes), hyperparameters=$([p[] for p in tree.mfparams]))")
end
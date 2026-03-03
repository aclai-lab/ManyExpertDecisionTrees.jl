import FuzzyLogic as FL
using DecisionTree

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
        mfleft::Vector{FuzzyLogic.AbstractMembershipFunction}
        mfright::Vector{FuzzyLogic.AbstractMembershipFunction}
        left::Union{MEDTNode{S, T}, MEDTLeaf{T}}
        right::Union{MEDTNode{S, T}, MEDTLeaf{T}}
    end

A node structure that stores information about the corresponding split, as well as references
to its child nodes and the N membership functions associated with its branches.
"""
struct MEDTNode{S, T}
    featval::S
    featid::Int
    mfleft::Vector{FL.AbstractMembershipFunction}
    mfright::Vector{FL.AbstractMembershipFunction}
    left::Union{MEDTNode{S, T}, MEDTLeaf{T}}
    right::Union{MEDTNode{S, T}, MEDTLeaf{T}}
end

# Union type for nodes 
const MEDTLeafOrNode{S, T} = Union{MEDTLeaf{T}, MEDTNode{S, T}}

"""
    struct ManyExpertDecisionTree{S, T}
        root::Union{MEDTNode{S, T}, MEDTLeaf{T}}
        nfeats::Int
        mftypes::Vector{DataType}
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
    root::MEDTLeafOrNode{S, T} # I'd like to handle type ambiguity a bit better
    nfeats::Int
    mftypes::Vector{DataType}

    function ManyExpertDecisionTree{S, T}(
        root::MEDTLeafOrNode{S, T},
        nfeats::Int,
        mftypes::UnionAll...
        ) where {
            S, T
        } 

        for f in mftypes
            if !(f <: FL.AbstractMembershipFunction)
                error("Unsupported Membership Function: only functions defined in the FuzzyLogic package are currently supported")
            end
        end
        return new{S, T}(root, nfeats, [mftypes[i]{Float64} for i in 1:length(mftypes)])
    end

    function ManyExpertDecisionTree(
        root::MEDTLeafOrNode{S, T},
        nfeats::Int,
        mftypes::UnionAll...
        ) where {
            S, T
        } 

        return ManyExpertDecisionTree{S, T}(root, nfeats, mftypes...)
    end
    
end


Base.length(leaf::MEDTLeaf) = 1
Base.length(node::MEDTNode) = length(node.left) + length(node.right)
Base.length(tree::ManyExpertDecisionTree) = length(tree.root)

depth(leaf::MEDTLeaf) = 0
depth(node::MEDTNode) = 1 + max(depth(node.left), depth(node.right))
depth(tree::ManyExpertDecisionTree) = depth(tree.root)

function Base.show(io::IO, leaf::MEDTLeaf)
    println("Many-Expert Leaf")
    println("Label: $(leaf.label)")
end

function Base.show(io::IO, node::MEDTNode)
    println("Many-Expert DecisionTree Node")
    println("Feat ID: $(node.featid)")
    println("Split value: $(node.featval)")
    println("L Membership Functions: $(node.mfleft)")
    println("R Membership Functions: $(node.mfright)")
end

function Base.show(io::IO, tree::ManyExpertDecisionTree)
    println("Many-Expert DecisionTree Root")
    println("Experts: $(tree.mftypes)")
end
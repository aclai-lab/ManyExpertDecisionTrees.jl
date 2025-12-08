using FuzzyLogic
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
    struct MEDTNode{T}
        featval::Float64
        featid::Int
        mfleft::Vector{FuzzyLogic.AbstractMembershipFunction}
        mfright::Vector{FuzzyLogic.AbstractMembershipFunction}
        left::Union{MEDTNode{T}, MEDTLeaf{T}}
        right::Union{MEDTNode{T}, MEDTLeaf{T}}
    end

A node structure that stores information about the corresponding split, as well as references
to its child nodes and the N membership functions associated with its branches.
"""
struct MEDTNode{T}
    featval::Float64
    featid::Int
    mfleft::Vector{FuzzyLogic.AbstractMembershipFunction}
    mfright::Vector{FuzzyLogic.AbstractMembershipFunction}
    left::Union{MEDTNode{T}, MEDTLeaf{T}}
    right::Union{MEDTNode{T}, MEDTLeaf{T}}
end


"""
    struct ManyExpertDecisionTree{N, T}
        root::Union{MEDTNode{T}, MEDTLeaf{T}}
        featnames::Vector{String}
        mftypes::Vector{DataType}
    end

A MEDT is a DecisionTree-like structure that implements concepts from Many-Valued and Fuzzy Logics, 
such as membership functions and partial ordering of truth values. In a MEDT, classical crisp splits
are replaced by fuzzy splits, allowing partial membership of instances to multiple branches. 
The degree of membership of an instance to a branch is defined by the corresponding membership functions,
each of which is associated with a different expert and parameterized on a different subset of data.
"""
struct ManyExpertDecisionTree{T}
    root::Union{MEDTNode{T}, MEDTLeaf{T}}
    featnames::Vector{String}
    mftypes::Vector{DataType}

    function ManyExpertDecisionTree(
        root::Union{MEDTNode{T}, MEDTLeaf{T}},
        featnames::Vector{String},
        mftypes::UnionAll...
        ) where {
            T
        } 

        for f in mftypes
            if !(f <: FuzzyLogic.AbstractMembershipFunction)
                error("Unsupported Membership Function: only functions defined in the FuzzyLogic package are currently supported")
            end
        end
        return new{T}(root, featnames, [mftypes[i]{Float64} for i in 1:length(mftypes)])
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
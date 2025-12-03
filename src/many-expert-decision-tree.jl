using SoleLogics
using FuzzyLogic
using DecisionTree

struct ME_Leaf{T}
    label::T
end

struct ME_Node{N, T}
    featval::Float64
    feature_idx::Int
    mem_l::NTuple{N, FuzzyLogic.AbstractMembershipFunction}
    mem_r::NTuple{N, FuzzyLogic.AbstractMembershipFunction}
    left::Union{ME_Node{N, T}, ME_Leaf{T}}
    right::Union{ME_Node{N, T}, ME_Leaf{T}}
end

struct ManyExpertDecisionTree{N, T}
    root::Union{ME_Node{N, T}, ME_Leaf{T}}
    featnames::Vector{String}
    memfunc_t::NTuple{N, DataType}

    function ManyExpertDecisionTree{N}(root::Union{ME_Node{N, T}, ME_Leaf{T}},
                                       featnames::Vector{String},
                                       memfunc_t::NTuple{N, UnionAll}) where {N, T}
        for f in memfunc_t
            if !(f <: FuzzyLogic.AbstractMembershipFunction)
                error("Unsupported Membership Function: only functions defined in the FuzzyLogic package are currently supported")
            end
        end
        return new{N, T}(root, featnames, ntuple(i -> memfunc_t[i]{Float64}, N))
    end
end


length(leaf::ME_Leaf) = 1
length(node::ME_Node) = length(node.left) + length(node.right)
length(tree::ManyExpertDecisionTree) = length(tree.root)

depth(leaf::ME_Leaf) = 0
depth(node::ME_Node) = 1 + max(depth(node.left), depth(node.right))
depth(tree::ManyExpertDecisionTree) = depth(tree.root)

function Base.show(io::IO, leaf::ME_Leaf)
    println("Many-Expert Leaf")
    println("Label: $(leaf.label)")
end

function Base.show(io::IO, node::ME_Node)
    println("Many-Expert DecisionTree Node")
    println("Feat ID: $(node.feature_idx)")
    println("Split value: $(node.featval)")
    println("L Membership Functions: $(node.mem_l)")
    println("R Membership Functions: $(node.mem_r)")
end

function Base.show(io::IO, tree::ManyExpertDecisionTree)
    println("Many-Expert DecisionTree Root")
    println("Experts: $(tree.memfunc_t)")
end
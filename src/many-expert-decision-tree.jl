using SoleLogics
using FuzzyLogic
using DecisionTree

struct Leaf{T}
    label::T
end

struct Node{N, T}
    memfuncs::NTuple{N, FuzzyLogic.AbstractMembershipFunction}
    featval::Float64
    feature_idx::Int
    left::Union{Node{N, T}, Leaf{T}}
    right::Union{Node{N, T}, Leaf{T}}
end

struct ManyExpertDecisionTree{N, T}
    root::Union{Node{N, T}, Leaf{T}}
    featnames::Vector{String}
    memfunc_t::NTuple{N, DataType}

    function ManyExpertDecisionTree{N}(root::Union{Node{N, T}, Leaf{T}},
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


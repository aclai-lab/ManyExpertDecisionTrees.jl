using DataFrames
using Statistics
import FuzzyLogic as FL

"""
    subdivide(N, X::AbstractMatrix{S})

Simply subdivide a feature matrix X into N different feature matrices. If X's number of
rows is not divisible by N, the remainder is appended to the last one.
"""
function subdivide(N, X::AbstractMatrix{S}) where {S}
    N > 0 || throw(ArgumentError("N must be positive"))
    n_rows = size(X, 1)
    n_rows >= N || throw(ArgumentError("Matrix must have at least N rows"))
    s = div(n_rows, N)

    ntuple(N) do i
        if(i != N)
            X[ (i-1)*s + 1 : i*s, : ]
        else
            X[ (i-1)*s + 1 : end, :]
        end
    end
end


"""
    function split_set(featval::S, featid::Int, expert_set::AbstractMatrix{S}) where {S}

Given a value to split around, return the left and right resulting feature matrices.
"""
function split_set(featval::S, featid::Int, expert_set::AbstractMatrix{S}) where {S}
    mask = expert_set[:, featid] .<= featval
    return expert_set[mask, :], expert_set[.!mask, :]
end


"""
    get_params(featid::Int, expertdata::AbstractMatrix{S}, mem_func::Type{GaussianMF})

Given a split and the feature matrix related to an expert, return the mean and variance of the two subsets
defined by the split.
"""
function get_params(featid::Int, expertdata::AbstractMatrix{S}, mem_func::Type{FL.GaussianMF}) where {S}
    featcol = expertdata[:, featid]
    return mean(featcol), std(featcol)
end

function get_params(featid::Int, expert_set::AbstractMatrix{S}, mem_func::Type{<:FL.AbstractMembershipFunction}) where {S}
    error("Currently, only Gaussian parameterization is supported. Received: $(mem_func)")
end
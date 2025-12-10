using DataFrames
using Statistics
import FuzzyLogic as FL

"""
    subdivide(N, X::DataFrame)

Simply subdivide a DataFrame X into N different SubDataFrames. If X's number of
rows is not divisible by N, the remainder is appended to the last SubDataFrame.
"""
function subdivide(N, X::DataFrame)
    N > 0 || throw(ArgumentError("N must be positive"))
    n_rows = size(X, 1)
    n_rows >= N || throw(ArgumentError("DataFrame must have at least N rows"))
    s = div(n_rows, N)

    ntuple(N) do i
        if(i != N)
            @view X[ (i-1)*s + 1 : i*s, : ]
        else
            @view X[ (i-1)*s + 1 : end, :]
        end
    end
end

function get_params(featval::Float64, featid::Int, expert_set::SubDataFrame, mem_func::Type{<:FL.AbstractMembershipFunction})
    error("Currently, only Gaussian parametrization is supported")
end


"""
    get_params(featval::Float64, featid::Int, expertdata::SubDataFrame, mem_func::Type{GaussianMF})

Given a split and the SubDataFrame related to an expert, return the mean and variance of the two subsets
defined by the split.
"""
function get_params(featval::Float64, featid::Int, expertdata::SubDataFrame, mem_func::Type{FL.GaussianMF})
    l = filter(x -> x <= featval, expertdata[:, featid])
    r = filter(x -> x > featval, expertdata[:, featid])

    lp = mean(l), std(l)    # How do i handle limit cases such as empty sets? Should the predict function treat NaN values as a bot? 
    rp = mean(r), std(r)    # What about 1 element sized arrays? 
    
    return lp, rp 
end
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
    error("Currently, only Gaussian parameterization is supported")
end

"""
    get_params(featval::Float64, featid::Int, expertdata::SubDataFrame, mem_func::Type{GaussianMF})

Given a split and the SubDataFrame related to an expert, return the mean and variance of the two subsets
defined by the split.
"""
function get_params(featval::Float64, featid::Int, expertdata::SubDataFrame, mem_func::Type{FL.GaussianMF})
    n_l, sum_l, sumsq_l = 0, 0.0, 0.0
    n_r, sum_r, sumsq_r = 0, 0.0, 0.0
    
    data_column = expertdata[:, featid]
    for x in data_column
        if x <= featval
            n_l += 1
            sum_l += x
            sumsq_l += x^2
        else
            n_r += 1
            sum_r += x
            sumsq_r += x^2
        end
    end

    lp = (sum_l / n_l), sqrt((sumsq_l - (sum_l^2)/n_l) / (n_l - 1))
    rp = (sum_r / n_r), sqrt((sumsq_r - (sum_r^2)/n_r) / (n_r - 1))
    
    return lp, rp 


    # l = filter(x -> x <= featval, expertdata[:, featid])
    # r = filter(x -> x > featval, expertdata[:, featid])

    # lp = mean(l), std(l) 
    # rp = mean(r), std(r)    
    
   # return lp, rp 
end
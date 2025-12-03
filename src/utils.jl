using DataFrames
using Statistics

function subdivide(N, X::DataFrame)
    n_rows = size(X, 1)
    s = div(n_rows, N)

    ntuple(N) do i
        if(i != N)
            @view X[ (i-1)*s + 1 : i*s, : ]
        else
            @view X[ (i-1)*s + 1 : end, :]
        end
    end
end

function get_params(featval::Float64, featid::Int, expert_set::SubDataFrame, mem_func::FuzzyLogic.AbstractMembershipFunction)
    error("Currently, only Gaussian parametrization is supported")
end

function get_params(featval::Float64, featid::Int, expert_set::SubDataFrame, mem_func::Type{GaussianMF})
    l = filter(x -> x <= featval, expert_set[:, featid])
    r = filter(x -> x > featval, expert_set[:, featid])

    lp = mean(l), std(l)
    rp = mean(r), std(r)
    
    return lp, rp 
end
using DataFrames
using Statistics
import FuzzyLogic as FL

const CONSTANT_MF = FL.PiecewiseLinearMF([(0, 1)])

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
    function split_set(featval::Real, featid::Int, expert_set::AbstractMatrix)

Given a value to split around, return the left and right resulting feature matrices.
"""
function split_set(featval::Real, featid::Int, expert_set::AbstractMatrix)
    mask = expert_set[:, featid] .<= featval
    return expert_set[mask, :], expert_set[.!mask, :]
end

function build_mfs(
    mem_func::Type{FL.GaussianMF}, 
    featid::Int, 
    feat_val::Real, 
    expertdata::AbstractMatrix; 
    kwargs...
)
    leftset, rightset = split_set(feat_val, featid, expertdata)
    split_val_fl = convert(Float64, feat_val)

    # Left Branch 
    mf_l = if size(leftset, 1) < 15
        FL.PiecewiseLinearMF([(split_val_fl, 1.0), (split_val_fl + 1e-5, 0.0)])
    else
        col = leftset[:, featid]
        mu, sigma = mean(col), std(col)
        if isnan(mu) || isnan(sigma)
            CONSTANT_MF
        else
            FL.GaussianMF(mu, sigma)
        end
    end

    # Right Branch
    mf_r = if size(rightset, 1) < 15
        FL.PiecewiseLinearMF([(split_val_fl, 0.0), (split_val_fl + 1e-5, 1.0)])
    else
        col = rightset[:, featid]
        mu, sigma = mean(col), std(col)
        if isnan(mu) || isnan(sigma)
            CONSTANT_MF
        else
            FL.GaussianMF(mu, sigma)
        end
    end

    return mf_l, mf_r, leftset, rightset
end

function build_mfs(
    mem_func::Type{FL.SigmoidMF},
    featid::Int,
    feat_val::Real,
    expertdata::AbstractMatrix; 
    slope::Union{Real, Nothing} = nothing, 
    slope_scaling::Float64 = 1.0, 
    kwargs...
)
    leftset, rightset = split_set(feat_val, featid, expertdata)
    split_val = convert(Float64, feat_val)
    
    # Left Branch
    mf_l = if !isnothing(slope)
        FL.SigmoidMF(-abs(slope), split_val)
    else
        col = leftset[:, featid]
        mu, sigma = mean(col), std(col)
        s = -slope_scaling / (sigma / mu)
        if isnan(s) || isinf(s)
            CONSTANT_MF
        else
            FL.SigmoidMF(s, split_val)
        end
    end

    # Right Branch
    mf_r = if !isnothing(slope)
        FL.SigmoidMF(abs(slope), split_val)
    else
        col = rightset[:, featid]
        mu, sigma = mean(col), std(col)
        s = slope_scaling / (sigma / mu)
        if isnan(s) || isinf(s)
            CONSTANT_MF
        else
            FL.SigmoidMF(s, split_val)
        end
    end

    return mf_l, mf_r, leftset, rightset
end


function build_mfs(mem_func::Type{<:FL.AbstractMembershipFunction}, args...; kwargs...) 
    error("Currently, only Gaussian parameterization is supported. Received: $(mem_func)")
end
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

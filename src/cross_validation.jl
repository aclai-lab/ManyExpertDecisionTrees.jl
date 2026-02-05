using Random
using Statistics
using SoleLogics.ManyValuedLogics
using Base.Threads
using ProgressBars
using PrettyTables
using Printf
import MLJ: partition
import DecisionTree as DT
import FuzzyLogic as FL

const metrics_dict = Dict(
    :accuracy => accuracy,
    :recall => recall,
    :precision => precision,
    :vagueness => vagueness
)

struct CVResults{T}
    n_folds::Int 
    
    row_labels::Vector{Union{T, Symbol}} 
    metrics::Vector{Symbol}              

    fuzzy::Matrix{NamedTuple{(:mean, :std), Tuple{Float64, Float64}}}
    crisp::Matrix{NamedTuple{(:mean, :std), Tuple{Float64, Float64}}}

    raw_folds::Vector{NamedTuple{(:fuzzy, :crisp), Tuple{ConfusionMatrix{T}, ConfusionMatrix{T}}}}
end

function Base.show(io::IO, res::CVResults)
    println(io, "Cross Validation Results (n_folds=$(res.n_folds))")
    
    function format_matrix(data)
        map(data) do v
            if v isa NamedTuple && haskey(v, :mean) && haskey(v, :std)
                isnan(v.mean) ? "NaN" : @sprintf("%.3f ± %.3f", v.mean, v.std)
            else
                string(v)
            end
        end
    end

    header = String.(res.metrics)
    
    println(io, "\n--- Crisp Results ---")
    pretty_table(
        io,
        format_matrix(res.crisp);
        column_labels=res.metrics,
        row_labels=res.row_labels,
        alignment=:c
    )
    
    println(io, "\n--- Fuzzy Results ---")
    pretty_table(
        io,
        format_matrix(res.fuzzy);
        column_labels=res.metrics,
        row_labels=res.row_labels,
        alignment=:c
    )
end

function montecarlocv(
    X::AbstractMatrix{S},
    y::AbstractVector{T},
    expert::FuzzyLogic,
    metrics::AbstractVector{Symbol};
    n_splits::Int=50,
    test_size::Float64=0.2,
    rng::Union{Int, AbstractRNG}=Random.GLOBAL_RNG
) where {S, T}

    class_names = sort(unique(y))
    row_labels = Vector{Union{T, Symbol}}(class_names)
    push!(row_labels, :Macro_Average)
    
    n_rows = length(row_labels)
    n_cols = length(metrics)
    train_size = 1 - test_size

    fuzzy_scores = Array{Float64}(undef, n_splits, n_rows, n_cols)
    crisp_scores = Array{Float64}(undef, n_splits, n_rows, n_cols)
    
    raw_folds = Vector{NamedTuple{(:fuzzy, :crisp), Tuple{ConfusionMatrix{T}, ConfusionMatrix{T}}}}(undef, n_splits)

    Threads.@threads for i in ProgressBar(1:n_splits)

        fold_rng = if rng isa Integer
            Random.MersenneTwister(rng + i)
        else
            Random.MersenneTwister(rand(rng, UInt32) + i)
        end

        X_train, y_train, X_test, y_test = begin
            train, test = partition(eachindex(y), train_size, shuffle=true, rng=fold_rng)
            X_train, y_train = X[train, :], y[train]
            X_test, y_test = X[test, :], y[test]
            X_train, y_train, X_test, y_test
        end

        cdt = DT.build_tree(y_train, X_train, 0, -1, 5; rng=fold_rng)
        fdt = fuzzify(cdt, X_train, FL.GaussianMF)

        cy_pred = DT.apply_tree(cdt, X_test)
        fy_pred = apply(fdt, expert, X_test)

        crisp_cm = convert(ConfusionMatrix{eltype(y_test)}, DT.confusion_matrix(y_test, cy_pred))
        fuzzy_cm = confusionmatrix(y_test, fy_pred)
        
        raw_folds[i] = (fuzzy=fuzzy_cm, crisp=crisp_cm)

        for (c_idx, metric) in enumerate(metrics)
            if !haskey(metrics_dict, metric)
                @warn "Metric $metric not found"
                continue
            end
            func = metrics_dict[metric]

            for (r_idx, cls) in enumerate(class_names)
                crisp_scores[i, r_idx, c_idx] = func(crisp_cm, cls)
                fuzzy_scores[i, r_idx, c_idx] = func(fuzzy_cm, cls)
            end

            crisp_scores[i, n_rows, c_idx] = func(crisp_cm)
            fuzzy_scores[i, n_rows, c_idx] = func(fuzzy_cm)
        end
    end
    
    function collapse_tensor(tensor)
        means = dropdims(mean(tensor, dims=1), dims=1)
        stds  = dropdims(std(tensor, dims=1), dims=1)
        return [ (mean=means[r,c], std=stds[r,c]) for r in 1:n_rows, c in 1:n_cols ]
    end

    return CVResults(
        n_splits,
        row_labels,
        Vector{Symbol}(metrics),
        collapse_tensor(fuzzy_scores),
        collapse_tensor(crisp_scores),
        raw_folds
    )
end
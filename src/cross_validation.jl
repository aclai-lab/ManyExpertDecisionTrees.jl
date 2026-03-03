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
    mftype::Type{<:FL.AbstractMembershipFunction},
    metrics::AbstractVector{Symbol};
    n_splits::Int=50,
    apply_depth::Int=-1,
    test_size::Float64=0.3,
    #Crisp tree specific args#
    n_subfeatures=0,
    max_depth=-1,
    min_samples_leaf=1,
    min_samples_split=2,
    min_purity_increase=0.0,
    ##########################
    rng::Union{Int, AbstractRNG}=Random.GLOBAL_RNG,
    kwargs...
) where {S, T}

    class_names = sort(unique(y))
    row_labels = Vector{Union{T, Symbol}}(class_names)
    push!(row_labels, :Macro_Average)
    
    n_rows = length(row_labels)
    n_cols = length(metrics)
    train_size = 1.0 - test_size

    fuzzy_scores = Array{Float64}(undef, n_splits, n_rows, n_cols)
    crisp_scores = Array{Float64}(undef, n_splits, n_rows, n_cols)
    
    raw_folds = Vector{NamedTuple{(:fuzzy, :crisp), Tuple{ConfusionMatrix{T}, ConfusionMatrix{T}}}}(undef, n_splits)

    seeds = if rng isa Integer
        [rng + i for i in 1:n_splits]
    else
        [rand(rng, UInt32) + UInt32(i) for i in 1:n_splits]
    end

    Threads.@threads for i in ProgressBar(1:n_splits)
        fold_rng = Random.MersenneTwister(seeds[i])

        X_train, y_train, X_test, y_test = begin
            train, test = partition(eachindex(y), train_size, shuffle=true, rng=fold_rng)
            X_train, y_train = X[train, :], y[train]
            X_test, y_test = X[test, :], y[test]
            X_train, y_train, X_test, y_test
        end

        cdt = DT.build_tree(
            y_train, 
            X_train, 
            n_subfeatures, 
            max_depth, 
            min_samples_leaf, 
            min_samples_split, 
            min_purity_increase; 
            rng=fold_rng
        )

        fdt = fuzzify(cdt, X_train, mftype; kwargs...)

        cy_pred = DT.apply_tree(cdt, X_test)
        fy_pred = apply(fdt, expert, X_test; depth=apply_depth)

        crisp_cm = convert(ConfusionMatrix{eltype(y_test)}, DT.confusion_matrix(y_test, cy_pred); classes=class_names)
        fuzzy_cm = confusionmatrix(y_test, fy_pred; classes=class_names)
        
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

            # Macro average derived from per-class scores for consistency
            crisp_scores[i, n_rows, c_idx] = mean(crisp_scores[i, r, c_idx] for r in 1:(n_rows - 1))
            fuzzy_scores[i, n_rows, c_idx] = mean(fuzzy_scores[i, r, c_idx] for r in 1:(n_rows - 1))
        end
    end
    
    function collapse_tensor(tensor)
        function safestats(x)
            valid = filter(!isnan, x)
            isempty(valid) && return (mean=NaN, std=NaN)
            m = mean(valid)
            s = length(valid) > 1 ? std(valid) : 0.0
            return (mean=m, std=s)
        end

        return [safestats(tensor[:, r, c]) for r in 1:n_rows, c in 1:n_cols]
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
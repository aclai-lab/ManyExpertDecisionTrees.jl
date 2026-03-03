using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, ".."))

using ManyExpertDecisionTrees
using DataFrames
using Random
using CSV
using SoleLogics.ManyValuedLogics
using PrettyTables
using Printf
import FuzzyLogic as FL

formats = Dict(
    :latex => ".tex",
    :text => ".txt",
    :html => ".html"
)



function runexps(
    outputname::String,
    metrics::Vector{Symbol}, 
    mftype::Type{<:FL.AbstractMembershipFunction}; 
    format::Symbol=:text, 
    kwargs...
)
    format ∉ keys(formats) && return println("invalid output format")

    data_dir = joinpath(@__DIR__, "datasets/processed/")
    datasets = filter(endswith(".csv"), readdir(data_dir))
    isempty(datasets) && return println("No datasets found in $data_dir")

    results_dir = joinpath(@__DIR__, "results/")
    mkpath(results_dir)
    results_path = joinpath(results_dir, outputname * formats[format])

    n_metrics = length(metrics)

    logics = [
        "Godel" => GodelLogic,
        "Product" => ProductLogic,
        "Lukasiewicz" => LukasiewiczLogic,
    ]

    n_logic_groups = length(logics) + 1

    println("Starting Experiments...")

    row_labels = String[]
    row_group_labels = Pair{Int, String}[]


    raw_data = Matrix{NamedTuple{(:mean, :std), Tuple{Float64, Float64}}}[]

    current_row = 1
    for dataset in datasets
        dataset_name = replace(dataset, ".csv" => "")
        println("> Processing $dataset_name...")

        df = DataFrame(CSV.File(joinpath(data_dir, dataset)))
        X = Matrix(df[:, 1:end-1])
        y = df[:, end]

        # Montecarlo cv for all experts
        results = map(logics) do (name, logic)
            println("   - Evaluating $name...")
            montecarlocv(X, y, logic, mftype, metrics; kwargs...)
        end

        # Push raw results to results table (exclude vagueness from crisp)
        crisp_col_indices = findall(m -> m != :vagueness, metrics)
        raw_block = hcat(results[1].crisp[:, crisp_col_indices], [r.fuzzy for r in results]...)
        push!(raw_data, raw_block)

        # Fill row labels and set row group labels for each dataset
        labels = string.(results[1].row_labels)
        append!(row_labels, labels)
        push!(row_group_labels, current_row => dataset_name)

        current_row += length(labels)
    end

    println("\n", "="^40, "\nFINAL RESULTS\n", "="^40)

    crisp_metrics = filter(m -> m != :vagueness, metrics)
    n_crisp_metrics = length(crisp_metrics)

    header_top = [
        n_crisp_metrics > 1 ? MultiColumn(n_crisp_metrics, "Crisp") : "Crisp",
        [n_metrics > 1 ? MultiColumn(n_metrics, name) : name for (name, _) in logics]...]
    header_bot = vcat(String.(crisp_metrics), repeat(String.(metrics), length(logics)))

    # Map each column to its metric (for highlighting)
    col_metric_map = vcat(crisp_metrics, repeat(metrics, length(logics)))
    
    raw_data = reduce(vcat, raw_data)

    # Format cell values depending on output format
    function format_val(x)
        s = @sprintf("%.2f", x)
        s == "1.00" && return "1"
        s == "0.00" && return "0"
        startswith(s, "0.") && return s[2:end]
        return s
    end

    fmt = if format == :latex
        v -> isnan(v.mean) ? LatexCell("NaN") : LatexCell("\$$(format_val(v.mean)) \\pm $(format_val(v.std))\$")
    else
        v -> isnan(v.mean) ? "NaN" : "$(format_val(v.mean)) ± $(format_val(v.std))"
    end

    table_data = fmt.(raw_data)

    rawmean(v) = v.mean
    raw_means = rawmean.(raw_data)

    function is_best(data, i, j)
        metric = col_metric_map[j]
        metric == :vagueness && return false
        same_metric_cols = findall(==(metric), col_metric_map)
        row_vals = [data[i, c] for c in same_metric_cols]
        best_val = maximum(filter(!isnan, row_vals); init=-Inf)
        return data[i, j] == best_val && !isnan(data[i, j])
    end

    # Build backend and highlighter based on format
    if format == :text
        hl = TextHighlighter(
            (data, i, j) -> is_best(raw_means, i, j),
            bold = true,
            foreground = :green
        )

        kwargs = (
            column_labels = [header_top, header_bot],
            merge_column_label_cells = :auto,
            display_size = (-1, -1),
            row_labels = row_labels,
            row_group_labels = row_group_labels,
            alignment = :c,
            highlighters = [hl,],
        )

        pretty_table(table_data; kwargs...)

        open(results_path, "w") do io
            pretty_table(io, table_data; kwargs...)
        end

    elseif format == :latex
        hl = LatexHighlighter(
            (data, i, j) -> is_best(raw_means, i, j),
            ["textbf"]
        )

        kwargs = (
            backend = :latex,
            column_labels = [header_top, header_bot],
            merge_column_label_cells = :auto,
            row_labels = row_labels,
            row_group_labels = row_group_labels,
            alignment = :c,
            highlighters = [hl,],
        )

        pretty_table(table_data; kwargs...)

        open(results_path, "w") do io
            pretty_table(io, table_data; kwargs...)
        end

    elseif format == :html
        hl = HtmlHighlighter(
            (data, i, j) -> is_best(raw_means, i, j),
            HtmlDecoration(font_weight = "bold", color = "green")
        )

        kwargs = (
            backend = :html,
            column_labels = [header_top, header_bot],
            merge_column_label_cells = :auto,
            row_labels = row_labels,
            row_group_labels = row_group_labels,
            alignment = :c,
            highlighters = [hl,],
        )

        pretty_table(table_data; kwargs...)

        open(results_path, "w") do io
            pretty_table(io, table_data; kwargs...)
        end
    end

    println("\nSaved table to $results_path")
end



runexps(
    "results_table", 
    [:accuracy, :recall, :precision], 
    FL.SigmoidMF; slope=2.5, 
    test_size=0.5,
    #Crisp tree specific args#
    min_samples_leaf=5,
    rng=69)

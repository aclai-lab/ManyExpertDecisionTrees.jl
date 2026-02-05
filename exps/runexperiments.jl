using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, ".."))

using ManyExpertDecisionTrees
using DataFrames
using CSV
using SoleLogics.ManyValuedLogics
using PrettyTables
using Printf

metrics = [:accuracy, :precision, :recall, :vagueness]

fmt(v) = isnan(v.mean) ? "NaN" : @sprintf("%.3f ± %.3f", v.mean, v.std)

function main()
    data_dir = joinpath(@__DIR__, "datasets")
    datasets = filter(endswith(".csv"), readdir(data_dir))
    isempty(datasets) && return println("No datasets found in $data_dir")

    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    results_path = joinpath(results_dir, "results_table.txt")

    n_metrics = length(metrics)

    logics = [
        "Godel"        => GodelLogic,
        "Product"      => ProductLogic,
        "Lukasiewicz"  => LukasiewiczLogic,
    ]

    println("Starting Experiments...")

    row_labels = String[]
    row_group_labels = Pair{Int,String}[]
    data_blocks = Matrix{String}[]
    current_row = 1

    for ds_file in datasets
        ds_name = replace(ds_file, ".csv" => "")
        println("> Processing $ds_name...")

        df = CSV.read(joinpath(data_dir, ds_file), DataFrame)
        X = Matrix(df[:, 1:end-1])
        y = df[:, end]

        results = map(logics) do (name, logic)
            println("  - Running $name...")
            montecarlocv(X, y, logic, metrics; n_splits=50, rng=1)
        end

        block = hcat((fmt.(r.fuzzy) for r in results)..., fmt.(results[1].crisp))
        push!(data_blocks, block)

        labels = string.(results[1].row_labels)
        append!(row_labels, labels)
        push!(row_group_labels, current_row => ds_name)
        current_row += length(labels)
    end

    println("\n", "="^40, "\nFINAL RESULTS\n", "="^40)

    header_top = [
        [MultiColumn(n_metrics, name) for (name, _) in logics]...,
        MultiColumn(n_metrics, "Crisp"),
    ]
    header_bot = repeat(String.(metrics), length(logics) + 1)

    table_data = reduce(vcat, data_blocks)
    pretty_table(
        table_data;
        column_labels = [header_top, header_bot],
        merge_column_label_cells = :auto,
        row_labels,
        row_group_labels,
        alignment = :c,
    )

    open(results_path, "w") do io
        pretty_table(
            io,
            table_data;
            column_labels = [header_top, header_bot],
            merge_column_label_cells = :auto,
            row_labels,
            row_group_labels,
            alignment = :c,
            display_size = (-1, -1),
        )
    end

    println("\nSaved table to $results_path")
end

main()
using ManyExpertDecisionTrees
using SoleLogics.ManyValuedLogics
using DataFrames
using DecisionTree
using Random
using MLJ
using FuzzyLogic
using Statistics

X, y = begin 
    X, y = @load_iris
    X = DataFrame(X)
    X, y
end

n_runs = 100
seeds = 1:n_runs
logic_experts = GodelLogic, ProductLogic
mf_experts = ntuple(_ -> GaussianMF, length(logic_experts))

medt_correct_percentages = Float64[]
medt_wrong_percentages = Float64[]
medt_vague_percentages = Float64[]

dt_correct_percentages = Float64[]
dt_wrong_percentages = Float64[]

for seed in seeds
    println("Running experiment with seed: $seed")
    
    X_train, y_train, X_test, y_test = begin
        train, test = partition(eachindex(y), 0.8, shuffle=true, rng = Random.MersenneTwister(seed));
        X_train, y_train = X[train, :], String.(y[train]);
        X_test, y_test = X[test, :], String.(y[test]);
        X_train, y_train, X_test, y_test
    end;

    dt = build_tree(y_train, Matrix(X_train))
    dt = prune_tree(dt, 0.9)

    medt = manify(dt, X_train, mf_experts...)

    MXA = ManyExpertAlgebra(logic_experts...)

    y_pred = map(eachrow(X_test)) do row 
        result = apply(medt, MXA, Vector{Float64}(row))
        
        if result isa Base.KeySet || result isa AbstractVector
            return "vague"
        else
            return result
        end
    end

    n_total = length(y_test)
    n_vague = count(==("vague"), y_pred)
    n_correct = count(i -> y_pred[i] == y_test[i], 1:n_total)
    n_wrong = n_total - n_correct - n_vague

    push!(medt_correct_percentages, n_correct/n_total*100)
    push!(medt_wrong_percentages, n_wrong/n_total*100)
    push!(medt_vague_percentages, n_vague/n_total*100)

    y_pred_dt = apply_tree(dt, Matrix(X_test))
    n_correct_dt = count(y_pred_dt .== y_test)
    n_wrong_dt = n_total - n_correct_dt
    
    push!(dt_correct_percentages, n_correct_dt/n_total*100)
    push!(dt_wrong_percentages, n_wrong_dt/n_total*100)
end

println("\n=== Average Results over $n_runs runs ===")

println("\n--- Many Expert Decision Tree ---")
println("Correct: $(round(mean(medt_correct_percentages), digits=2))% ± $(round(std(medt_correct_percentages), digits=2))")
println("Wrong: $(round(mean(medt_wrong_percentages), digits=2))% ± $(round(std(medt_wrong_percentages), digits=2))")
println("Vague: $(round(mean(medt_vague_percentages), digits=2))% ± $(round(std(medt_vague_percentages), digits=2))")

println("\n--- Classic Decision Tree ---")
println("Correct: $(round(mean(dt_correct_percentages), digits=2))% ± $(round(std(dt_correct_percentages), digits=2))")
println("Wrong: $(round(mean(dt_wrong_percentages), digits=2))% ± $(round(std(dt_wrong_percentages), digits=2))") 

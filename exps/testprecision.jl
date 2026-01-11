using Pkg

Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, ".."))

if !("results" in readdir(@__DIR__))
    result_dir = mkdir(joinpath(@__DIR__, "results"))
else
    result_dir = joinpath(@__DIR__, "results")
end

#Pkg.add("DataFrames")
#Pkg.add("CSV")
#Pkg.add("Combinatorics")
#Pkg.add("SoleLogics")
#Pkg.add("MLJ")
#Pkg.add("DecisionTree")
#Pkg.add("FuzzyLogic")

using ManyExpertDecisionTrees
using DataFrames
using CSV
using Combinatorics
using SoleLogics.ManyValuedLogics
using MLJ
using DecisionTree
import FuzzyLogic as FL
using Profile

function main()
    n_runs = 10  
    allexperts = (GodelLogic, LukasiewiczLogic, ProductLogic)

    # Compute all possible expert compbinations (with replacement)
    expertcomb = begin
        c = Vector{Vector{FuzzyLogic}}()
        for i in 1:length(allexperts)
            append!(c, collect(Combinatorics.with_replacement_combinations(allexperts, i)))
        end
        c
    end


    # Doing this, otherwise results are unreadable 
    expertcombreadable = map(expertcomb) do experts
        result = ""
        for expert in experts
            if (expert === GodelLogic)
                result *= "G"
            end
            if (expert === LukasiewiczLogic)
                result *= "L"
            end
            if (expert === ProductLogic)
                result *= "P"
            end
        end

        return result
    end

    data_dir = joinpath(@__DIR__, "datasets/")
    
    MXAs = [ManyExpertAlgebra(experts...) for experts in expertcomb]

    # Loop over each dataset in the subfolder to compute metrics
    for dataset in readdir(joinpath(data_dir))
        println("--- Evaluating precision on $(dataset) ---")

        correct = [[0.0, 0.0] for _ in 1:length(expertcomb)]
        wrong = [[0.0, 0.0] for _ in 1:length(expertcomb)]
        vague = [[0.0, 0.0] for _ in 1:length(expertcomb)]

        X, y = begin
            df = DataFrame(CSV.File(joinpath(data_dir, dataset)))
            X = df[:, 1:end-1]
            y = df[:, size(df, 2)]
            X, y
        end

        for i in 1:n_runs
            # Partition set into training and validation
            X_train, y_train, X_test, y_test = begin
                train, test = partition(eachindex(y), 0.8, shuffle=true, rng=i)
                X_train, y_train = X[train, :], y[train]
                X_test, y_test = X[test, :], y[test]
                X_train, y_train, X_test, y_test
            end

            # Build a standard decision tree
            dt = build_tree(y_train, Matrix(X_train))
            dt = prune_tree(dt, 0.9)

            X_test_matrix = Matrix(X_test)

            # For each expert combination, build a ManyExpertDecisionTree 
            Threads.@threads for k in eachindex(expertcomb)
                mf_experts = ntuple(_ -> FL.GaussianMF, length(expertcomb[k]))
                MXA = MXAs[k]  

                medt = manify(dt, X_train, mf_experts...)

                y_pred = map(eachrow(X_test_matrix)) do row
                    result = apply(medt, MXA, row)
                    return length(result) != 1 ? :vague : first(result)
                end

                # Extrapolating statistics
                n_total = length(y_test)
                n_vague = 0
                n_correct = 0
                
                @inbounds for i in 1:n_total
                    if y_pred[i] == :vague
                        n_vague += 1
                    elseif y_pred[i] == y_test[i]
                        n_correct += 1
                    end
                end
                
                n_wrong = n_total - n_correct - n_vague
                pvague = (n_vague / n_total) * 100
                pcorrect = (n_correct / n_total) * 100
                pwrong = (n_wrong / n_total) * 100

                deltacorrect = (pcorrect - correct[k][1])
                correct[k][1] += deltacorrect / i
                correct[k][2] += deltacorrect * (pcorrect - correct[k][1])

                deltawrong = (pwrong - wrong[k][1])
                wrong[k][1] += deltawrong / i
                wrong[k][2] += deltawrong * (pwrong - wrong[k][1])

                deltavague = (pvague - vague[k][1])
                vague[k][1] += deltavague / i
                vague[k][2] += deltavague * (pvague - vague[k][1])

            end
        end

        # Process results: extract means and compute standard deviations (sample std)
        correct_mean = [x[1] for x in correct]
        correct_std = [sqrt(x[2] / (n_runs - 1)) for x in correct]

        wrong_mean = [x[1] for x in wrong]
        wrong_std = [sqrt(x[2] / (n_runs - 1)) for x in wrong]

        vague_mean = [x[1] for x in vague]
        vague_std = [sqrt(x[2] / (n_runs - 1)) for x in vague]

        df = DataFrame(
            experts=expertcombreadable,
            correct_mean=correct_mean,
            correct_std=correct_std,
            wrong_mean=wrong_mean,
            wrong_std=wrong_std,
            vague_mean=vague_mean,
            vague_std=vague_std
        )
        CSV.write(joinpath(result_dir, "pred_" * dataset), df)
    end
end

# Warmup (optional, checks for compilation time vs runtime)
@time main()



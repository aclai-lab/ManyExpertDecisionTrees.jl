struct ConfusionMatrix{T}
    classes::Vector{T}
    labels::Vector{Vector{T}}
    matrix::Matrix{Int}
end

struct ClassStats
    TP::Float64
    FP::Float64
    TN::Float64
    FN::Float64
end

function confusionmatrix(
    actual::AbstractVector{T}, 
    predicted::AbstractVector{<:AbstractVector{T}}
) where {T}

    classes = Vector{T}(sort(unique(actual)))
    N = length(classes)

    # Ensure labels are explicitly Vector{Vector{T}} for struct compatibility
    labels_raw = unique(vcat([ [l] for l in classes ], predicted))
    labels = Vector{Vector{T}}(map(x -> Vector{T}(x), labels_raw))
    sort!(labels)
    
    M = length(labels)

    _actual = zeros(Int, length(actual))
    _pred = zeros(Int, length(predicted))

    for i in 1:N
        _actual[actual .== classes[i]] .= i
    end

    for i in 1:M
        _pred[predicted .== Ref(labels[i])] .= i
    end

    CM = zeros(Int, N, M)
    for i in zip(_actual, _pred)
        CM[i[1], i[2]] += 1
    end

    return ConfusionMatrix(classes, labels, CM)
end

function getstats(
    cm::ConfusionMatrix{T}, 
    target_class
) where {T}

    row = findfirst(==(target_class), cm.classes)

    if isnothing(row) 
        error("Class $target_class not found")
    end

    total_target_instances = sum(cm.matrix[row, :])
    total_other_instances = sum(cm.matrix) - total_target_instances

    # Find all columns that contain the target class 
    class_cols = [i for (i, pred_set) in enumerate(cm.labels) if target_class in pred_set]
    
    #= 
        In the "fuzzy"/"multilabel" scenario, confusion matrix statistics have been
        generalized to account for both crisp and vague classifications. Thus, we've 
        defined:
           - TP: in the row associated with the target class, sum the predictions 
             that contain the target class weighted by 1/k, where k is the cardinality
             of the set of predicted labels.

           - FP: in all other rows beside the one associated with the target, sum 
             predictions that contain the target class weighted by 1/k, where k is the
             cardinality of the predicted set.

           - TN: in all other rows beside the one associated with the target, 
             classifications that don't contain the target class score 1; classifications 
             that contain the target class score 1 - 1/k.

           - FN: in the row associated with the target class, classifications that 
             don't contain the target class score 1; classifications that contain the 
             target class score 1 - 1/k.
        
        Note that: 
        - TP + FN = n° of instances associated with the taret, while
        - FP + TN = n° of all other instances
    =#

    TP = sum(class_cols; init=0.0) do c
        count = cm.matrix[row, c]
        k = length(cm.labels[c])
        return count * (1.0 / k) 
    end

    FP = sum(class_cols; init=0.0) do c 
        col_total = sum(cm.matrix[:, c])
        count = col_total - cm.matrix[row, c]
        k = length(cm.labels[c])
        return count * (1.0 / k)
    end 

    FN = total_target_instances - TP
    TN = total_other_instances - FP

    return ClassStats(TP, FP, TN, FN)
end

accuracy(stats::ClassStats) = (stats.TP + stats.FP + stats.TN + stats.FN) == 0 ? 0.0 : (stats.TP + stats.TN) / (stats.TP + stats.TN + stats.FP + stats.FN)
precision(stats::ClassStats)   = (stats.TP + stats.FP) == 0 ? 0.0 : stats.TP / (stats.TP + stats.FP)
recall(stats::ClassStats)      = (stats.TP + stats.FN) == 0 ? 0.0 : stats.TP / (stats.TP + stats.FN)
# specificity(stats::ClassStats) = (stats.TN + stats.FP) == 0 ? 0.0 : stats.TN / (stats.TN + stats.FP)
# f1_score(stats::ClassStats)    = (precision(stats) + recall(stats)) == 0 ? 0.0 : 2 * (precision(stats) * recall(stats)) / (precision(stats) + recall(stats))

accuracy(cm::ConfusionMatrix, target_class) = accuracy(getstats(cm, target_class))
precision(cm::ConfusionMatrix, target_class)   = precision(getstats(cm, target_class))
recall(cm::ConfusionMatrix, target_class)      = recall(getstats(cm, target_class))
# specificity(cm::ConfusionMatrix, target_class) = specificity(getstats(cm, target_class))
# f1_score(cm::ConfusionMatrix, target_class)    = f1_score(getstats(cm, target_class))





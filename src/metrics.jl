using SoleLogics.ManyValuedLogics

"""
    struct ConfusionMatrix{T}
        classes::Vector{T}
        labels::Vector{Vector{T}}
        matrix::Matrix{Int}
    end

Rectangular confusion matrix constructed by having real classes as rows and 
predicted label sets as columns. Other statistics are computed starting from 
this structure.
"""
struct ConfusionMatrix{T}
    classes::Vector{T}
    labels::Vector{Vector{T}}
    matrix::Matrix{Int}
end

function Base.convert(
    ::Type{ManyExpertDecisionTrees.ConfusionMatrix{T}}, 
    cm::DecisionTree.ConfusionMatrix;
    classes::Union{Nothing, AbstractVector{T}}=nothing
) where {T}
    dt_classes = map(x -> convert(T, x), cm.classes)

    if classes === nothing
        labels = Vector{Vector{T}}([[l] for l in dt_classes])
        return ConfusionMatrix(dt_classes, labels, cm.matrix)
    end

    global_classes = sort(classes)
    labels = Vector{Vector{T}}([[l] for l in global_classes])
    N = length(global_classes)
    matrix = zeros(Int, N, N)

    for (di, dc) in enumerate(dt_classes)
        gi = findfirst(==(dc), global_classes)
        gi === nothing && continue
        for (dj, dc2) in enumerate(dt_classes)
            gj = findfirst(==(dc2), global_classes)
            gj === nothing && continue
            matrix[gi, gj] = cm.matrix[di, dj]
        end
    end

    return ConfusionMatrix(global_classes, labels, matrix)
end

"""
    struct ClassStats
        TP::Float64
        FP::Float64
        TN::Float64
        FN::Float64
        vagueness::Float64
    end

Simple container structure for storing class-wise statistics starting from a 
confusion matrix
"""
struct ClassStats
    TP::Float64
    FP::Float64
    TN::Float64
    FN::Float64
    vagueness::Float64
end

"""
    function confusionmatrix(
        actual::AbstractVector{T}, 
        predicted::AbstractVector{<:AbstractVector{T}}
    ) where {T}

Given an array of actual labels and an arry of predicted label subsets, construct 
a rectangular confusion matrix.
"""
function confusionmatrix(
    actual::AbstractVector{T}, 
    predicted::AbstractVector{<:AbstractVector{T}};
    classes::Union{Nothing, AbstractVector{T}}=nothing
) where {T}

    classes = isnothing(classes) ? Vector{T}(sort(unique(actual))) : Vector{T}(sort(classes))
    N = length(classes)

    certain_labels = [ [l] for l in classes ]

    other_labels = unique([Vector{T}(p) for p in predicted ])
    filter!(l -> !(l in certain_labels), other_labels)
    sort!(other_labels)

    labels = Vector{Vector{T}}(vcat(certain_labels, other_labels))
    
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

"""
    function getstats(
        cm::ConfusionMatrix{T}, 
        target_class
    )::ClassStats where {T} 

Compute statistics associated with a given class. Returns a ClassStats container
"""
function getstats(
    cm::ConfusionMatrix{T}, 
    target_class
)::ClassStats where {T}

    row = findfirst(==(target_class), cm.classes)

    if isnothing(row) 
        error("Class $target_class not found")
    end

    total_target_instances = sum(cm.matrix[row, :])
    total_other_instances = sum(cm.matrix) - total_target_instances

    # Find all columns that contain the target class 
    class_cols = [i for (i, pred_set) in enumerate(cm.labels) if target_class in pred_set]
    
    tot_preds = 0
    tot_card = 0
    
    # Compute vagueness 1 minus 1 over average cardinality weighted by number of instances 
    for c in class_cols
        count = sum(cm.matrix[:, c]) 
        k = length(cm.labels[c])
        tot_preds += count
        tot_card += count * k
    end
    
    vagueness = tot_preds == 0 ? 0.0 : 1 - (1/( tot_card / tot_preds))
    
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

    FN = max(0.0, total_target_instances - TP)
    TN = max(0.0, total_other_instances - FP)

    return ClassStats(TP, FP, TN, FN, vagueness)
end

# Single class Metrics
accuracy(stats::ClassStats)    = (stats.TP + stats.FP + stats.TN + stats.FN) == 0 ? 0.0 : (stats.TP + stats.TN) / (stats.TP + stats.TN + stats.FP + stats.FN)
precision(stats::ClassStats)   = (stats.TP + stats.FP) == 0 ? 0.0 : stats.TP / (stats.TP + stats.FP)
recall(stats::ClassStats)      = (stats.TP + stats.FN) == 0 ? 0.0 : stats.TP / (stats.TP + stats.FN)
vagueness(stats::ClassStats)   = stats.vagueness
# specificity(stats::ClassStats) = (stats.TN + stats.FP) == 0 ? 0.0 : stats.TN / (stats.TN + stats.FP)
# f1_score(stats::ClassStats)    = (precision(stats) + recall(stats)) == 0 ? 0.0 : 2 * (precision(stats) * recall(stats)) / (precision(stats) + recall(stats))

accuracy(cm::ConfusionMatrix, target_class)    = accuracy(getstats(cm, target_class))
precision(cm::ConfusionMatrix, target_class)   = precision(getstats(cm, target_class))
recall(cm::ConfusionMatrix, target_class)      = recall(getstats(cm, target_class))
vagueness(cm::ConfusionMatrix, target_class)   = vagueness(getstats(cm, target_class))
# specificity(cm::ConfusionMatrix, target_class) = specificity(getstats(cm, target_class))
# f1_score(cm::ConfusionMatrix, target_class)    = f1_score(getstats(cm, target_class))

# Macro averaged metrics
accuracy(cm::ConfusionMatrix)  = mean(accuracy(cm, c) for c in cm.classes)
precision(cm::ConfusionMatrix) = mean(precision(cm, c) for c in cm.classes)
recall(cm::ConfusionMatrix)    = mean(recall(cm, c) for c in cm.classes)
vagueness(cm::ConfusionMatrix) = mean(vagueness(cm, c) for c in cm.classes)

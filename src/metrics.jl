function predict(medt::ManyExpertDecisionTree, algebra::ManyExpertAlgebra, X::AbstractMatrix{S}) where {S}
    return [apply(medt, algebra, row) for row in eachrow(X)]
end

struct ConfusionMatrix
    classes::Vector
    matrix::Matrix{Int}
    accuracy::Float64
end

function Base.show(io::IO, cm::ConfusionMatrix)
    print(io, "Classes:  ")
    show(io, MIME("text/plain"), cm.classes)
    println(io)
    print(io, "Matrix:   ")
    show(io, MIME("text/plain"), cm.matrix)
    println(io)
    print(io, "Accuracy: ")
    show(io, cm.accuracy)
end

function confusion_matrix(actual::AbstractVector, predicted::AbstractVector)
    @assert length(actual) == length(predicted)
    
    labels = sort(unique(actual))
    N = length(labels)

    classes = sort(unique(vcat([ [l] for l in labels ], predicted)))
    M = length(classes)

    _actual = zeros(Int, length(actual))
    _pred = zeros(Int, length(predicted))

    for i in 1:N
        _actual[actual .== labels[i]] .= i
    end

    for i in 1:M
        _pred[predicted .== Ref(classes[i])] .= i
    end

    CM = zeros(Int, N, M)
    for i in zip(_actual, _pred)
        CM[i[1], i[2]] += 1
    end

    total_score = 0.0
    for r in 1:N
        for c in 1:M
            count = CM[r, c]
            if count > 0
                true_val = labels[r]
                pred_val = classes[c]
                
                if true_val in pred_val
                    k = length(pred_val)
                    if k < N
                        total_score += count * (1.0 / k)
                    end
                end
            end
        end
    end

    accuracy = total_score / length(actual)

    return ConfusionMatrix(classes, CM, accuracy)
end

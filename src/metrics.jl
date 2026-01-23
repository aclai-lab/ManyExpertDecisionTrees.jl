function predict(medt::ManyExpertDecisionTree, algebra::ManyExpertAlgebra, X::AbstractMatrix{S}) where {S}
    return [apply(medt, algebra, row) for row in eachrow(X)]
end

struct ConfusionMatrix
    classes::Vector
    matrix::Matrix{Int}
    accuracy::Float64
end

function show(io::IO, cm::ConfusionMatrix)
    print(io, "Classes:  ")
    show(io, cm.classes)
    print(io, "\nMatrix:   ")
    display(cm.matrix)
    print(io, "\nAccuracy: ")
    show(io, cm.accuracy)
end
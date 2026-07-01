function updateExpertHyperParameters(fdt::ManyExpertDecisionTree, id::Int64; kwargs...)
    updateHyperParameters(fdt.mfparams[id][]; kwargs...);
end
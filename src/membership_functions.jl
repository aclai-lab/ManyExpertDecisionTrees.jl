
# Pure functions
sigmoid(x, c, s) = 1 / (1 + exp(-s * (x - c)))
gaussian(x, mu, std) = exp(-(x - mu)^2 / (2std^2))

# HyperParameters specific to membership functions
abstract type AbstractHyperParameters end

mutable struct SigmoidHyperParameters <: AbstractHyperParameters
    sigmoid_slope_scaling
    sigmoid_slope_cap

    function SigmoidHyperParameters(;
        sigmoid_slope_scaling=1,
        sigmoid_slope_cap=Inf
    )
        return new(sigmoid_slope_scaling, sigmoid_slope_cap)
    end    
end

function updateHyperParameters(
    sig::SigmoidHyperParameters;
    sigmoid_slope_scaling=1,
    sigmoid_slope_cap=Inf
    )

    sig.sigmoid_slope_scaling = sigmoid_slope_scaling;
    sig.sigmoid_slope_cap = sigmoid_slope_cap;
end


function Base.show(io::IO, hp::SigmoidHyperParameters)
    print(io, "(slope_scaling = $(hp.sigmoid_slope_scaling), slope_cap = $(hp.sigmoid_slope_cap))")
end

# Membership function wrappers 
abstract type AbstractMembershipFunction end

struct SigmoidMF{Tc <: Real, Ts <: Real} <: AbstractMembershipFunction
    center::Tc
    slope::Ts
    hyperparameters::Ref{SigmoidHyperParameters}

    function SigmoidMF(
        center::Tc, 
        slope::Ts, 
        hyperparameters::Ref{SigmoidHyperParameters}
    ) where {Tc <: Real, Ts <: Real}
        slope_cap = hyperparameters[].sigmoid_slope_cap
        capped_slope = abs(slope) >= slope_cap ? sign(slope) * slope_cap : slope

        return new{Tc, Ts}(center, capped_slope, hyperparameters)
    end
end

function (mf::SigmoidMF)(x::Real)
    scaled_slope = mf.slope * mf.hyperparameters[].sigmoid_slope_scaling
    slope_cap = mf.hyperparameters[].sigmoid_slope_cap

    if abs(scaled_slope) >= slope_cap
        sigmoid(x, mf.center, sign(scaled_slope) * slope_cap)
    else
        sigmoid(x, mf.center, scaled_slope)
    end
end

function Base.show(io::IO, mf::SigmoidMF)
    print(io, "Sigmoid(c=$(mf.center), s=$(mf.slope))")
end

# Functions to init hyperparameters and bind them to their respective functions during construction
function initHyperParameters(
    mfs::Type{<:AbstractMembershipFunction}...;
    kwargs...
)
    return map(mf -> initHyperParameters(mf; kwargs...), mfs)
end

function initHyperParameters(
    ::Type{SigmoidMF};
    kwargs...
    )
    return Ref(SigmoidHyperParameters(;kwargs...))
end

function initHyperParameters(
    mf::Type{<:AbstractMembershipFunction};
    kwargs...
)
    @error "No initialization processes defined for $(mf) type"
end

# Membership function builders 
function build_mfs(
    mf::Type{SigmoidMF},
    feat_id::Int,
    feat_val::Real,
    expert_data::AbstractMatrix,
    hyperparameters::Ref{SigmoidHyperParameters}
)
    leftset, rightset = split_set(feat_val, feat_id, expert_data)
    
    col_data = expert_data[:, feat_id]
    mu, sigma = mean(col_data), std(col_data) # Might do something with the mean still

    slope = isnan(sigma) ? 1e-5 : abs(1 / sigma)

    mf_l = SigmoidMF(feat_val, -slope, hyperparameters)
    mf_r = SigmoidMF(feat_val, slope, hyperparameters)

    return mf_l, mf_r, leftset, rightset
end

function build_mfs(mem_func::Type{<:AbstractMembershipFunction}, args...; kwargs...) 
    error("Unsupported membership function")
end
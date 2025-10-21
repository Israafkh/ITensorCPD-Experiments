using Pkg
## I don't have this path to add. you could add
# Pkg.develop(url="github.com/kmp5VT/ITensorCPD.jl")
## and tell people to uncomment to add ITensorCPD
#Pkg.develop(path = "$(@__DIR__)/../../ITensorCPD.jl")

using ITensorCPD
using ITensors
using HDF5
using Plots
using ITensorNetworks
using Profile


using Random
using ITensorCPD: had_contract

using ITensorCPD: had_contract
function check_fit(als, factors, cprank, λ, fact)
    target = als.target
    ref_norm = sum(target.^2)
    inner_prod = (had_contract([target, dag.(factors)...], cprank) * dag(λ))[]
    partial_gram = [fact * dag(prime(fact; tags=tags(cprank))) for fact in factors];
    fact_square = ITensorCPD.norm_factors(partial_gram, λ)
    normResidual =
        sqrt(abs(ref_norm + fact_square - 2 * abs(inner_prod)))
    fit = normResidual / sqrt(ref_norm)
    println("CPD rank\tMode\tCPD Accuracy")
    println("$(dim(cprank))\t\t$(fact)\t$(fit)")
    return fit
end

#include("./well.jl")
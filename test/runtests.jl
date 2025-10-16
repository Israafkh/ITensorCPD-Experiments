using Pkg
Pkg.develop(path = "$(@__DIR__)/../../ITensorCPD.jl")

using ITensorCPD
using ITensors

include("./well.jl")
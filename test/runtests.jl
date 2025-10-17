using Pkg
## I don't have this path to add. you could add
# Pkg.develop(url="github.com/kmp5VT/ITensorCPD.jl")
## and tell people to uncomment to add ITensorCPD
#Pkg.develop(path = "$(@__DIR__)/../../ITensorCPD.jl")

using ITensorCPD
using ITensors

include("./well.jl")
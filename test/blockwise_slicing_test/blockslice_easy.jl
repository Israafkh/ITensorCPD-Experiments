## This makeas a 5 x 2 x 5 tensor and embeds 7 large leverage scores in the problem. 
## The leveraage scores are then shuffled. then I try to identify them with a single QR, then I go to 
## a set of smaller QR's. The smaller QR's seem to do well!
using Random, LinearAlgebra
include("$(@__DIR__)/../test_env.jl")
T = randn(5,2,5)
T1 = reshape(T, 5, 10)
T1[:,1:7] .*= 100
T1[:,8:10] ./= 100

T1 = T1[:, randperm(10)]
T = permutedims(reshape(T1, 5,2,5), (3,2,1))

levs = compute_lev_score(copy(reshape(T, (5,10))'))
Tf = reshape(T, (5,10))
_,_,p = qr(Tf, ColumnNorm())

plot(sort(levs, rev=true))
plot!(levs[p])

Tf[:,1:2:10]
_,r1,p1 = qr(Tf[:,1:2:10], ColumnNorm())
_,r2,p2 = qr(Tf[:,2:2:10], ColumnNorm())

nr1 = sum.([r1[:,i] for i in 1:5])
nr2 = sum.([r2[:,i] for i in 1:5])
max = maximum([nr1..., nr2...])

nr1 ./ max
nr2 ./ max

p2 .+= 5
pcombo = vcat(p1[[1,2,3,4,5]])

plot(sort(levs, rev=true))
plot!(levs[p1], marker=:circle)
plot!(levs[p2], marker=:circle)

@show sortperm(levs; rev=true)
@show p1, p2

@show vcat(p1[[1,2,3]], p2[[1,2,3]], p1[[4,5]], p2[[4,5]])
@show vcat(p1, p2)[sortperm(vcat(nr1 ./ max, nr2 ./ max))]
@show sortperm(levs)
@show sortperm(levs; rev=true)
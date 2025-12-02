# This is to check to see how the pivots compare to the positions of the leverage scores in the QR of a random tensor
include("test_env.jl")

using LinearAlgebra, AppleAccelerate, Plots, Random, Revise
elt = Float64

########
# order 3 mode 1
########
c = 0.8
#i,j,k = 500,500,500
i,j,k = 531,308,640
#A, cp = Colinearity_Tensor(25, 3, (i,j,k), elt(c), nothing, elt)
#r = ITensorCPD.cp_rank(cp)
#A = random_itensor(elt, Index.((i,j,k)))
# cp = ITensorCPD.random_CPD(A, 5)
# r = Index(10, "CPRank")
bad = 160
r = 200
cpd = construct_large_lev_score_cpd((i,j,k), r, bad);
r = ITensorCPD.cp_rank(cpd);
T = had_contract(cpd[1], cpd[2], r) * had_contract(cpd[3], cpd[], r)

target = T

_,_,pT = qr(reshape(array(target), (i,j*k)), ColumnNorm())
krpm = reshape(array(ITensorCPD.had_contract(cpd[2], cpd[3], r)), (j * k , dim(r)))
Tm = compute_lev_score(krpm)

rel = Tm ./ maximum(Tm)
sum(i > 0.1 for i in rel)
plot(sort(Tm; rev=true)[1:50], marker=:star, label="Sorted Scores")
plot!((Tm[pT][1:50];), label="Pivoted QR Ordering")
plot!(title="Sorted Leverage Scores of the KRP",ylabel="Leverage Score Value", xlabel="Sorted Position")
# plot!(yrange=[-0.01,0.5])

savefig("plots/lev_score_order/rank_($(i),$(j),$(k))_bad_$(bad)_rank_$(dim(r)).pdf")


###################
## Middle Mode order 3
###################
#i,j,k = 500,500,500
i,j,k = 531,308,640
#A, cp = Colinearity_Tensor(25, 3, (i,j,k), elt(c), nothing, elt)
#r = ITensorCPD.cp_rank(cp)
#A = random_itensor(elt, Index.((i,j,k)))
# cp = ITensorCPD.random_CPD(A, 5)
# r = Index(10, "CPRank")
bad = 2
r = 10
cpd = construct_large_lev_score_cpd((i,j,k), r, bad);
r = ITensorCPD.cp_rank(cpd);
T = had_contract(cpd[2], cpd[1], r) * had_contract(cpd[3], cpd[], r)

target = T

_,_,pT = qr(reshape(array(target), (j, i*k)), ColumnNorm())
krpm = reshape(array(ITensorCPD.had_contract(cpd[1], cpd[3], r)), (i * k , dim(r)))
Tm = compute_lev_score(krpm)

plot(sort(Tm; rev=true)[1:50], marker=:star, label="Exact Sorting")
plot!((Tm[pT][1:50];), label="Pivoted QR Ordering")
plot!(title="Sorted Leverage Scores of the KRP",ylabel="Leverage Score Value", xlabel="Sorted Position")
# plot!(yrange=[-0.01,0.5])

savefig("plots/lev_score_order/rank_($(i),$(j),$(k))_bad_$(bad)_rank_$(dim(r)).pdf")


####################
## order 4 test
####################
#i,j,k,l = 40,40,40,40
i,j,k,l = 40, 53, 73, 32
#A, cp = Colinearity_Tensor(25, 3, (i,j,k), elt(c), nothing, elt)
#r = ITensorCPD.cp_rank(cp)
#A = random_itensor(elt, Index.((i,j,k)))
# cp = ITensorCPD.random_CPD(A, 5)
# r = Index(10, "CPRank")
bad = 2
r = 30
cpd = construct_large_lev_score_cpd((i,j,k,l), r, bad);
r = ITensorCPD.cp_rank(cpd);
T = had_contract([cpd[1], cpd[2],cpd[3]], r) * had_contract(cpd[4], cpd[], r)

target = T

_,_,pT = qr(reshape(array(target), (i,j*k*l)), ColumnNorm())
krpm = reshape(array(ITensorCPD.had_contract([cpd[2], cpd[3], cpd[4]], r)), (j * k * l, dim(r)))
Tm = compute_lev_score(krpm)

plot(sort(Tm; rev=true)[1:50], marker=:star, label="True Sorted Scores")
plot!(sort(Tm[pT][1:50]; rev=true), label="Pivoted QR Ordering")
plot!(title="Sorted Leverage Scores of the KRP",ylabel="Leverage Score Value", xlabel="Sorted Position")
# plot!(yrange=[-0.01,0.5])
savefig("plots/lev_score_order/rank_($(i),$(j),$(k),$(l))_bad_$(bad)_rank_$(dim(r)).pdf")

####################
## mode 2 order 4 test
####################
#i,j,k,l = 40,40,40,40
i,j,k,l = 40, 53, 73, 32
#A, cp = Colinearity_Tensor(25, 3, (i,j,k), elt(c), nothing, elt)
#r = ITensorCPD.cp_rank(cp)
#A = random_itensor(elt, Index.((i,j,k)))
# cp = ITensorCPD.random_CPD(A, 5)
# r = Index(10, "CPRank")
bad = 2
r = 30
cpd = construct_large_lev_score_cpd((i,j,k,l), r, bad);
r = ITensorCPD.cp_rank(cpd);
T = had_contract([cpd[2], cpd[1],cpd[3]], r) * had_contract(cpd[4], cpd[], r)

target = T

_,_,pT = qr(reshape(array(target), (j,i*k*l)), ColumnNorm())
krpm = reshape(array(ITensorCPD.had_contract([cpd[1], cpd[3], cpd[4]], r)), (i * k * l, dim(r)))
Tm = compute_lev_score(krpm)

rel = Tm ./ maximum(Tm)
sum(i > 0.1 for i in rel)
plot(sort(Tm; rev=true)[1:50], marker=:star, label="True Sorted Scores")
plot!(sort(Tm[pT][1:50]; rev=true), label="Pivoted QR Ordering")
plot!(title="Sorted Leverage Scores of the KRP",ylabel="Leverage Score Value", xlabel="Sorted Position")
# plot!(yrange=[-0.01,0.5])
savefig("plots/lev_score_order/rank_($(i),$(j),$(k),$(l))_bad_$(bad)_rank_$(dim(r)).pdf")


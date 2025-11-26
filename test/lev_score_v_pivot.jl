# This is to check to see how the pivots compare to the positions of the leverage scores in the QR of a random tensor
include("test_env.jl")

using LinearAlgebra, AppleAccelerate, Plots, Random
elt = Float64
c = 0.8
i,j,k = 330,34,35
#A, cp = Colinearity_Tensor(25, 3, (i,j,k), elt(c), nothing, elt)
#r = ITensorCPD.cp_rank(cp)
#A = random_itensor(elt, Index.((i,j,k)))
# cp = ITensorCPD.random_CPD(A, 5)
# r = Index(10, "CPRank")
bad = 2
cpd = construct_with_large_levs((i,j,k), 5, bad)
r = ITensorCPD.cp_rank(cpd)

krp = ITensorCPD.had_contract(cpd[2], cpd[3], r)
krpm = reshape(array(krp), (j * k), dim(r))
levs = compute_lev_score(krpm)
_,_,p = qr(krpm', ColumnNorm())

T = had_contract(cpd[1], cpd[2], r) * had_contract(cpd[3], cpd[], r)
Tm = reshape(array(T), i, j*k)
_,_,pT = qr(Tm, ColumnNorm())
levT = compute_lev_score(copy(Tm'))

plot(levs)
plot!(levs[p])
plot!(levs[pT])

#plot(levT)



A = ITensorCPD.had_contract(cp[1], cp[2], r) * ITensorCPD.had_contract(cp[3], cp[], r)
t = reshape(array(A), (i, j * k))
_,rp,p = qr(t, ColumnNorm());

# k_sk = 1000
# m = dim(i)
# l=Int(round(3 * m * log(m))) 
# s=Int(round(log(m)))
# n = 1
# _,rp,p = ITensorCPD.SEQRCS(A,n,ind(A, 1),l,s,k_sk)

krp = had_contract(cp[2], cp[3], r)

krpm = reshape(array(krp), (j * k, dim(r)))

s = qr(krpm');
combo = Index(j * k)
q,_ = qr(krpm)
q = copy(q)
q .*= q
levs = [sum(q[i,1:dim(r)]) for i in 1:dim(combo)]
#levs = ITensorCPD.compute_leverage_score_probabilitiy(itensor(krpm, combo, r), combo)

sorted_levs = sortperm(levs; rev=true)

plot(sortperm(levs; rev=true)[1:i]; marker=:circle)
plot!(p[1:i]; marker=:circle)
levs_in_p = [sorted_levs[x] ∈ p[1:i] for x in 1:i]
@show sum(levs_in_p) / i
sum([sorted_levs[x] ∈ p[1:i] for x in 1:bad]) / bad

opt = nothing
r = Index(25, "CPD")
@time begin
    s =  Int(round(dim(r)^(1.5)))
    guess = ITensorCPD.random_CPD(A, r);
    als = ITensorCPD.compute_als(A, guess; alg=ITensorCPD.SEQRCSPivProjected(1, s, (1,2,3), (10)), check=ITensorCPD.CPDiffCheck(1e-3, 100,));
    #als = ITensorCPD.compute_als(A, guess; alg=ITensorCPD.QRPivProjected(1,s), check=ITensorCPD.CPDiffCheck(1e-3, 1000,));
    #als = ITensorCPD.compute_als(A, guess; alg=ITensorCPD.LevScoreSampled(s), check=ITensorCPD.CPDiffCheck(1e-3, 1000,));
    opt = ITensorCPD.optimize(guess, als; verbose=true);
    check_fit(als, opt.factors, r, opt.λ, 1)
end;



###
#i,j,k = (200, 300, 400)
#cpdT = construct_with_large_levs((i,j,k), 30, 5);
# A, cpdT = Colinearity_Tensor(10, (i,j,k), 0.8);
# r = cp_rank(cpdT)

# i,j = size(cpdT[1])
# A = array(cpdT[1])
# q,_ = qr(A)
# q = copy(q)
# q .*= q

# levs = [sum(q[i,1:j]) for i in 1:i]

# _,_,p = qr(A', ColumnNorm())

# plot(levs)
# plot!(levs[p])
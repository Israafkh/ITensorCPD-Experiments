# This is to check to see how the pivots compare to the positions of the leverage scores in the QR of a random tensor
include("test_env.jl")

using LinearAlgebra, AppleAccelerate, Plots, Random
elt = Float64
c = 0.8
i,j,k = 200,300,400
#A, cp = Colinearity_Tensor(5, 3, (i,j,k), elt(c), nothing, elt)
#A = random_itensor(elt, Index.((i,j,k)))
# cp = ITensorCPD.random_CPD(A, 5)
cp = construct_large_lev_score_cpd((i,j,k), 10, 2)

r = ITensorCPD.cp_rank(cp)
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
levs = ITensorCPD.compute_leverage_score_probabilitiy(itensor(krpm, combo, r), combo)

sorted_levs = sortperm(levs; rev=true)

plot(sortperm(levs; rev=true)[1:i]; marker=:circle)
plot!(p[1:i]; marker=:circle)

levs_in_p = [sorted_levs[x] ∈ p[1:i] for x in 1:i]
@show sum(levs_in_p) / i
sum([sorted_levs[x] ∈ p[1:i] for x in 1:2])

guess = ITensorCPD.random_CPD(A, r);
als = ITensorCPD.compute_als(A, guess; alg=ITensorCPD.QRPivProjected(1,500,), check=ITensorCPD.FitCheck(1e-3, 300, norm(A)));
ITensorCPD.optimize(guess, als; verbose=true);
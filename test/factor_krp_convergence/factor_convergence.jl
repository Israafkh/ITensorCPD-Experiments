include("$(@__DIR__)/../test_env.jl")
using Plots
using Random
using LinearAlgebra: ColumnNorm, diagm, svdvals
using LinearAlgebra


i,j,k = 90,90,90
samps = 1500
bad = 3
r = 10

cpd_exact = ITensorCPD.random_CPD(zeros(i,j,k), r);
for x in 1:3
array(cpd_exact[x])[1:bad, :] .*= 10;
array(cpd_exact[x]) .= array(cpd_exact[x])[randperm(90), :]
end
r = ITensorCPD.cp_rank(cpd_exact);
T = had_contract(cpd_exact[1], cpd_exact[2], r) * had_contract(cpd_exact[3], cpd_exact[], r)

cpd_exact = ITensorCPD.als_optimize(T, cpd_exact; check=ITensorCPD.FitCheck(1e-10, 100, norm(T)), verbose=true);
T = had_contract(cpd_exact[1], cpd_exact[2], r) * had_contract(cpd_exact[3], cpd_exact[], r)

## Compute the KRPs
function get_krp(cpd, mode)
    inds = filter(x -> x != mode, 1:3)
    krp = ITensorCPD.had_contract([cpd[x] for x in inds], r)
    fused_idx = Index(prod([dim(ind(cpd_exact, i)) for i in inds]))
    return krp, fused_idx
end

Q_exact_vect = Vector{Matrix{Float64}}() 
for x in 1:3
    krp, fused = get_krp(cpd_exact, x)
    krp_exact = ITensorCPD.itensor(array(krp), fused, r)
    krp_exact_matrix = Matrix(krp_exact,fused, r)
    Q_exact,_,_=svd(Matrix(krp_exact,fused, r))
    Q_exact=Matrix(Q_exact)[:,1:1]
    push!(Q_exact_vect,Q_exact)
end 

### Initial guesses for CPD 
#cpd = ITensorCPD.random_CPD(T, 100; rng=RandomDevice());
cpd = deepcopy(cpd_exact)
λ = 0.01
for i in cpd
    i .+= λ .* random_itensor(inds(i))
end
## Random guess information 
sref = [svd(array(cpd_exact[i] - cpd[i])).S for i in 1:3]

## Convergence checks
check=ITensorCPD.FitCheck(1e-8, 100, norm(T))

## ALS algorithms 
alg = ITensorCPD.SEQRCSPivProjected(1, 2000, (1,2,3), 100)
# alg = ITensorCPD.LevScoreSampled(4000)
# alg = ITensorCPD.direct()
als = ITensorCPD.compute_als(T, cpd; alg, check,normal=false);
als.target .= T

## Copy ALS to run a bunch of single ALS updates to learn about convergence behavior.
upcpd = deepcopy(cpd);
upals = deepcopy(als);
s1, s2, s3 = [],[],[]
ss = [s1,s2,s3]
for x in 1:20
    for (i,si) in zip(1:3,ss)
        upcpd, upals,_ = single_solve(upals, upcpd, i)
        s = svdvals(array(cpd_exact[i] - upcpd[i]))
        push!(si, s)
    end
end
tspec = [svdvals(array(cpd[x])) for x in 1:3]
lw = 3
plot(sref[1]./tspec, label="Random Guess"; lw)
for i in 1:5:length(s1)
    plot!(s1[i] ./ tspec, label="Iter $(i)"; lw)
end
plot!(xlabel="Singular Value Position", ylabel="Singular value", 
title="Relative Spectrum of Difference between \nFound and Best Mode 1 Factor",
legend=:right,
legendfontsize=9, 
labelfontsize=14,
tickfontsize=12,)
savefig("$(@__DIR__)/../../plots/factor_convergence/svd_vals_diff_factors_qr_mode_1.pdf")

plot(vcat(maximum(sref[1]), maximum.(s1)), label="Mode 1", marker=:circle; lw)
plot!(vcat(maximum(sref[2]), maximum.(s2)), label="Mode 2", marker=:circle; lw)
plot!(vcat(maximum(sref[3]), maximum.(s3)), label="Mode 2", marker=:circle; lw)
plot!(xlabel="ALS iteration", ylabel="Max Singular Value", 
title="Absolute largest singular value in Difference from\n found and known CPD factor",
legendfontsize=9, 
labelfontsize=14,
tickfontsize=12,)
savefig("$(@__DIR__)/../../plots/factor_convergence/svd_vals_diff_factors_periter_qr.pdf")

include("$(@__DIR__)/test_env.jl")
using Plots
using Random
using LinearAlgebra: ColumnNorm, diagm, svdvals
using LinearAlgebra


i,j,k = 90,90,90
samps = 1500
bad = 20
r = 60
# cpd_exact = ITensorCPD.random_CPD(zeros(i,j,k), r);
# array(cpd_exact[1])[1:bad, :] .*= 100;
# array(cpd_exact[2])[1:bad, :] .*= 100;
# array(cpd_exact[3])[1:bad, :] .*= 100;
# array(cpd_exact[1]) .= array(cpd_exact[1])[randperm(i), :]
# array(cpd_exact[2]) .= array(cpd_exact[2])[randperm(j), :]
# array(cpd_exact[3]) .= array(cpd_exact[3])[randperm(k), :]
# r = ITensorCPD.cp_rank(cpd_exact);
seed = Xoshiro(123)
rng = RandomDevice()
cpd_exact = construct_large_lev_score_cpd((i,j,k), r, bad;rng);
r = ITensorCPD.cp_rank(cpd_exact)

T = had_contract(cpd_exact[1], cpd_exact[2], r) * had_contract(cpd_exact[3], cpd_exact[], r)
Q_exact_vect = Vector{Matrix{Float64}}() 
for i in 1:3
    inds = filter(x -> x != i, 1:3)
    krp = ITensorCPD.had_contract([cpd_exact[i] for i in inds], r)
    fused = Index(prod([dim(ind(cpd_exact, i)) for i in inds]))
    krp_exact = ITensorCPD.itensor(array(krp), fused, r)
    krp_exact_matrix = Matrix(krp_exact,fused, r)
    Q_exact,_,_=svd(Matrix(krp_exact,fused, r))
    Q_exact=Matrix(Q_exact)[:,1:1]
    push!(Q_exact_vect,Q_exact)
end 




function proj_angle_compute(cpd, inds,fact)
    cpr = ITensorCPD.cp_rank(cpd)
    krp = ITensorCPD.had_contract([cpd[i] for i in inds], cpr)
    fused = Index(prod([dim(ind(cpd, i)) for i in inds]))
    krp = ITensorCPD.itensor(array(krp), fused, cpr)
    krp_matrix = Matrix(krp,fused, cpr)
    Q,_,_=svd(krp_matrix)
    Q1= Matrix(Q)[:,1:1]
    Q2 = Q_exact_vect[fact]
    σ = svdvals(Q1' * Q2)
    # sqrt(1-(minimum(σ)^2))
    # sqrt(100-(sum(σ.^2)))/sqrt(100)
    return sqrt(1-(minimum(σ))^2)
end

global upcpd, upals
cpd = ITensorCPD.random_CPD(T, 100; rng=RandomDevice());
check=ITensorCPD.FitCheck(1e-3, 100, norm(T))
check_piv = ITensorCPD.CPAngleCheck(1e-5, 100,cpd_exact,norm(T))
alg = ITensorCPD.SEQRCSPivProjected(1, samps, (1,2,3), (40,40,40))
# alg = ITensorCPD.direct()
als = ITensorCPD.compute_als(T, cpd; alg, check=check_piv, trunc_tol=1e-10,normal=false)
int_opt_T =ITensorCPD.optimize(cpd,als;verbose=true);
als.target .= T
# alslev = ITensorCPD.compute_als(T,cpd_exact;alg = ITensorCPD.LevScoreSampled(samps),check=check_piv)
# int_opt_T = ITensorCPD.optimize(cpd, alslev; verbose=true)
upcpd, upals,_ = single_solve(als, cpd, 1)
δ1 = proj_angle_compute(upcpd, (2,3),1)

upcpd, upals, _ = single_solve(upals, upcpd, 2)
δ2 = proj_angle_compute(upcpd, (1,3),2)
upcpd, upals, _ = single_solve(upals, upcpd, 3)
δ3 = proj_angle_compute(upcpd, (1,2),3)
δ1_vect = [δ1]
δ2_vect = [δ2]
δ3_vect = [δ3]
for i in 1:100
    global upcpd, upals 
    upcpd, upals,_ = single_solve(upals, upcpd, 1)
    δ1= proj_angle_compute(upcpd, (2,3),1)
    upcpd, upals,_ = single_solve(upals, upcpd, 2)
    δ2 = proj_angle_compute(upcpd, (1,3),2)
    upcpd, upal,_ = single_solve(upals, upcpd, 3)
    δ3 = proj_angle_compute(upcpd, (1,2),3)
    push!(δ1_vect, δ1)
    push!(δ2_vect, δ2)
    push!(δ3_vect, δ3)
end
plot(δ1_vect, label="δ1")
plot!(δ2_vect, label="δ2")
plot!(δ3_vect, label="δ3")

xlabel!("Iteration")
ylabel!("Proj_distance")
title!("δ values over iterations")



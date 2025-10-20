using HDF5
using Plots
using ITensorNetworks
using Profile


using Random
using ITensorCPD: had_contract
path = "$(@__DIR__)/datasets/gray_scott_reaction_diffusion/data/train/"

files = readdir(path)
file = h5open(path*files[1])
press = file["t0_fields"]["A"];
press_traj = press[:,:,1:200, 140:160];

 nsteps = size(press_traj,3)
 A = ITensors.itensor(press_traj,Index.(size(press_traj)));
 println(size(A))

 cp_A = ITensorCPD.random_CPD(A, 1,rng=RandomDevice())
check = ITensorCPD.CPDiffCheck(1e-5, 50)
alg = ITensorCPD.SEQRCSPivProjected((1,1,1,1), (400, 400, 400, 400), (1,2,3,4),(10,10,10,10))
alsQR = ITensorCPD.compute_als(A, cp_A; alg, check);

alg = ITensorCPD.LevScoreSampled((400, 400, 400,400))
alsLev = ITensorCPD.compute_als(A, cp_A; alg, check);

# [3,5,7,10,20,30,40,50,100,150,200,250]
ranks = [3,10,30,50,100,200]
error_SEQRCS = Vector{Float64}()
error_leverage = Vector{Float64}()
error_direct = Vector{Float64}()

for rk in ranks
    check = ITensorCPD.FitCheck(1e-3, 50,sqrt(sum(A.^2)) )
    r = Index(rk, "CP_rank")
    cp_A = ITensorCPD.random_CPD(A, r,rng=RandomDevice())
    verbose = true
     opt_A = ITensorCPD.als_optimize(A, cp_A; alg = ITensorCPD.direct(), check, verbose)
    direct_error =  norm(A - ITensorCPD.reconstruct(opt_A)) / sqrt(sum(A.^2))
    push!(error_direct,direct_error)

    int_opt_A = ITensorCPD.optimize(cp_A, alsQR; verbose);
    SEQRCS_error = check_fit(alsQR, int_opt_A.factors, r, int_opt_A.λ, 1)
    println("result for active using SEQRCS: ",SEQRCS_error)
    push!(error_SEQRCS,SEQRCS_error)

    
     lev_opt_A = ITensorCPD.optimize(cp_A, alsLev; verbose);
    lev_error = check_fit(alsLev, lev_opt_A.factors, r, lev_opt_A.λ, 1)
    println("result for active using leverage method is ",lev_error)
    push!(error_leverage,lev_error)
end
plt = plot(ranks, error_SEQRCS, marker=:o, label="SEQRCSPivProjected")
plot!(ranks, error_leverage, marker=:o, label="LevScoreSampled")
plot!(ranks, error_direct, marker=:o, label="Direct")
xlabel!("Rank")
ylabel!("Error")
title!("Error vs rank for gray_scott_reaction_diffusion")
display(plt)


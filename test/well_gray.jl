include("test_env.jl")
using BenchmarkTools

path = "$(@__DIR__)/datasets/gray_scott_reaction_diffusion/data/train/"

files = readdir(path)
file = h5open(path*files[1])
press = file["t0_fields"]["A"];
press_traj = press[:,:,1:200, 150:160];

 nsteps = size(press_traj,3)
 A = ITensors.itensor(press_traj,Index.(size(press_traj)));
 println(size(A))

cp_A = ITensorCPD.random_CPD(A, 1,rng=RandomDevice())
check = ITensorCPD.CPDiffCheck(1e-7, 50)
alg = ITensorCPD.SEQRCSPivProjected((1,1,1,1), (1500, 1500, 1500, 1500), (1,2,3,4),(1,1,1,1))
@time alsQR = ITensorCPD.compute_als(A, cp_A; alg, check);

alg = ITensorCPD.LevScoreSampled((1500,))
@time alsLev = ITensorCPD.compute_als(A, cp_A; alg, check);

# [3,5,7,10,20,30,40,50,100,150,200,250]
ranks = [3,10,40,100,200,400]
error_SEQRCS = Vector{Float64}()
error_leverage = Vector{Float64}()
error_direct = Vector{Float64}()

for rk in ranks
    check = ITensorCPD.FitCheck(1e-3, 50,sqrt(sum(A.^2)) )
    r = Index(rk, "CP_rank")
    cp_A = ITensorCPD.random_CPD(A, r,rng=RandomDevice())
    verbose = false


    alsNormal = ITensorCPD.compute_als(A, cp_A; alg=ITensorCPD.direct(), check);
    opt_A = ITensorCPD.optimize(cp_A, alsNormal; verbose);
    direct_error = check_fit(alsNormal, opt_A.factors, r, opt_A.λ, 1)
    push!(error_direct,direct_error)
  

    # println("SEQRCS")
    # @btime 
    int_opt_A = ITensorCPD.optimize(cp_A, alsQR; verbose=false);
    SEQRCS_error = check_fit(alsQR, int_opt_A.factors, r, int_opt_A.λ, 1)
    println("result for active using SEQRCS: ",SEQRCS_error)
    push!(error_SEQRCS,SEQRCS_error)

    #println("LevScore")
    #@btime 
    lev_opt_A = ITensorCPD.optimize(cp_A, alsLev; verbose=false);
    lev_error = check_fit(alsLev, lev_opt_A.factors, r, lev_opt_A.λ, 1)
    println("result for active using leverage method is ",lev_error)
    push!(error_leverage,lev_error)
end
plt = plot(ranks, error_SEQRCS, marker=:o, label="SEQRCSPivProjected")
plot!(ranks, error_leverage, marker=:o, label="LevScoreSampled")
plot!(ranks, error_direct, marker=:o, label="Direct")

xlabel!("Rank")
ylabel!("Relative Error in CP Fit")
title!("Error in CP Fit vs Rank\n Data: Gray Scott Reaction Diffusion")
# savefig("$(@__DIR__)/../plots/well_gray/Gray_Error_t0_A.pdf")
display(plt)

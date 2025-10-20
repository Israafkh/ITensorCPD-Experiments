using HDF5
using Plots
using ITensorNetworks
using Profile


using Random


## IT would be good here to add a warning to add the correct data maybe

path = "$(@__DIR__)/datasets/active_matter/data/train/"
files = readdir(path)
dat = Array{Float32}(undef, 256, 256, 81);
i = 1
file = h5open(path*files[i])
press = file["t2_fields"]["D"];
press_traj = press[1,1,:,:,:,1]
dat[:,:,:] = press_traj

nsteps = size(dat)[end]
A = ITensors.itensor(dat, Index.(size(dat)))
println(size(A))
error_SEQRCS = Vector{Float64}()
error_leverage = Vector{Float64}()
error_direct = Vector{Float64}()
ranks = [1,2,3,5,10,15,30,50]
for rk in ranks
    r = Index(rk, "CP_rank")
    check = ITensorCPD.FitCheck(1e-4, 50,sqrt(sum(A.^2)) )
    verbose = true
    cp_A = ITensorCPD.random_CPD(A, r, rng=RandomDevice())
    als = ITensorCPD.compute_als(A, cp_A; alg=ITensorCPD.direct(), check);

     opt_A = ITensorCPD.optimize(cp_A, als; verbose);
    #direct_error =  norm(A - ITensorCPD.reconstruct(opt_A)) / sqrt(sum(A.^2)) 
     direct_error = check_fit(als, opt_A.factors, r, opt_A.λ, 1)
     push!(error_direct,direct_error)

    check = ITensorCPD.CPDiffCheck(1e-7, 50)

    alg = ITensorCPD.SEQRCSPivProjected((1,1,1,1), (200,200,200,200), (1,2,3,4), (10,10,10,10))
    als = ITensorCPD.compute_als(A, cp_A; alg, check);
    #  int_opt_A =
    # ITensorCPD.als_optimize(A, cp_A; alg, verbose);
    int_opt_A = ITensorCPD.optimize(cp_A, als; verbose);
    #SEQRCS_error =   norm(A - ITensorCPD.reconstruct(int_opt_A)) / sqrt(sum(A.^2))
    SEQRCS_error = check_fit(als, int_opt_A.factors, r, int_opt_A.λ, 1)
    println("result for active using SEQRCS: ",SEQRCS_error)
    push!(error_SEQRCS,SEQRCS_error)
    
    
    alg = ITensorCPD.LevScoreSampled((200,))
    lev_opt_A = ITensorCPD.als_optimize(A, cp_A; alg, check, verbose);
    #lev_error =   norm(A - ITensorCPD.reconstruct(int_opt_A)) / norm(A) 
    lev_error = check_fit(als, lev_opt_A.factors, r, lev_opt_A.λ, 1)
    println("result for active using leverage method is ",lev_error)
    push!(error_leverage,lev_error)
    
end
plt = plot(ranks, error_SEQRCS, marker=:o, label="SEQRCSPivProjected")
plot!(ranks, error_leverage, marker=:o, label="LevScoreSampled")
plot!(ranks, error_direct, marker=:o, label="Direct")
xlabel!("Rank")
ylabel!("Error")
title!("Error vs rank for active_matter")
display(plt)




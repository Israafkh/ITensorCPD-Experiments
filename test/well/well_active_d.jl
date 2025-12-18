include("$(@__DIR__)/../test_env.jl")

## IT would be good here to add a warning to add the correct data maybe

path = "$(@__DIR__)/datasets/active_matter/data/train/"
files = readdir(path)
dat = Array{Float32}(undef, 2,256, 256, 81);
 i = 1
#for i in 1:45
    file = h5open(path*files[i])
    press = file["t2_fields"]["D"];
    press_traj = press[:,1,:,:,:,1]
    dat[:,:,:,:] = press_traj
#end

nsteps = size(dat)[end]
A = ITensors.itensor(dat, Index.(size(dat)))
println(size(A))
ranks = [1,10,20,50,100,200]

verbose = true
cp_A = ITensorCPD.random_CPD(A, 1, rng=RandomDevice())

check = ITensorCPD.CPDiffCheck(1e-6, 50)
alg = ITensorCPD.SEQRCSPivProjected((1,1,1,1,1), (400,400,400,400,400), (1,2,3,4,5), (2,10,10,10,10))
alsQR = ITensorCPD.compute_als(A, cp_A; alg, check);

alg = ITensorCPD.LevScoreSampled((400,))
alsLev = ITensorCPD.compute_als(A, cp_A; alg, check);

error_SEQRCS = Vector{Float64}()
error_leverage = Vector{Float64}()
error_direct = Vector{Float64}()
for rk in ranks
    r = Index(rk, "CP_rank")
    check = ITensorCPD.FitCheck(1e-3, 50, norm(A) )
    cp_A = ITensorCPD.random_CPD(A, r, rng=RandomDevice())

    alsNormal = ITensorCPD.compute_als(A, cp_A; alg=ITensorCPD.direct(), check);
    opt_A = ITensorCPD.optimize(cp_A, alsNormal; verbose);
    direct_error = check_fit(alsNormal, opt_A.factors, r, opt_A.λ, 1)
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
title!("Error vs rank for D data: active_matter")
display(plt)
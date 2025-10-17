using HDF5
using Plots
using ITensorNetworks
using Profile


using Random
using ITensorCPD: had_contract
function check_fit(als, factors, cprank, λ, fact)
    target = als.target
    ref_norm = sum(target.^2)
    inner_prod = (had_contract([target, dag.(factors)...], cprank) * dag(λ))[]
    partial_gram = [fact * dag(prime(fact; tags=tags(cprank))) for fact in factors];
    fact_square = ITensorCPD.norm_factors(partial_gram, λ)
    normResidual =
        sqrt(abs(ref_norm + fact_square - 2 * abs(inner_prod)))
    fit = normResidual / sqrt(ref_norm)
    println("CPD rank\tMode\tCPD Accuracy")
    println("$(dim(cprank))\t\t$(fact)\t$(fit)")
    return fit
end

## IT would be good here to add a warning to add the correct data maybe

path = "$(@__DIR__)/datasets/active_matter/data/train/"
files = readdir(path)
dat = Array{Float32}(undef, 45, 256, 256, 81);
for i in 1:45
    file = h5open(path*files[10])
    press = file["t0_fields"]["concentration"];
    press_traj = press[:,:,:,1]
    dat[i,:,:,:] = press_traj
end

nsteps = size(dat)[end]
A = ITensors.itensor(dat, Index.(size(dat)))
println(size(A))
error_SEQRCS = Vector{Float64}()
error_leverage = Vector{Float64}()
error_direct = Vector{Float64}()
for rk in [3,5,7,10,20,30,40,50,100,150,200,250,300]

    r = Index(rk, "CP_rank")
    check = ITensorCPD.FitCheck(1e-3, 50,sqrt(sum(A.^2)) )
    verbose = true
    cp_A = ITensorCPD.random_CPD(A, r, rng=RandomDevice())
    als = ITensorCPD.compute_als(A, cp_A; alg=ITensorCPD.direct(), check);

     opt_A = ITensorCPD.optimize(cp_A, als; verbose);
    #direct_error =  norm(A - ITensorCPD.reconstruct(opt_A)) / sqrt(sum(A.^2)) 
     direct_error = check_fit(als, opt_A.factors, r, opt_A.λ, 1)
     push!(error_direct,direct_error)

    check = ITensorCPD.CPDiffCheck(1e-5, 50)

    alg = ITensorCPD.SEQRCSPivProjected((1,1,1,1), (600,600,600,600), (1,2,3,4), (400,400,400,100))
    als = ITensorCPD.compute_als(A, cp_A; alg, check);
    #  int_opt_A =
    # ITensorCPD.als_optimize(A, cp_A; alg, verbose);
    int_opt_A = ITensorCPD.optimize(cp_A, als; verbose);
    #SEQRCS_error =   norm(A - ITensorCPD.reconstruct(int_opt_A)) / sqrt(sum(A.^2))
    SEQRCS_error = check_fit(als, int_opt_A.factors, r, int_opt_A.λ, 1)
    println("result for active using SEQRCS: ",SEQRCS_error)
    push!(error_SEQRCS,SEQRCS_error)
    
    
    alg = ITensorCPD.LevScoreSampled((600,))
     int_opt_A = ITensorCPD.als_optimize(A, cp_A; alg, check, verbose);
    lev_error =   norm(A - ITensorCPD.reconstruct(int_opt_A)) / norm(A) 
    println("result for active using leverage method is",lev_error)
    push!(error_leverage,lev_error)
    
end
plt = plot([3,5,7,10,20,30,40,50,100,150,200,250,300], error_SEQRCS, marker=:o, label="SEQRCSPivProjected")
plot!([3,5,7,10,20,30,40,50,100,150,200,250,300], error_leverage, marker=:o, label="LevScoreSampled")
plot!([3,5,7,10,20,30,40,50,100,150,200,250,300], error_direct, marker=:o, label="Direct")
xlabel!("Rank")
ylabel!("Error")
title!("Error vs rank for active_matter")
display(plt)


path = "$(@__DIR__)/datasets/gray_scott_reaction_diffusion/data/train/"

files = readdir(path)
file = h5open(path*files[1])
 press = file["t0_fields"]["A"];
 press_traj = press[:,:,1:200,1]

 nsteps = size(press_traj,3)
 A = ITensors.itensor(press_traj,Index(128),Index(128), Index(nsteps))
 println(size(A))
error_SEQRCS = Vector{Float64}()
error_leverage = Vector{Float64}()
error_direct = Vector{Float64}()

for rk in [3,5,7,10,20,30,40,50,100,150,200,250]

    check = ITensorCPD.FitCheck(1e-5, 50,sqrt(sum(A.^2)) )
    r = Index(rk, "CP_rank")
    cp_A = ITensorCPD.random_CPD(A, r,rng=RandomDevice())
    verbose = true
     opt_A = ITensorCPD.als_optimize(A, cp_A; alg = ITensorCPD.direct(), check)
    direct_error =  norm(A - ITensorCPD.reconstruct(opt_A)) / sqrt(sum(A.^2))
    push!(error_direct,direct_error)

    check = ITensorCPD.CPDiffCheck(1e-5, 50)
     int_opt_A =
    ITensorCPD.als_optimize(A, cp_A; alg = ITensorCPD.SEQRCSPivProjected((1,1,1), (400, 400, 400), (1,2,3),(200,200,200)),check, verbose);
    SEQRCS_error =   norm(A - ITensorCPD.reconstruct(int_opt_A)) / sqrt(sum(A.^2))
    println("result for gray using SEQRCS: ",SEQRCS_error)
    push!(error_SEQRCS,SEQRCS_error)

    
    alg = ITensorCPD.LevScoreSampled((400, 400, 400))
     int_opt_A = ITensorCPD.als_optimize(A, cp_A; alg, check, verbose);
    lev_error =  norm(A - ITensorCPD.reconstruct(int_opt_A)) / sqrt(sum(A.^2))
    println("result for gray using leverage method is",lev_error)
    push!(error_leverage,lev_error)



end
plt = plot([3,5,7,10,20,30,40,50,100,150,200,250], error_SEQRCS, marker=:o, label="SEQRCSPivProjected")
plot!([3,5,7,10,20,30,40,50,100,150,200,250], error_leverage, marker=:o, label="LevScoreSampled")
plot!([3,5,7,10,20,30,40,50,100,150,200,250], error_direct, marker=:o, label="Direct")
xlabel!("Rank")
ylabel!("Error")
title!("Error vs rank for gray_scott_reaction_diffusion")
display(plt)



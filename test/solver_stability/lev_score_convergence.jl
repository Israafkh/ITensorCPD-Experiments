include("$(@__DIR__)/../test_env.jl")
#using Makie
using Plots
is = (400, 400, 400)

## Set up the input variables and save location
r = 5
samps = 6
c = 0.2
rguess = 5
basedir = "$(@__DIR__)/../../plots/lev_score_convergence/$(samps)_samples_truerank_$(r)_guessrank_$(rguess)_c_$(c)_mode_1/"
mkdir(basedir)
name = Dict(ITensorCPD.direct => "Normal Equations",
ITensorCPD.LevScoreSampled => "Leverage Score Sampled",
ITensorCPD.BlockLevScoreSampled => "Block Leverage Score Sampled",
ITensorCPD.QRPivProjected => "QR Samples",
ITensorCPD.SEQRCSPivProjected => "SE-QRCS Samples")

function fuse_compute_lev(cpd, inds)
    cpr = ITensorCPD.cp_rank(cpd)
    krp = ITensorCPD.had_contract([cpd[i] for i in inds], cpr)

    fused = Index(prod([dim(ind(cpd, i)) for i in inds]))
    return ITensorCPD.compute_leverage_score_probabilitiy(itensor(array(krp), fused, cpr), fused)
end


## Make the tensor and the exact CPD
T, cpd_exact = Colinearity_Tensor(r, length(is), is, c, nothing, Float64);

lev23 = fuse_compute_lev(cpd_exact, (2,3))

plot(sort(lev23), title="True Leverage Scores",legend=nothing, marker=:circle)
savefig("$(basedir)true_lev_scores.pdf")

using Random

cpd = ITensorCPD.random_CPD(T, rguess; rng=RandomDevice());
check=ITensorCPD.FitCheck(1e-3, 100, norm(T))
first_guess = deepcopy(cpd)

levappx = fuse_compute_lev(cpd, (2,3))
plot(levappx)
algs = [
#    ITensorCPD.direct(), 
        ITensorCPD.QRPivProjected(1, samps), 
        #ITensorCPD.SEQRCSPivProjected(1, samps, (1,2,3), (50,50,50)),
        #ITensorCPD.LevScoreSampled(samps),
        # ITensorCPD.BlockLevScoreSampled(samps,4)
        ]
for alg in algs
    als = ITensorCPD.compute_als(T, cpd_exact; alg, check, trunc_tol=1e-10);

    ps = []
    ## First run
    upcpd, upals = single_solve(als, cpd, 1);
    up1 = fuse_compute_lev(upcpd, (2,3));
    upcpd, upals = single_solve(upals, upcpd, 2);
    # up2 = fuse_compute_lev(upcpd, (2,3));
    upcpd, upals = single_solve(upals, upcpd, 3);
    # up3 = fuse_compute_lev(upcpd, (2,3));

    ups1 = [up1]
    # ups2 = [up2]
    # ups3 = [up3]

    for i in 1:40
        upcpd, upals = single_solve(upals, upcpd, 1);
        up1 = fuse_compute_lev(upcpd, (2,3));
        upcpd, upals = single_solve(upals, upcpd, 2);
        # up2 = fuse_compute_lev(upcpd, (2,3));
        upcpd, upals = single_solve(upals, upcpd, 3);
        # up3 = fuse_compute_lev(upcpd, (2,3));
        push!(ups1, up1)
        # push!(ups2, up2)
        # push!(ups3, up3)

        # p = plot((sort(up1) - sort(lev23)) ./ sort(lev23))
        # plot!((sort(up2) - sort(lev23)) ./ sort(lev23))
        # plot!((sort(up3) - sort(lev23)) ./ sort(lev23))
        # push!(ps, p)
        #display(p)
    end

    slev = (lev23)
    #./ slev
    anim = @animate for i ∈ 1:41
        plot(((ups1[i]) - slev) .* 100000 , yrange=[-10, 10], legend=:topright, 
        title="$(name[typeof(alg)]) \n ALS Iteration $(i)")
    end 
    gif(anim, "$(basedir)levs_$(typeof(alg))_resamp.gif", fps = 5)
end
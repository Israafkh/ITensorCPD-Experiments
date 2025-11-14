include("$(@__DIR__)/../test_env.jl")
using Makie
#using Plots
is = (200, 300, 400)

T, cpd_exact = Colinearity_Tensor(20, length(is), is, 0.2, nothing, Float64);

name = Dict(ITensorCPD.direct => "Normal Equations",
ITensorCPD.LevScoreSampled => "Leverage Score Sampled",
ITensorCPD.QRPivProjected => "QR Samples",
ITensorCPD.SEQRCSPivProjected => "SE-QRCS Samples")
# array(T) .+= 0.0001 .* randn(size(T))

#cpd_exact = ITensorCPD.decompose(T, 300; 
#    check=ITensorCPD.FitCheck(1e-4, 100, norm(T)), verbose=true);

function fuse_compute_lev(cpd, inds)
    cpr = ITensorCPD.cp_rank(cpd)
    krp = ITensorCPD.had_contract([cpd[i] for i in inds], cpr)

    fused = Index(prod([dim(ind(cpd, i)) for i in inds]))
    return ITensorCPD.compute_leverage_score_probabilitiy(itensor(array(krp), fused, cpr), fused)
end

lev23 = fuse_compute_lev(cpd_exact, (2,3))

plot(sort(lev23), title="True Leverage Scores",)
savefig("plots/true_lev_scores")

cpd = ITensorCPD.random_CPD(T, 20);
check=ITensorCPD.FitCheck(1e-3, 100, norm(T))


#alg=ITensorCPD.direct()
#alg = ITensorCPD.QRPivProjected(1, 80)
#alg = ITensorCPD.SEQRCSPivProjected(1, 80, (1,2,3))
alg = ITensorCPD.LevScoreSampled(80)
als = ITensorCPD.compute_als(T, cpd; alg, check);

ps = []
## First run
upcpd, upals = single_solve(als, cpd, 1);
up1 = fuse_compute_lev(upcpd, (2,3));
upcpd, upals = single_solve(upals, upcpd, 2);
up2 = fuse_compute_lev(upcpd, (2,3));
upcpd, upals = single_solve(upals, upcpd, 3);
up3 = fuse_compute_lev(upcpd, (2,3));

xs = [(i - sort(lev23)) ./ sort(lev23) for i in [up1, up2, up3]]

ups1 = [up1]
ups2 = [up2]
ups3 = [up3]

#lines(xs)

# p = plot((sort(up1) - sort(lev23)) ./ sort(lev23))
# plot!((sort(up2) - sort(lev23)) ./ sort(lev23))
# plot!((sort(up3) - sort(lev23)) ./ sort(lev23))
# push!(ps, p)



for i in 1:20
    upcpd, upals = single_solve(upals, upcpd, 1);
    up1 = fuse_compute_lev(upcpd, (2,3));
    upcpd, upals = single_solve(upals, upcpd, 2);
    up2 = fuse_compute_lev(upcpd, (2,3));
    upcpd, upals = single_solve(upals, upcpd, 3);
    up3 = fuse_compute_lev(upcpd, (2,3));
    push!(ups1, up1)
    push!(ups2, up2)
    push!(ups3, up3)

    # p = plot((sort(up1) - sort(lev23)) ./ sort(lev23))
    # plot!((sort(up2) - sort(lev23)) ./ sort(lev23))
    # plot!((sort(up3) - sort(lev23)) ./ sort(lev23))
    # push!(ps, p)
    #display(p)
end

slev = sort(lev23)
anim = @animate for i ∈ 1:21
    plot((sort(ups1[i]) - slev) ./ slev, yrange=[-0.6, 0.6], legend=:topright, 
    title="$(name[typeof(alg)]) \n ALS Iteration $(i)")
end 
gif(anim, "plots/levs_$(typeof(alg))_23.gif", fps = 5)

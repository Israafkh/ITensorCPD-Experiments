include("$(@__DIR__)/read_coil.jl")

path = "$(@__DIR__)/coil-100/"
image_names = readdir(path)

coil = readcoil(path, 20, Float64);

ncoil = sqrt(sum(hadamard_product(coil,coil)))
rng = RandomDevice()
# Computes the CP-ALS using an efficient version of the normal equations.
# Stops when change in fit falls below 1e-3 
algNormal = ITensorCPD.direct()
check_exact = ITensorCPD.FitCheck(1e-5, 100, ncoil)
## Create a random initial guess with rank 20
init_guess = ITensorCPD.random_CPD(coil, 20; rng);
# opt = ITensorCPD.als_optimize(coil, init_guess; verbose=true, alg=algNormal, check=check_exact);
# check_exact.final_fit

## Compute a sampled CP-ALS using random single samples of the khatri-rao product.
check = ITensorCPD.CPDiffCheck(1e-2, 100);
## Using 100 samples
nsamples = 1000
als_list = [
    ITensorCPD.compute_als(coil, init_guess; alg=alg, check, trunc_tol = 0.1, shuffle_pivots=shuffle)
    for (alg, shuffle) in zip([
        ITensorCPD.QRPivProjected(nsamples),
        ITensorCPD.QRPivProjected(nsamples),
        ITensorCPD.SEQRCSPivProjected(1,1000,(1,2,3,4),(100,1,1,1)),
        ITensorCPD.SEQRCSPivProjected(1,1000,(1,2,3,4),(100,1,1,1)),
        ITensorCPD.LevScoreSampled(nsamples),
    ], 
    [true,false,true,false,true])
];

rks = [20, 50, 100, 200, 300]
fitQR, fitQRNS, fitSE, fitSENS, fitRand = [],[],[],[],[]
fits = [
    fitQR,
    fitQRNS,
    fitSE,
    fitSENS,
    fitRand
]
for rk in rks
    r = Index(rk, "CP")
    guess = ITensorCPD.random_CPD(coil, r; rng)
    for ( als, fit) in zip(als_list, fits)
        @time opt = ITensorCPD.optimize(guess, als; verbose=true);
        push!(fit, check_fit(coil, opt))
        GC.gc()
    end
end

lw = 5
plot(rks, fitRand, label="Leverage Scores"; lw)
plot!(rks, fitQR, label="QR"; lw)
plot!(rks, fitSE, label="SE-QRCS"; lw)
plot!(legend=:bottomleft,
title="20 Image COIL data, 1000 Samples", 
xlabel="Rank", 
legendtitle="Sampling Method",
legendtitlefontsize=10,
legendfontsize=10,
titlefontsize=15,
labelfontsize=15,
tickfontsize=12,
ylabel="Fit")
savefig("$(@__DIR__)/../../plots/coil/shuffle_rank_scan_samples_$(nsamples).pdf")

plot(rks, fitRand, label="Leverage Scores";lw)
plot!(rks, fitQRNS, label="QR"; lw)
plot!(rks, fitSENS, label="SE-QRCS";lw)
plot!(legend=:bottomleft,
title="20 Image COIL data, 1000 Samples", 
xlabel="Rank", 
legendtitle="Sampling Method",
legendtitlefontsize=10,
legendfontsize=10,
titlefontsize=15,
labelfontsize=15,
tickfontsize=12,
ylabel="Fit")
savefig("$(@__DIR__)/../../plots/coil/no_shuffle_rank_scan_samples_$(nsamples).pdf")



###### Scan samples fixed rank
rks = [20, 50, 100, 200, 300]
samps = [400:50:1000...]
fitQR, fitQRNS, fitSE, fitSENS, fitRand = [],[],[],[],[]
fits = [
    fitQR,
    fitQRNS,
    fitSE,
    fitSENS,
    fitRand
]

rk = 20
r = Index(rk, "CP")
guess = ITensorCPD.random_CPD(coil, r; rng)
als_list = [
    ITensorCPD.compute_als(coil, init_guess; alg=alg, check, trunc_tol = 0.1, shuffle_pivots=shuffle)
    for (alg, shuffle) in zip([
        ITensorCPD.QRPivProjected(nsamples),
        ITensorCPD.QRPivProjected(nsamples),
        ITensorCPD.SEQRCSPivProjected(1,1000,(1,2,3,4),(100,1,1,1)),
        ITensorCPD.SEQRCSPivProjected(1,1000,(1,2,3,4),(100,1,1,1)),
        ITensorCPD.LevScoreSampled(nsamples),
    ], 
    [true,false,true,false,true])
    ];

for nsamples in samps
    als_list[1] = ITensorCPD.update_samples(coil, als_list[1], nsamples);
    als_list[2] = ITensorCPD.update_samples(coil, als_list[2], nsamples);
    als_list[3] = ITensorCPD.update_samples(coil, als_list[3], nsamples);
    als_list[4] = ITensorCPD.update_samples(coil, als_list[4], nsamples);
    als_list[5] = ITensorCPD.compute_als(coil, init_guess; alg=ITensorCPD.LevScoreSampled(nsamples), check =ITensorCPD.CPDiffCheck(1e-2, 100) )
    for ( als, fit) in zip(als_list, fits)
        opt = ITensorCPD.optimize(guess, als; verbose=true);
        push!(fit, check_fit(coil, opt))
    end
end

plot(samps, fitRand, label="Leverage Scores"; lw)
plot!(samps, fitQR, label="QR"; lw)
plot!(samps, fitSE, label="SE-QRCS"; lw)
plot!(legend=:bottomright,
title="20 Image COIL data, CP Rank: 20", 
xlabel="Samples", 
legendtitle="Sampling Method",
legendtitlefontsize=10,
legendfontsize=10,
titlefontsize=15,
labelfontsize=15,
tickfontsize=12,
ylabel="Fit")
savefig("$(@__DIR__)/../../plots/coil/shuffle_sample_scan_rank_$(rk).pdf")

plot(samps, fitRand, label="Leverage Scores"; lw)
plot!(samps, fitQRNS, label="QR"; lw)
plot!(samps, fitSENS, label="SE-QRCS"; lw)
plot!(legend=:bottomright,
title="20 Image COIL data, CP Rank: 20", 
xlabel="Samples", 
legendtitle="Sampling Method",
legendtitlefontsize=10,
legendfontsize=10,
titlefontsize=15,
labelfontsize=15,
tickfontsize=12,
ylabel="Fit")
savefig("$(@__DIR__)/../../plots/coil/no_shuffle_sample_scan_rank_$(rk).pdf")
include("$(@__DIR__)/../test_env.jl")
using LinearAlgebra, Statistics, DataFrames, CategoricalArrays, StatsPlots
using Random

elt = Float64 
i, j, k = Index.((90, 90, 90))
r = Index(100, "CP_rank")
rng = RandomDevice()
T = random_itensor(rng, elt, i, j, k)
cpd = ITensorCPD.random_CPD(T, r; rng)
bad = 40

# array(cpd[1])[1:bad, :] .*= 100;
# array(cpd[2])[1:bad, :] .*= 100;
# array(cpd[3])[1:bad, :] .*= 100;
# array(cpd[1]) .= array(cpd[1])[randperm(dim(i)), :]
# array(cpd[2]) .= array(cpd[2])[randperm(dim(j)), :]
# array(cpd[3]) .= array(cpd[3])[randperm(k), :]
# r = ITensorCPD.cp_rank(cpd);
# T = had_contract(cpd[1], cpd[2], r) * had_contract(cpd[3], cpd[], r)

## Old method (unknown rank)
T = ITensorCPD.reconstruct(cpd)
T1 =reshape(array(T, (i, j, k)), (dim(i), dim(j)*dim(k)))
T1[:,1:bad].*=100
T1 = T1[:,randperm(dim(j) * dim(k))]
T= itensor(T1,i,j,k)
verbose= true

#samples = [400,500,600,800,1000,1200,1500,2000]
samples = [200, 400, 600, 800, 1000, 1500, 2000]
check_piv = ITensorCPD.CPDiffCheck(1e-2, 100)
check_direct = ITensorCPD.FitCheck(1e-5, 50,sqrt(sum(T.^2)) )
SEQRCS_error_mat = Matrix{Float64}(undef, 20, length(samples))
lev_error_mat = Matrix{Float64}(undef, 20, length(samples))
rng=RandomDevice()
cp_T = nothing
for rk in [90, 100, 110]    
    r = Index(rk, "CP_rank")
    err_SEQRCS = Vector{Float64}()
    err_leverage = Vector{Float64}()
    cp_T = ITensorCPD.random_CPD(T,r; rng)

    SEQRCS_error_mat = Matrix{Float64}(undef, 20, length(samples))
    lev_error_mat = Matrix{Float64}(undef, 20, length(samples))
    SEQRCS_error_vect = Vector{Float64}(undef, 0)
    for (s, i) in zip(samples[1:end], 1:length(samples))
        SEQRCS_error_vect = Vector{Float64}()
        @show s
        for q = 1:(20 - length(SEQRCS_error_vect))
            success = false
            while !success
                try
                    alsQR = ITensorCPD.compute_als(T,cp_T; alg = ITensorCPD.SEQRCSPivProjected(1, s, (1,2,3), (90,)),check = check_piv, injective=false);
                    @show alsQR.mttkrp_alg
                    int_opt_T =
                    ITensorCPD.optimize(cp_T,alsQR;verbose=true);
                    push!(SEQRCS_error_vect,check_fit(T, int_opt_T))
                    success = true
                catch
                end
            end
        end
        SEQRCS_error_mat[:,i] .= SEQRCS_error_vect
        
        lev_error_vect = Vector{Float64}()
        for q=1:20
            success = false
            while !success
                try
                    alslev = ITensorCPD.compute_als(T,cp_T;alg = ITensorCPD.LevScoreSampled(s),check = check_piv)
                    int_opt_T = ITensorCPD.optimize(cp_T, alslev; verbose=false)
                    push!(lev_error_vect,check_fit(T, int_opt_T))
                    success = true
                catch
                end
            end
        end
        lev_error_mat[:,i] .= lev_error_vect
    end
    
    err_SEQRCS = median.(eachcol(SEQRCS_error_mat))
    err_leverage = median.(eachcol(lev_error_mat))

    alsNormal = ITensorCPD.compute_als(T, cp_T; alg=ITensorCPD.direct(), check = check_direct);
    opt_T = ITensorCPD.optimize(cp_T, alsNormal; verbose);
    direct_error = check_fit(alsNormal, opt_T.factors, r, opt_T.λ, 1)

    ms = 7
    lw = 4
    plt2 = plot(samples, direct_error .* ones(length(samples)), marker=:o, label="Normal Equations"; ms, lw)
    plot!(samples, err_SEQRCS, marker=:o, label="SE-QRCS Sampling"; ms, lw)
    plot!(samples, err_leverage, marker=:o, label="Leverage Score Sampling"; ms, lw)
    plot!(legendtitle="ALS Method",
    title="Modified Synthetic Tensor Test:\n Rank $(dim(r))",
    xlabel="Number of Samples",
    ylabel = "CPD Fit",
     yrange=[-0.6,1.01], 
     yticks=-0.5:0.2:1.01,
    legendtitlefontsize=10,
    legendfontsize=10,
    labelfontsize=15,
    titlefontsize=14,
    tickfontsize=10,
    xticks=samples[1]:200:samples[end],
    )


    n = nothing
    if elt == Float64
        n = "F64"
    else
        n = "F32"
    end

    savefig("$(@__DIR__)/../../plots/synthetic_tensor/rank_$(rk)_modified_test_$(n).pdf")

    # display(plt2)

    # ## Violin Plot of modified tensor
    df = DataFrame(
        sample = vcat(repeat(samples, inner=20), repeat(samples, inner=20)),
        method = vcat(fill("SE-QRCS", 20*length(samples)),
                    fill("Leverage", 20*length(samples))),
        error  = error  = vcat(vec(SEQRCS_error_mat), vec(lev_error_mat))
    )

    df.sample = CategoricalArray(string.(df.sample), ordered=true, levels=string.(samples))

    df_SEQRCS = filter(row -> row.method == "SE-QRCS", df)
    df_lev  = filter(row -> row.method == "Leverage", df)

    p = @df df_SEQRCS violin(:sample, :error, side=:left, label="SE-QRCS", color=:blue)
    @df df_lev  violin!(:sample, :error, side=:right, label="Leverage", color=:orange)
    xlabel!("Number of Samples")
    ylabel!("Distribution CPD Fit")
    title!("Modified Synthetic tensor test: \n Rank $rk")
    plot!(legend=:left, yticks=-0.5:0.1:1.01, yrange=[-0.5, 1.0])
    savefig("$(@__DIR__)/../../plots/synthetic_tensor/distribution_rank_$(rk)_modified_test_$(n).pdf")
end

c = ITensorCPD.random_CPD(T, 300; rng)
alsNormal = ITensorCPD.compute_als(T, c; alg=ITensorCPD.InvKRP(), check = check_direct);
ITensorCPD.decompose(T, 1200; rng,  alg=ITensorCPD.InvKRP(), check=ITensorCPD.FitCheck(1e-3, 20, norm(T)), verbose=true);
direct_error = check_fit(alsNormal, opt_T.factors, r, opt_T.λ, 1)

rk = 90

using ITensorCPD: reconstruct
for s in [200, 1000, 2000, 5000, 10000]
    r = Index(rk, "CP_rank")
    cp_T = ITensorCPD.random_CPD(T,r; rng)
    alsQR = ITensorCPD.compute_als(T,cp_T; alg = ITensorCPD.SEQRCSPivProjected(1, s, (1,2,3),(90,)),check = check_piv, normal=false);
    int_opt_T_1 = ITensorCPD.optimize(cp_T,alsQR;verbose=false);
    check_fit(T, int_opt_T_1)
    diff = T - reconstruct(int_opt_T_1)
    @show dot(diff, diff)

    alsQRNormal = deepcopy(alsQR);
    alsQRNormal.additional_items[:normal]=true
    int_opt_T_2 = ITensorCPD.optimize(cp_T,alsQRNormal;verbose=false);
    check_fit(T, int_opt_T_2)
    diff = T - reconstruct(int_opt_T_1)
    @show dot(diff, diff)

    #@show norm(reconstruct(int_opt_T_1) - reconstruct(int_opt_T_2)) / norm(reconstruct(int_opt_T_1))
    # alslev = ITensorCPD.compute_als(T,cp_T;alg = ITensorCPD.LevScoreSampled(s),check = check_piv, normal=false);
    # int_opt_T = ITensorCPD.optimize(cp_T, alslev; verbose=false)
    # check_fit(T, int_opt_T)

    # alslevNormal = deepcopy(alslev)
    # alslevNormal.additional_items[:normal]=true
    # int_opt_T = ITensorCPD.optimize(cp_T, alslevNormal; verbose=false)
    # check_fit(T, int_opt_T)
end


### Table in performance results.
samples = [200, 1000, 2000]
qr = [0.3778247829226412, 0.7218399358095993, 0.6734058439142732,]
qrn = [0.4941730088897982,  0.7218399212895974, 0.673405843916232,]
lev = [-0.45470748554161444, 0.16607021233282293, 0.2650053205364551]
levn = [-0.3647305183604339, 0.19231770560445416, 0.31380912923128035]

### This is for the Timing of SE-QRCS ALS
using ITensors, ITensorCPD, Random
include("../test_env.jl")

# rng = RandomDevice();
# i,j,k = 500, 500, 500;
# tr = ITensorCPD.random_CPD(ITensor(Float64, Index.((i,j,k))), 200; rng);
# rank = ITensorCPD.cp_rank(tr);
# A = ITensorCPD.had_contract(tr[1], tr[2], rank) * ITensorCPD.had_contract(tr[3], tr[], rank)

r = 1
cpd = ITensorCPD.random_CPD(A, r; rng);

s = 1
check = ITensorCPD.FitCheck(1e-3, 20, norm(A));
alsQR = ITensorCPD.compute_als(A, cpd; alg=ITensorCPD.SEQRCSPivProjected(1, 5, (1,2,3), (10,)), check, normal = false);
alsQRn = deepcopy(alsQR);
alsQRn.additional_items[:normal] = true

times = []
timesn = []
vals = []
valsn = []
ranks = [10, 50, 100, 150, 200]
samples = [10000, 20000, 30000, 40000, 50000]
for r in ranks
    cpd = ITensorCPD.random_CPD(A, r; rng);
    time = Vector{Float64}()
    timen = Vector{Float64}()
    val = Vector{Float64}()
    valn = Vector{Float64}()
    for s in samples
        alsQR = ITensorCPD.update_samples(A, alsQR, s);
        alsQRn = ITensorCPD.update_samples(A, alsQRn, s);
        push!(time, @elapsed opt = ITensorCPD.optimize(cpd, alsQR));
        push!(timen, @elapsed optn = ITensorCPD.optimize(cpd, alsQRn));

        push!(val, check_fit(A, opt))
        push!(valn, check_fit(A, optn))
    end
    push!(times, time)
    push!(timesn, timen)
    push!(vals, val)
    push!(valsn, valn)
end

colors = [:blue, :red, :black, :indigo, :brown]
p = plot()
for (i, j, r, c) in zip(times, timesn, ranks, colors)
    plot!(samples, i./20, label="Rank $(r)", lc=c)
    plot!(samples, i./20, label=nothing, lc=c, marker=:circle)
    plot!(samples, j./20, label=nothing, lc=c, marker=:square)
end
plot!(ylabel="Time per ALS iteration (s)", xlabel="Number of Samples", 
title="CP-ALS Optimization Time")

savefig("NormalEqationSamplesTime.pdf")

colors = [:blue, :red, :black, :indigo, :brown]
p = plot()
for (i, j, r, c) in zip(vals, valsn, ranks, colors)
    plot!(samples, i - j, label="Rank $(r)", lc=c, marker=:circle)
end
plot!(ylabel="Difference in CPD Fits", xlabel="Number of Samples", 
title="Difference in CPD fits",
legend=nothing)

savefig("NormalEqationSamplesTime.pdf")
include("test_env.jl")
using LinearAlgebra, Statistics, DataFrames, CategoricalArrays, StatsPlots
## Generating a random tensor with CPD rank greater than dimension

for elt in (Float32,Float64), c in [0.2, 0.8]
    T, cp = Collinearity_Tensor(90, 3, (300,300,300), elt(c), nothing, elt)
    verbose= true
    samples = [1, 50, 150, 300, 500, 1000, 1300,1500]
    check_piv = ITensorCPD.CPDiffCheck(1e-5, 50)
    check_direct = ITensorCPD.FitCheck(1e-5, 50,sqrt(sum(T.^2)) )
    nrepeats = 5
    for rk in [80,90,100]
        rng=RandomDevice()
        r = Index(rk, "CP_rank")
        cp_T = ITensorCPD.random_CPD(T,r;rng)

        SEQRCS_error_vect = Vector{Float64}()
        err_SEQRCS = Vector{Float64}()
        ## SE-QRCS 
        for s in samples
            for q = 1:nrepeats
                alg = ITensorCPD.SEQRCSPivProjected(1, s, (1,2,3),(5,5,5))
                alsQR = ITensorCPD.compute_als(T,cp_T;alg, check =check_piv);
                int_opt_T =
                ITensorCPD.optimize(cp_T,alsQR;verbose);
                push!(SEQRCS_error_vect,check_fit(alsQR, int_opt_T.factors, r, int_opt_T.λ, 1))
            end
        end

        err_leverage = Vector{Float64}()
        lev_error_vect = Vector{Float64}()
        ## Leverage Score 
        for s in samples
            for q=1:nrepeats
                alslev = ITensorCPD.compute_als(T,cp_T;alg = ITensorCPD.LevScoreSampled(s),check = check_piv)
                int_opt_T = ITensorCPD.optimize(cp_T, alslev; verbose)
                push!(lev_error_vect,check_fit(alslev, int_opt_T.factors, r, int_opt_T.λ, 1))
            end

        end
    
        SEQRCS_error_mat = reshape(SEQRCS_error_vect, (nrepeats,length(samples)))
        lev_error_mat = reshape(lev_error_vect, (nrepeats,length(samples)))
        
        err_SEQRCS = median.(eachcol(SEQRCS_error_mat))
        err_leverage = median.(eachcol(lev_error_mat))

        err_direct = Vector{Float64}()
        alsNormal = ITensorCPD.compute_als(T, cp_T; alg=ITensorCPD.direct(), check =check_direct)
        opt_T = ITensorCPD.optimize(cp_T, alsNormal; verbose)
        direct_error = check_fit(alsNormal, opt_T.factors, r, opt_T.λ, 1)
        push!(err_direct,direct_error)

        plt2 = plot(samples[2:end], err_SEQRCS[2:end], marker=:o, label="SEQRCSPivProjected")
        plot!(samples[2:end], err_leverage[2:end], marker=:o, label="LevScoreSampled")
        plot!(samples[2:end], direct_error .* ones(length(samples[2:end])), marker=:o, label="Direct")

        xlabel!("Number of Samples")
        ylabel!("Error in CPD Fit")
        title!("Synthetic Tensor Test:\n Rank $rk")


        n = nothing
        if elt == Float64
            n = "F64"
        else
            n = "F32"
        end
        savefig("$(@__DIR__)/../plots/synthetic_tensor/rank_$(rk)_test_$(n)_colin_$(c).pdf")
        display(plt2)

        #Violin plot of synthetic tensor
        df = DataFrame(
            sample = vcat(repeat(samples, inner=nrepeats), repeat(samples, inner=nrepeats)),
            method = vcat(fill("SE-QRCS", nrepeats*length(samples)),
                        fill("Leverage", nrepeats*length(samples))),
            error  = vcat(SEQRCS_error_vect, lev_error_vect)
        )

        df.sample = CategoricalArray(string.(df.sample), ordered=true, levels=string.(samples))

        df_SEQRCS = filter(row -> row.method == "SE-QRCS", df)
        df_lev  = filter(row -> row.method == "Leverage", df)

        p = @df df_SEQRCS violin(:sample, :error, side=:left, label="SE-QRCS", color=:blue, yscale=:log10)
        @df df_lev  violin!(:sample, :error, side=:right, label="Leverage", color=:orange)
        xlabel!("Sampling size")
        ylabel!("Error distribution in CPD fit")
        title!("Synthetic tensor test: \n Rank $rk")
        legend=:topleft
        display(p)

    end
end

## Generating a random tensor with certain columns in matricized
## matrix along certain mode is amplified

for elt in [Float32,Float64]
    i, j, k = Index.((90, 90, 90))
    r = Index(100, "CP_rank")
    T = random_itensor(elt, i, j, k)
    cp = ITensorCPD.random_CPD(T, r)
    T = ITensorCPD.reconstruct(cp)
    T1 =reshape(array(T, (i, j, k)), (dim(i), dim(j)*dim(k)))
    T1[:,1:40].*=100
    T= itensor(T1,i,j,k)
    verbose= false
    samples = [400,500,600,700,800,900,1000,1100,1200,1300]
    check_piv = ITensorCPD.CPDiffCheck(1e-5, 50)
    check_direct = ITensorCPD.FitCheck(1e-5, 50,sqrt(sum(T.^2)) )
    for rk in [90,100,110]    
        r = Index(rk, "CP_rank")
        err_SEQRCS = Vector{Float64}()
        err_leverage = Vector{Float64}()
        cp_T = ITensorCPD.random_CPD(T,r,rng=RandomDevice())

        SEQRCS_error_vect = Vector{Float64}()
        lev_error_vect = Vector{Float64}()
        for s in samples
            for q = 1:50
                alsQR = ITensorCPD.compute_als(T,cp_T; alg = ITensorCPD.SEQRCSPivProjected(1, s, (1,2,3),(90,90,90)),check = check_piv)
                int_opt_T =
                ITensorCPD.optimize(cp_T,alsQR;verbose)
                push!(SEQRCS_error_vect,check_fit(alsQR, int_opt_T.factors, r, int_opt_T.λ, 1))
            end
            

            for q=1:50
                alslev = ITensorCPD.compute_als(T,cp_T;alg = ITensorCPD.LevScoreSampled(s),check = check_piv)
                int_opt_T = ITensorCPD.optimize(cp_T, alslev; verbose)
                push!(lev_error_vect,check_fit(alslev, int_opt_T.factors, r, int_opt_T.λ, 1))
            end
        end

        SEQRCS_error_mat = reshape(SEQRCS_error_vect, (50,length(samples)))
        lev_error_mat = reshape(lev_error_vect, (50,length(samples)))
        
        err_SEQRCS = median.(eachcol(SEQRCS_error_mat))
        err_leverage = median.(eachcol(lev_error_mat))

        alsNormal = ITensorCPD.compute_als(T, cp_T; alg=ITensorCPD.direct(), check = check_direct)
        opt_T = ITensorCPD.optimize(cp_T, alsNormal; verbose)
        direct_error = check_fit(alsNormal, opt_T.factors, r, opt_T.λ, 1)

        plt2 = plot(samples, err_SEQRCS, marker=:o, label="SEQRCSPivProjected")
        plot!(samples, err_leverage, marker=:o, label="LevScoreSampled")
        plot!(samples, direct_error .* ones(length(samples)), marker=:o, label="Direct")

        xlabel!("Number of Samples")
        ylabel!("Error in CPD Fit")
        title!("Modified Synthetic Tensor Test:\n Rank $rk")

        n = nothing
        if elt == Float64
            n = "F64"
        else
            n = "F32"
        end

        savefig("$(@__DIR__)/../plots/synthetic_tensor/rank_$(rk)_modified_test_$(n).pdf")

        display(plt2)

        ## Violin Plot of modified tensor
        df = DataFrame(
            sample = vcat(repeat(samples, inner=50), repeat(samples, inner=50)),
            method = vcat(fill("SE-QRCS", 50*length(samples)),
                        fill("Leverage", 50*length(samples))),
            error  = error  = vcat(SEQRCS_error_vect, lev_error_vect)
        )

        df.sample = CategoricalArray(string.(df.sample), ordered=true, levels=string.(samples))

        df_SEQRCS = filter(row -> row.method == "SE-QRCS", df)
        df_lev  = filter(row -> row.method == "Leverage", df)

        p = @df df_SEQRCS violin(:sample, :error, side=:left, label="SE-QRCS", color=:blue, yscale=:log10)
        @df df_lev  violin!(:sample, :error, side=:right, label="Leverage", color=:orange)
        xlabel!("Sampling size")
        ylabel!("Error Distribution in CPD fit")
        title!("Modified Synthetic tensor test: \n Rank $rk")
        legend=:topleft
        display(p)
    end
end


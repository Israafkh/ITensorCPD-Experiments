include("test_env.jl")
using LinearAlgebra, Statistics
## Generating a random tensor with CPD rank greater than dimension

for elt in [Float32,Float64]
    i, j, k = Index.((90, 90, 90))
    r = Index(100, "CP_rank")
    T = random_itensor(elt, i, j, k)
    cp = ITensorCPD.random_CPD(T, r)
    T = ITensorCPD.reconstruct(cp)
    verbose= false
    samples = [400,500,600,700,800,900,1000,1100,1200,1300]
    check_piv = ITensorCPD.CPDiffCheck(1e-5, 50)
    check_direct = ITensorCPD.FitCheck(1e-5, 50,sqrt(sum(T.^2)) )
    for rk in [90,100,110]
        r = Index(rk, "CP_rank")
        err_SEQRCS = Vector{Float64}()
        err_leverage = Vector{Float64}()
        err_direct = Vector{Float64}()
        cp_T = ITensorCPD.random_CPD(T,r,rng=RandomDevice())
        for s in samples
            SEQRCS_error_vect = Vector{Float64}()
            lev_error_vect = Vector{Float64}()
            for q = 1:10
                alg = ITensorCPD.SEQRCSPivProjected(1, s, (1,2,3),(5,5,5))
                alsQR = ITensorCPD.compute_als(T,cp_T;alg, check =check_piv);
                int_opt_T =
                ITensorCPD.optimize(cp_T,alsQR;verbose);
                push!(SEQRCS_error_vect,check_fit(alsQR, int_opt_T.factors, r, int_opt_T.λ, 1))
            end
            # SEQRCS_error = median(SEQRCS_error_vect) 
            # push!(err_SEQRCS,SEQRCS_error)

            for q=1:10
                alslev = ITensorCPD.compute_als(T,cp_T;alg = ITensorCPD.LevScoreSampled(s),check = check_piv)
                int_opt_T = ITensorCPD.optimize(cp_T, alslev; verbose)
                push!(lev_error_vect,check_fit(alslev, int_opt_T.factors, r, int_opt_T.λ, 1))
            end

            # Lev_error = median(lev_error_vect)
            # push!(err_leverage,Lev_error)
        end
    
        SEQRCS_error_mat = reshape(SEQRCS_error_vect, (10,length(samples)))
        lev_error_mat = reshape(lev_error_vect, (10,length(samples)))
        
        err_SEQRCS = median.(eachcol(SEQRCS_error_mat))
        err_leverage = median.(eachcol(lev_error_mat))

        alsNormal = ITensorCPD.compute_als(T, cp_T; alg=ITensorCPD.direct(), check =check_direct)
        opt_T = ITensorCPD.optimize(cp_T, alsNormal; verbose)
        direct_error = check_fit(alsNormal, opt_T.factors, r, opt_T.λ, 1)
        push!(err_direct,direct_error)

        plt2 = plot(samples, err_SEQRCS, marker=:o, label="SEQRCSPivProjected")
        plot!(samples, err_leverage, marker=:o, label="LevScoreSampled")
        plot!(samples, direct_error .* ones(length(samples)), marker=:o, label="Direct")

        xlabel!("Number of Samples")
        ylabel!("Error in CPD Fit")
        title!("Synthetic Tensor Test:\n Rank $rk")
        n = nothing
        if elt == Float64
            n = "F64"
        else
            n = "F32"
        end
        savefig("$(@__DIR__)/../plots/synthetic_tensor/rank_$(rk)_test_$(n).pdf")
        display(plt2)

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
        err_direct = Vector{Float64}()
        cp_T = ITensorCPD.random_CPD(T,r,rng=RandomDevice())
        for s in samples

            SEQRCS_error_vect = Vector{Float64}()
            lev_error_vect = Vector{Float64}()
            for q = 1:10
                alsQR = ITensorCPD.compute_als(T,cp_T; alg = ITensorCPD.SEQRCSPivProjected(1, s, (1,2,3),(50,50,50)),check = check_piv)
                int_opt_T =
                ITensorCPD.optimize(cp_T,alsQR;verbose)
                push!(SEQRCS_error_vect,check_fit(alsQR, int_opt_T.factors, r, int_opt_T.λ, 1))
            end
            SEQRCS_error = median(SEQRCS_error_vect) 
            push!(err_SEQRCS,SEQRCS_error)

            for q=1:10
                alslev = ITensorCPD.compute_als(T,cp_T;alg = ITensorCPD.LevScoreSampled(s),check = check_piv)
                int_opt_T = ITensorCPD.optimize(cp_T, alslev; verbose)
                push!(lev_error_vect,check_fit(alslev, int_opt_T.factors, r, int_opt_T.λ, 1))
            end

            Lev_error = median(lev_error_vect)
            push!(err_leverage,Lev_error)

            alsNormal = ITensorCPD.compute_als(T, cp_T; alg=ITensorCPD.direct(), check = check_direct)
            opt_T = ITensorCPD.optimize(cp_T, alsNormal; verbose)
            direct_error = check_fit(alsNormal, opt_T.factors, r, opt_T.λ, 1)
            push!(err_direct,direct_error)
        end
    

        plt2 = plot(samples, err_SEQRCS, marker=:o, label="SEQRCSPivProjected")
        plot!(samples, err_leverage, marker=:o, label="LevScoreSampled")
        plot!(samples, err_direct, marker=:o, label="Direct")

        xlabel!("Sampling size")
        ylabel!("Error")
        title!("Modified Synthetic tensor with rank $rk")
        display(plt2)
    end
end


include("test_env.jl")
using LinearAlgebra, Statistics, DataFrames, CategoricalArrays, StatsPlots
## Generating a random tensor with CPD rank greater than dimension

for elt in (Float32,Float64), c in [0.2, 0.8]
    A, cp = Colinearity_Tensor(90, (400,400,400), elt(c), nothing, elt)
    ind_A = inds(A)  
    dims = dim.(ind_A)
    C = randn(dims...)
    N = itensor(C,ind_A...)
    T = A+ 0.1*(norm(A)/norm(N))*N
    verbose= true
    samples = [1, 50, 150, 300, 500, 1000, 1300,1500]
    check_piv = ITensorCPD.CPDiffCheck(1e-5, 50)
    check_direct = ITensorCPD.FitCheck(1e-5, 50,sqrt(sum(T.^2)) )
    nrepeats = 1
    for rk in [80,90,100]
        rng=RandomDevice()
        r = Index(rk, "CP_rank")
        cp_T = ITensorCPD.random_CPD(T,r;rng)
        
        SEQRCS_error_vect = Vector{Float64}()
        SEQRCS_error_vect_HOSVD = Vector{Float64}()
        SEQRCS_error_vect_High = Vector{Float64}()
        err_SEQRCS = Vector{Float64}()
        err_SEQRCS_HOSVD = Vector{Float64}()
        err_SEQRCS_High = Vector{Float64}()
        ## SE-QRCS 
        for s in samples
            for q = 1:nrepeats
                alg_QR =ITensorCPD.SEQRCSPivProjected(1, s, (1,2,3),(30,30,30))
                alsQR = ITensorCPD.compute_als(T,cp_T;alg=alg_QR, check =check_piv);
                # alsQR.target .= T
                # println("Start SEQRCSPIVProjected with random start")
                int_opt_T =ITensorCPD.optimize(cp_T,alsQR;verbose);
                push!(SEQRCS_error_vect,check_fit(alsQR, int_opt_T.factors, r, int_opt_T.λ, 1))

                factors = Vector{ITensor}()
                lambdaa= nothing
                for (i, Q) in enumerate(alsQR.additional_items[:qr_factors])
                    mode_idx = inds(T)[i]
                    fact = ITensors.itensor(Q[:,1:dim(r)],mode_idx,r)
                    fact,lambda = ITensorCPD.row_norm(fact,mode_idx)
                    push!(factors,fact)  
                    lambdaa= lambda
                end
                cpd_HOSVD = ITensorCPD.CPD{ITensor}(factors, lambdaa)
                alsQR = ITensorCPD.compute_als(T,cpd_HOSVD;alg=alg_QR, check =check_piv);
                # println("Start SEQRCSPIVProjected with HOSVD start")
                int_opt_T = ITensorCPD.optimize(cpd_HOSVD,alsQR;verbose);
                push!(SEQRCS_error_vect_HOSVD,check_fit(alsQR, int_opt_T.factors, r, int_opt_T.λ, 1))

                inds_T= inds(T)
                cpd_High = construct_large_lev_score_cpd(inds_T,r , 2)
                alsQR = ITensorCPD.compute_als(T,cpd_High;alg=alg_QR, check =check_piv);
                # println("Start SEQRCSPIVProjected with High_coh")
                int_opt_T =ITensorCPD.optimize(cpd_High,alsQR;verbose);
                push!(SEQRCS_error_vect_High,check_fit(alsQR, int_opt_T.factors, r, int_opt_T.λ, 1))

            end
        end

        err_leverage = Vector{Float64}()
        lev_error_vect = Vector{Float64}()
        ## Leverage Score 
        for s in samples
            for q=1:nrepeats
                alslev = ITensorCPD.compute_als(T,cp_T;alg = ITensorCPD.LevScoreSampled(s),check = check_piv)
                int_opt_T = ITensorCPD.optimize(cp_T, alslev; verbose)
                # println("Start lev")
                push!(lev_error_vect,check_fit(alslev, int_opt_T.factors, r, int_opt_T.λ, 1))
            end

        end
    
        SEQRCS_error_mat = reshape(SEQRCS_error_vect, (nrepeats,length(samples)))
        SEQRCS_error_mat_HOSVD = reshape(SEQRCS_error_vect_HOSVD, (nrepeats,length(samples)))
        SEQRCS_error_mat_High = reshape(SEQRCS_error_vect_High, (nrepeats,length(samples)))
        lev_error_mat = reshape(lev_error_vect, (nrepeats,length(samples)))
        
        err_SEQRCS = median.(eachcol(SEQRCS_error_mat))
        err_SEQRCS_HOSVD = median.(eachcol(SEQRCS_error_mat_HOSVD))
        err_SEQRCS_High = median.(eachcol(SEQRCS_error_mat_High))
        err_leverage = median.(eachcol(lev_error_mat))

        err_direct = Vector{Float64}()
        alsNormal = ITensorCPD.compute_als(T, cp_T; alg=ITensorCPD.direct(), check =check_direct)
        opt_T = ITensorCPD.optimize(cp_T, alsNormal; verbose)
        direct_error = check_fit(alsNormal, opt_T.factors, r, opt_T.λ, 1)
        push!(err_direct,direct_error)

        plt2 = plot(samples[2:end], err_SEQRCS[2:end], marker=:o, label="SEQRCSPivProjected")
        plot!(samples[2:end], err_SEQRCS_HOSVD[2:end], marker=:o, label="SEQRCSPivProjected_HOSVD")
        plot!(samples[2:end], err_SEQRCS_High[2:end], marker=:o, label="SEQRCSPivProjected_High")
        plot!(samples[2:end], err_leverage[2:end], marker=:o, label="LevScoreSampled")
        plot!(samples[2:end], direct_error .* ones(length(samples[2:end])), marker=:o, label="Direct")

        xlabel!("Number of Samples")
        ylabel!("Error in CPD Fit")
        title!("Synthetic Tensor Test:\n Rank $rk")
    end
end
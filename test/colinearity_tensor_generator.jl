function Collinearity_Tensor(r::Int, nb_factors, dim_vect, c, lam=nothing, elt=Float64)
    factors = Vector{ITensor}()
    l = Index(r,"cpd_rank")
    c = elt(c)
    for (i,d) in zip(1:nb_factors, dim_vect)
        # Generate a random unit direction vector
        u = randn(elt, d)
        u=u./norm(u)
        u = reshape(u, d, 1)

        # Generate columns of the factor matrix
        V = randn(elt, d,r)
        for j in 1:r
            V[:,j] -= dot(u,V[:,j])*u
        end

        Q,_=qr(V)
        V = Q[:,1:r]

        # Generate the factor matrices
        A_array = sqrt(c)*u .+ sqrt(1-c)*V
        A = ITensors.itensor(A_array,Index(d),l)
        push!(factors,A)
    end
    
    lam_tensor = nothing
    if isnothing(lam)
        lam_tensor = ITensor(ones(elt, dim(l)), l)
    else
        lam_tensor = ITensors.itensor(lam,l)
    end
    cpd = ITensorCPD.CPD{ITensor}(factors, lam_tensor)
    #T = ITensorCPD.reconstruct(cpd)
    h1 = ITensorCPD.had_contract(factors[1:nb_factors ÷ 2], l)
    h2 = ITensorCPD.had_contract(ITensorCPD.had_contract(factors[nb_factors ÷ 2+1:end], l),lam_tensor, l)
    T = h1 * h2

    return T, cpd
end

## Validation.
#  T, cpd = Collinearity_Tensor(10, 3, (1000,500,600), 0.1);

#  f1 = array(cpd[1])
#  s1 = 1
#  s2 = 9
#  (f1[:,s1]' * f1[:,s2]) / (norm(f1[:,s1]) * norm(f1[:,s2]))
#  using Random
#  TP = T 
#  rng = RandomDevice()
#  check = ITensorCPD.FitCheck(1e-3, 20, norm(TP))
#  guess = ITensorCPD.random_CPD(TP, 10; rng)
#  ITensorCPD.als_optimize(TP, guess; check, verbose=true);

#  alg=ITensorCPD.BlockLevScoreSampled(40, 1)
#  ITensorCPD.als_optimize(TP, guess; check, verbose=true, alg);

#  ## What is this bug
#  #alg = ITensorCPD.SEQRCSPivProjected((1,), (40,), (1,2,3), (1,1,1))

#  alg = ITensorCPD.SEQRCSPivProjected((1,), (40,), (1,2,3), (10,10,10))
#  ITensorCPD.als_optimize(TP, guess; check, verbose=true, alg);

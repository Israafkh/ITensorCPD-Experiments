using Pkg
using CSV, DataFrames
using Dates
using LazyFunctionArray
include("../test_env.jl")
path = "$(@__DIR__)/../../data/uber-pickups-in-new-york-city/"
#file_names = readdir(path)
file_names = [
    "uber-raw-data-apr14.csv"
    "uber-raw-data-may14.csv"
    "uber-raw-data-jun14.csv"
    "uber-raw-data-jul14.csv"
    "uber-raw-data-aug14.csv"
]
dfs = map(fn -> CSV.read(path * fn, DataFrame, delim=',', header=true), file_names)

lats = map(x->unique(x.Lat), dfs)
unlats = unique(vcat(lats...))

lons = map(x->unique(x.Lon), dfs)
unlons = unique(vcat(lons...))

hour.(Matrix(dfs[1][:,["Time"]]))
for i in 1:length(dfs)
    dfs[i][:,["Time"]] = Time.(hour.(Matrix(dfs[i][:,["Time"]])))
end

dateform = DateFormat("m/d/y")
ds = [30, 31, 30, 31, 31]

daydict = Dict()
i = 1
for month in 1:5
    for day in 1:ds[month]
        daydict["$(month+3)/$(day)/2014"] = i
        i += 1
    end
end
latdict = Dict(unlats[1:1100] .=> 1:1100);
londict = Dict(unlons[1:1100] .=> 1:1100);

data = zeros(Float32, 183, 24, 1100, 1100);
latgroup = groupby(dfs[1], ["Date","Time","Lat","Lon"])
for i in latgroup
    if !haskey(latdict, i.Lat[1]) || !haskey(londict, i.Lon[1])
        continue
    end
    pos = vcat((daydict[i.Date[1]], hour(i.Time[1]) + 1, latdict[i.Lat[1]], londict[i.Lon[1]])...)
    data[pos] .= size(i)[1]
end



reddit_itensor = itensor(data, Index.(size(data)));

cpd_guess = ITensorCPD.random_CPD(reddit_itensor, 100);
alsLev = ITensorCPD.compute_als(reddit_itensor, cpd_guess; 
alg = ITensorCPD.SEQRCSPivProjected(1,1000, (1,2,3,4), (1,1,1,1)),
check=ITensorCPD.FitCheck(1, 100, 1),
);

opt = ITensorCPD.optimize(cpd_guess, alsLev; verbose=true);
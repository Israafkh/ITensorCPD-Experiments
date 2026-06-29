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
    "uber-raw-data-sep14.csv"
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
ds = [30, 31, 30, 31, 31, 30]

# data = zeros(Int, (sum(ds), 24, length(unlats), length(unlons)))
### This is organized as day [0,183], time, lat, lon 
### day gets decomposed by finding which month it's in and selecting the right list, then selecting 
### the day from that list from [1, ds[month]]
function get_reddit_data(x... ;data=dfs)
    day = x[1]
    month = 0
    for i in 1:6
        if day ≤ ds[i]
            month = i
            break
        else
            day = day - ds[i]
        end
    end
    
    daymonthgroup = groupby(dfs[month], "Date")[day]
    try
        daymonthtime = DataFrame(groupby(daymonthgroup, "Time")[x[2]])
    catch   
        # println("Missing time on this date")
        return 0
    end
    tmp = daymonthgroup[(daymonthgroup.Lat .== unlats[x[3]]),:]
    return size(tmp[(tmp.Lon .== unlons[x[4]]), :])[1]
    # for mon in 1:length(dfs)
    #     for (dategroup, i) in zip(groupby(dfs[mon], "Date"), 1:ds[mon])
    #         for (timegroup, j) in zip(groupby(dategroup, "Time"), 1:24)
                
    #         end
    #     end
    # end
end

rngs = [
    1:183,
    1:24,
    1:1000,
    1:1000,
]
fa = FunctionArray(get_reddit_data, rngs);
tn = LazyFunctionArray

reddit_itensor = itensor(fa, Index.(size(fa)))

cpd_guess = ITensorCPD.random_CPD(reddit_itensor, 100);
alsLev = ITensorCPD.compute_als(reddit_itensor, cpd_guess; 
alg = ITensorCPD.LevScoreSampled(1000),
check=ITensorCPD.FitCheck(1, 100, 1),
);

opt = ITensorCPD.optimize(cpd_guess, alsLev; verbose=true);
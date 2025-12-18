include("$(@__DIR__)/../test_env.jl")
using Images, FileIO

function readcoil(coil_path,numobjects, elt=Float32)
# 	% Read COIL-100 images, given a number of objects
	X = Array{elt}(undef, 72*numobjects,128,128,3);
	for obj_num = 1:numobjects
		for i = 1:72
		    degrees = 5*(i-1);
		    filename = joinpath(coil_path, "obj$(obj_num)__$(degrees).png")
            # sprintf('%s/obj%d__%d.png',coil_path,obj_num,degrees);
		    # %X(i,:,:,obj_num) = rgb2gray(imread(filename, 'png'));
            img = load(filename)
			m = convert.(elt, Array(channelview(img)))
		    X[i + 72*(obj_num-1),:,:,:] .= permutedims(m, (2,3,1))
		end
	end
    is = Index.(size(X))
	return itensor(X, is);
end

# Learning Summary Statstics for Bayesian Inference with Autoencoders

This is the repository for assignment 2 of MXN442. All testing, implementation and plotting was performed in Julia version 1.6.7.

The folder `Demonstrations/` contains the code to recreate the plots demonstrated in the results section, as well as, a script to recreate the data used. Note, this data is not included due to file upload limits. Running GenerateData.jl saves three files, `savedData/true.jld2` a single simulated observation using the true parameter $k=0.25$, `savedData/PriorPredictiveSamples.jld2` one million samples for training purposes, and `savedData/InferenceSamples.jld2` ten million samples for generating inference results.

The folder `models/` contains `AutoEncoder.jl` my implementaion of the INCA autoencoder using FLUX.jl, and `train_AutoEncoder.jl` has a script which trains the network over 100 epochs of the data stored in `savedData/PriorPredictiveSamples.jld2`. Finally `train_PCA.jl` computes the PCA transform of of the prior predictive and stores the transform in `savedData/PcaTransform.jld2`. 

The folder `Modules/` contains `CustomPlots` a Julia module I wrote for common plotting needs. This model needs to be added to the Julia path, and it is not done automatically unless you have intentionally setup julia. To add `CustomPlots` to path use the command `push!(LOAD_PATH, joinpath(pwd(), "Modules\\CustomPlots\\"))`. You can check the path with the command ``LOAD_PATH``, the path must contain `...\\Modules\\CustomPlots\\`, note it will be the root path plus these folders. This process can be automated by going to your `.julia` file, adding a file `.julia/config/startup.jl` and including,
```{julia}
if "Modules" in readdir()
    for path in readdir("Modules")
        push!(LOAD_PATH, joinpath(pwd(), "Modules\\" * path * "\\"))
    end
end
```
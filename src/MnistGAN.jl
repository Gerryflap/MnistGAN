module MnistGAN
using MLDatasets: MNIST
using Flux
using Printf
using Images

import Flux.MLUtils: MLUtils
import Flux.Optimise: Optimise
import Flux.Losses: Losses 

include("util.jl")

function train(epochs :: Int64; batch_size = 64, h_size = 32, latent_size = 64)
    dataset = MNIST(Float32, :train)
    features = dataset.features

    # Get number of samples (last dim)
    n_samples = size(features, 3)

    features = reshape(features, (28, 28, 1, n_samples))

    D = Chain(
        Conv((3, 3), 1 => h_size, relu; stride=2, pad=1), # 14x14
        Conv((3, 3), h_size => h_size*2, relu; stride=2, pad=1), # 7x7
        Flux.flatten,
        # No activation fn, we'll deal with that in the loss fn
        Dense(7*7*h_size*2 => 1, identity)
    ) |> gpu

    G = Chain(
        Dense(latent_size => h_size*2*7*7, relu),
        # Reshape to image shape
        vec -> reshape(vec, (7, 7, h_size*2, batch_size)),
        # Use asymetric padding to arrive at the right size
        ConvTranspose((3,3), h_size*2 => h_size, relu; stride=2, pad=(1,0,1,0)),
        ConvTranspose((3,3), h_size => 1, sigmoid; stride=2, pad=(1,0,1,0))
    ) |> gpu

    # Initialize the optimizer
    opt_d = Optimise.Adam(1e-3)
    opt_g = Optimise.Adam(1e-3)

    # Define a dataloader with the given batch size, shuffled order every epoch, and disallow partial batches (batches smaller than batch size)
    loader = MLUtils.DataLoader(features; batchsize=batch_size, shuffle=true, partial=false)

    for epoch in 1:epochs
        # Make loss variables available in scope
        local d_loss
        local g_loss

        for x in loader
            x = x |> gpu

            # ===== Train D =====
            Dp = Flux.params(D)
            fake = G(Float32.(randn(latent_size, batch_size)) |> gpu )
            gradients = gradient(Dp) do
                pred_real = D(x)
                pred_fake = D(fake)
                # NS-Gan losses (similar to StyleGan2 implementation)
                fake_loss = sum(softplus.(pred_fake))
                real_loss = sum(softplus.(-pred_real))
                d_loss = fake_loss + real_loss
                return d_loss
            end

            # Perfom an optimization step
            Optimise.update!(opt_d, Dp, gradients)

            # ===== Train G =====
            Gp = Flux.params(G)
            gradients = gradient(Gp) do
                fake = G(Float32.(randn(latent_size, batch_size)) |> gpu )
                pred_fake = D(fake)
                # NS-Gan losses (similar to StyleGan2 implementation)
                fake_loss = sum(softplus.(-pred_fake))
                g_loss = fake_loss
                return g_loss
            end

            # Perfom an optimization step
            Optimise.update!(opt_g, Gp, gradients)
        end

        # Print the performance of the model every now and then
        @printf "Epoch %d, d_loss = %.4f, g_loss = %.4f\n" epoch d_loss g_loss

        if epoch % 1 == 0
            fake = G(Float32.(randn(latent_size, batch_size)) |> gpu ) |> cpu
            # img = reshape(fake[:, :, :, 1], (28, 28))
            img = make_image_grid(fake)
            
            name = @sprintf("output/epoch_%06d.png", epoch)
            save(name, colorview(Gray, img))
        end
    end
end

end # module

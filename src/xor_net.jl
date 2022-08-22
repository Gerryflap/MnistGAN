#=
    Simple XOR gate Neural Net to get up to speed with the library
=#
using Flux
import Flux.Optimise: Optimise
using Flux.MLUtils
using Flux.Losses
using Printf

# Define the data (simple xor gate)
# For whatever cursed reason, batch size is last ;p
xs  = Float32.([0 1 1 0; 0 0 1 1])
ys  = Float32.(reshape([  0;   1;   1;   0], 1, :))

# Define the model
model = Chain(
    Dense( 2 => 32, sigmoid),
    Dense( 32 => 1, sigmoid)
)

# Initialize the optimizer
opt = Optimise.Adam(1e-3)

# Custom train loop
for step in 1:10000
    # Make loss variable available in scope
    local loss

    # Collect gradients
    parameters = Flux.params(model)
    gradients = gradient(parameters) do
        pred = model(x_batch)
        loss = crossentropy(pred, y_batch)
        return loss
    end

    # Perfom an optimization step
    Optimise.update!(opt, parameters, gradients)

    # Print the performance of the model every now and then
    if step % 100 == 0
        @printf "Step %d, loss = %.4f\n" step loss
    end
end
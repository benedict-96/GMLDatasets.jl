# One configuration of `mnist_cuda.jl`, trained several times, reported as a mean and a sample
# standard deviation instead of a single number.
#
# `mnist_cuda.jl` trains each of the four configurations once, which is what a comparison of
# their learning curves needs. It is not enough for the *accuracy* of the two `Adam`
# configurations, because that number does not reproduce run to run:
#
#   At the first step `Adam`'s update is essentially the sign of each coordinate of the
#   gradient — the moment estimates are the gradient itself, so the ratio is ±1 per coordinate
#   up to `ε`. The gradient of this network has vanished through 16 unnormalized transformer
#   blocks before it reaches the early layers, so a great many coordinates sit at ~0 and have
#   their sign decided by the last ulp of a reduction. A difference of one ulp — a different
#   accumulation order in `cuBLAS`, a different number of threads on the host — therefore
#   changes step one at O(1), and the two trajectories never rejoin. Two host runs of
#   `mnist.jl` with the *same* seed gave test accuracies `0.3761` and `0.3302` and drifts
#   `4.6e-04` and `3.7e-05` off the manifold; `GradientMethod` and `MomentumMethod`, which use
#   the gradient itself and not its sign, reproduced to four decimals.
#
# So a single accuracy from an `Adam` configuration is a sample and not a number, and this
# script is what turns it into one: it trains the same configuration `MNIST_REPETITIONS` times
# and ends on `mean ± deviation` over the repetitions, for the test accuracy, the final epoch
# loss and the drift off the manifold. Everything else — the network, the initialization, the
# objective, the gradient, the schedule and the bars the runs are judged against — is that of
# `mnist_cuda.jl`, so a repetition here is comparable to the corresponding run there.
#
# By default it repeats geometric Adam with Stiefel weights and the package-default Cayley
# retraction, which is the configuration the paper's result rests on. This is not Cayley ADAM:
# that name is reserved for `ScalarMomentAdam` from Li et al. (2020). `MNIST_CONFIGURATIONS`
# selects other configurations. For `gradient` and `momentum` the default
# (a seed per repetition) measures the spread over *initializations*, which is a meaningful
# number but a different one; with `MNIST_VARY_SEED=0` those two would reproduce their run and
# the repetitions would cost ≈1:50 h each to confirm it.
#
# ## What it costs
#
# One repetition is one configuration of the full run: 500 epochs × 29 batches at 0.37–0.47
# s/step on an RTX 4090, i.e. ≈1:35 h. Five repetitions of one configuration are therefore
# ≈8 h, about the same as the ≈6:53 h `mnist_cuda.jl` takes for all four. Run it the same way,
# in a `screen`, through the wrapper next to it:
#
#   screen -dmS mnist scripts/geometric_optimizers/run_mnist_repetitions.sh              # 10 × geometric Adam
#   screen -dmS mnist scripts/geometric_optimizers/run_mnist_repetitions.sh --repeat 3   # 3 of them
#   scripts/geometric_optimizers/run_mnist_repetitions.sh --smoke                        # 3 × 2 epochs, ≈4 min
#
# ## The report
#
# The same three self-contained files `mnist_cuda.jl` writes, with the repetition added
# wherever a run is identified:
#
#   mnist_repetitions_report.txt   the environment, the gradient checks, one line per epoch and
#                                  per repetition, the statistics and the verdict — flushed
#                                  after every line
#   mnist_repetitions_losses.csv   one row per optimizer step:
#                                  run, configuration, repetition, epoch, batch, step, loss
#   mnist_repetitions.jld2         the parameters, losses, timings and accuracies of every
#                                  repetition, plus the samples the statistics are computed
#                                  from — rewritten after every repetition
#
# Note that the CSV has one column more than the one `mnist_cuda.jl` writes, so
# `scripts/geometric_optimizers/distill_mnist_results.jl` — which feeds the figures of the documentation from a
# single full run — does not read this file. The figures are a comparison of the four
# configurations and remain the job of `mnist_cuda.jl`; this script answers how far the `Adam`
# number in them can be trusted.
#
# ## Environment variables
#
#   MNIST_REPETITIONS     how often each configuration is trained   (default 10)
#   MNIST_CONFIGURATIONS  which ones, comma separated: `geometric-adam-cayley`,
#                         `standard-adam`, `gradient`, `momentum`, or `all`
#                         (`adam-stiefel` and `adam-regular` remain accepted aliases)
#   MNIST_VARY_SEED       repetition `r` gets the seed `seed + r - 1` (default 1); with `0`
#                         every repetition uses the same seed, which measures the
#                         nondeterminism above rather than the spread of the method
#   MNIST_N_EPOCHS        epochs per repetition        (default 500; use 2 for a smoke test)
#   MNIST_BATCH_SIZE      images per batch                           (default 2048)
#   MNIST_ACCURACY_EVERY  epochs between test evaluations (default 25, `0` disables them)
#   MNIST_REPORT          path of the report            (default mnist_repetitions_report.txt)
#   MNIST_LOSSES          path of the loss CSV          (default mnist_repetitions_losses.csv)
#   MNIST_OUTPUT          path of the `.jld2`           (default mnist_repetitions.jld2)
#   MNIST_PROGRESS        print the per-batch progress line (default: only on a terminal)
#
# On the device split, the two `Zygote` workarounds and the memory the backward pass needs, see
# the header of `mnist_cuda.jl`: this script is that one with the run loop replaced, and it
# carries the same code for everything below the loop.

using GeometricOptimizers
using GeometricOptimizers: solver_step!, increase_iteration_number!, initialize_state!, ParameterHandling
using SimpleSolvers: Static
using LinearAlgebra: norm, I, Adjoint, Transpose
using NNlib: batched_mul, batched_transpose, softmax, BatchedAdjOrTrans
using Printf: @printf, @sprintf
import CUDA, ForwardDiff, JLD2, MLDatasets, Random, Zygote

# --------------------------------------------------------------------------- device ---

const use_cuda = CUDA.functional()

# `to_device` is a `const` binding to a *function*, so the element types of the forward pass
# are still inferred.
const to_device = use_cuda ? CUDA.cu : identity
to_host(x::AbstractArray) = Array(x)
synchronize_device() = use_cuda ? CUDA.synchronize() : nothing

# --------------------------------------------------------------------------- report ---

# As in `mnist_cuda.jl`: the run outlives the terminal it was started in, so everything it
# learns goes into a file as well, flushed after every line.

const report_path = get(ENV, "MNIST_REPORT", "mnist_repetitions_report.txt")
const output_path = get(ENV, "MNIST_OUTPUT", "mnist_repetitions.jld2")
const losses_path = get(ENV, "MNIST_LOSSES", "mnist_repetitions_losses.csv")
const records_path = get(ENV, "MNIST_RECORDS", "mnist_repetitions_runs.csv")
const report_io = open(report_path, "w")

const losses_io = open(losses_path, "w")
println(losses_io, "run,configuration,repetition,epoch,batch,step,loss")
flush(losses_io)

"""
    report(line)

Append `line` to the report file and flush it.
"""
function report(line::AbstractString="")
    println(report_io, line)
    flush(report_io)
    nothing
end

"""
    announce(line)

Write `line` to both the terminal and the report. `stdout` is flushed as well, so that the
redirected output of a `nohup`ed run is worth following.
"""
function announce(line::AbstractString="")
    println(line)
    flush(stdout)
    report(line)
end

timestamp() = Libc.strftime("%Y-%m-%d %H:%M:%S", time())

"""
    duration(seconds)

`seconds` as `h:mm:ss`, which is the only readable unit for the numbers in this script.
"""
function duration(seconds::Real)
    total = round(Int, seconds)
    @sprintf("%i:%02i:%02i", total ÷ 3600, (total ÷ 60) % 60, total % 60)
end

# Whether the individual batches are worth printing: on a terminal the progress line is how
# you see that the first run is moving at all, but redirected into a file its carriage returns
# are tens of thousands of lines of noise, so there it is epochs only.
const to_terminal = parse(Bool, get(ENV, "MNIST_PROGRESS", string(stdout isa Base.TTY)))

# clear the progress line before something permanent is printed over it
clear_progress() = to_terminal && print("\r", " "^78, "\r")

"""
    attempt(f, fallback)

`f()`, or `fallback` if it throws. Used for the environment section only: a `CUDA` accessor
that was renamed upstream should not be what stops a run of this length.
"""
attempt(f::Base.Callable, fallback="unknown") = try
    string(f())
catch
    fallback
end

function report_environment()
    report("mnist_cuda_repetitions.jl — " * (use_cuda ? "CUDA" : "host") * " run")
    report("=" ^ 100)
    report()
    report("started     " * timestamp())
    report("host        " * gethostname())
    report("julia       " * string(VERSION) * ", " * string(Threads.nthreads()) * " thread(s)")
    report("project     " * string(Base.active_project()))
    report("script      " * @__FILE__)
    report("revision    " * attempt(() -> readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)) *
           attempt(() -> isempty(readchomp(`git -C $(@__DIR__) status --porcelain`)) ? "" : " (dirty)", ""))
    report("package     GeometricOptimizers " * attempt(() -> pkgversion(GeometricOptimizers)) *
           ", CUDA.jl " * attempt(() -> pkgversion(CUDA)))
    if use_cuda
        report("device      " * attempt(() -> CUDA.name(CUDA.device())) *
               ", " * attempt(() -> string(CUDA.totalmem(CUDA.device()) ÷ 1024^2, " MB")) *
               ", capability " * attempt(() -> CUDA.capability(CUDA.device())))
        report("cuda        driver " * attempt(CUDA.driver_version) *
               ", runtime " * attempt(CUDA.runtime_version))
    else
        report("device      none — the network is evaluated on the host")
    end
    report()
end

# The high water mark of device memory, reported at the end.
const peak_used = Ref(0)

# `CUDA.available_memory` is gone: 5.8 renamed it to `free_memory`, and the 6.x split into
# `CUDACore`/`CUDATools` forwards the new name only. Resolve it once, and let the number be the
# thing that is lost if it is missing again — this one is called from inside the epoch loop, so
# a `MethodError` here would take the whole run with it.
const free_device_memory =
    !use_cuda                          ? () -> nothing :
    isdefined(CUDA, :free_memory)      ? CUDA.free_memory :
    isdefined(CUDA, :available_memory) ? CUDA.available_memory :
                                         () -> nothing

function note_device_memory()
    free = free_device_memory()
    free === nothing && return nothing
    peak_used[] = max(peak_used[], Int(CUDA.total_memory() - free))
    nothing
end

if use_cuda
    println("running on ", CUDA.name(CUDA.device()), " (", CUDA.totalmem(CUDA.device()) ÷ 1024^2, " MB)")
else
    @warn "no CUDA device available — the network is evaluated on the host instead"
end
report_environment()
println("writing the report to ", report_path, " and the parameters to ", output_path)

# ------------------------------------------------------------------- hyperparameters ---

const patch_length = 7      # MNIST images are 28×28, so the sequence length is (28÷7)² = 16
const n_heads = 7
const L = 16                # the number of transformer blocks
const add_connection = false
const T = Float32
const learning_rate = T(1e-3)
const momentum_coefficient = T(0.5)
const seed = parse(Int, get(ENV, "MNIST_BASE_SEED", "1234"))

const batch_size = parse(Int, get(ENV, "MNIST_BATCH_SIZE", "2048"))
const n_epochs = parse(Int, get(ENV, "MNIST_N_EPOCHS", "500"))
const accuracy_every = parse(Int, get(ENV, "MNIST_ACCURACY_EVERY", "25"))

# The revision protocol uses ten independently seeded repetitions. `1` remains valid for smoke
# tests and debugging.
const n_repetitions = parse(Int, get(ENV, "MNIST_REPETITIONS", "10"))
@assert n_repetitions ≥ 1 "MNIST_REPETITIONS must be at least 1"

# Whether the repetitions differ in their seed. See the header: with a varying seed the spread
# is that of the method over initializations *and* over the nondeterminism, which is the number
# to quote for a result; with a fixed seed it is the nondeterminism alone, which is what made
# this script necessary and is worth being able to measure separately.
const vary_seed = parse(Bool, get(ENV, "MNIST_VARY_SEED", "1"))
const requested_seeds = let value = strip(get(ENV, "MNIST_SEEDS", ""))
    isempty(value) ? Int[] : parse.(Int, strip.(split(value, ','; keepempty=false)))
end
isempty(requested_seeds) || length(requested_seeds) == n_repetitions ||
    error("MNIST_SEEDS contains $(length(requested_seeds)) seeds but MNIST_REPETITIONS=$n_repetitions")

repetition_seed(r::Integer) = !isempty(requested_seeds) ? requested_seeds[r] : vary_seed ? seed + r - 1 : seed

const dim = patch_length^2                      # the transformer dimension
const seq_length = (28 ÷ patch_length)^2        # the number of patches
const n_classes = 10
const Dₕ = dim ÷ n_heads                        # the dimension of a single head

@assert dim % n_heads == 0

# The bars the runs are judged against, as in `mnist_cuda.jl` — every repetition is judged
# individually, and the statistics are reported alongside those verdicts rather than instead of
# them: five repetitions whose mean clears the floor but of which one collapsed is not the same
# outcome as five that all worked, and only the individual verdicts can tell the two apart.
const chance_accuracy = T(1 / n_classes)
const accuracy_floor = T(0.30)
const loss_decrease_floor = T(0.02)
const trivial_loss = sqrt(T(2) * (one(T) - chance_accuracy))
const trivial_loss_tolerance = T(0.05)
const orthonormality_tolerance = T(1e-2)
const accuracy_floor_min_epochs = 50
const gradient_tolerance = 1e-4

# ----------------------------------------------------------------------------- data ---

@doc raw"""
    split_and_flatten(input, patch_length)

Rearrange a batch of images into *flattened patches*, i.e. turn an ``(N, N, k)`` array into
an ``(\mathtt{patch\_length}^2, (N \div \mathtt{patch\_length})^2, k)`` array.

This is the equivalent of `GeometricMachineLearning.split_and_flatten` and produces the same
ordering: the patches are numbered column-major over the image and the entries within a
patch are numbered column-major as well.
"""
function split_and_flatten(input::AbstractArray{<:Number,3}, patch_length::Integer)
    @assert size(input, 1) == size(input, 2)
    @assert size(input, 1) % patch_length == 0
    n = size(input, 1) ÷ patch_length
    # (i_red, patch_row, j_red, patch_column, k) → (i_red, j_red, patch_row, patch_column, k)
    output = permutedims(reshape(input, patch_length, n, patch_length, n, size(input, 3)), (1, 3, 2, 4, 5))
    reshape(output, patch_length^2, n^2, size(input, 3))
end

"""
    onehotbatch(target)

Turn a vector of labels (`0` to `9`) into a `10`×`length(target)` matrix of unit vectors.
"""
function onehotbatch(target::AbstractVector{<:Integer})
    output = zeros(T, n_classes, length(target))
    for (k, label) in pairs(target)
        output[label+1, k] = one(T)
    end
    output
end

# ----------------------------------------------------------------------- parameters ---

# The parameters are stored in a flat `NamedTuple` *on the host*. Note that only the
# projections of the attention layers are put on the `StiefelManifold`; the parameters of the
# `ResNetLayer`s and of the classification layer are ordinary arrays.
const parameter_layout = [
    [Symbol(s, "_", l, "_", h) => (dim, Dₕ) for l in 1:L for s in ("PQ", "PK", "PV") for h in 1:n_heads]
    vcat([[Symbol("Wres_", l) => (dim, dim), Symbol("bres_", l) => (dim,)] for l in 1:L]...)
    [:Wclass => (n_classes, dim)]
]

const parameter_keys = Tuple(first.(parameter_layout))
const parameter_sizes = last.(parameter_layout)
const n_attention_parameters = 3 * n_heads * L

# the position of a parameter within `parameter_layout`
attention_index(kind::Integer, l::Integer, h::Integer) = (l - 1) * 3 * n_heads + (kind - 1) * n_heads + h
resnet_index(l::Integer) = n_attention_parameters + 2 * (l - 1) + 1      # the bias follows right after
const classification_index = lastindex(parameter_layout)

# the range that a parameter occupies in the flattened parameter vector, i.e. in
# `ParameterHandling.flatten(ps)[1]`
const parameter_ranges = let offsets = cumsum([0; prod.(parameter_sizes)])
    [(offsets[i]+1):offsets[i+1] for i in eachindex(parameter_sizes)]
end
const n_parameters = last(last(parameter_ranges))

# `GlorotUniform` of `AbstractNeuralNetworks`, which is what the original script uses for all
# parameters that are not on a manifold.
function glorot_uniform(rng::Random.AbstractRNG, size::Tuple)
    x = rand(rng, T, size...)
    sqrt(T(24) / sum(size)) * (x .- T(0.5))
end

function initial_parameters(rng::Random.AbstractRNG, stiefel::Bool)
    values = map(enumerate(parameter_sizes)) do (i, size)
        if i ≤ n_attention_parameters
            stiefel ? rand(rng, StiefelManifold{T}, size...) : glorot_uniform(rng, size)
        elseif length(size) == 1
            zeros(T, size...)               # the biases are initialized with zeros
        else
            glorot_uniform(rng, size)
        end
    end
    NamedTuple{parameter_keys}(Tuple(values))
end

_array(Y::StiefelManifold) = Y.A
_array(Y::AbstractArray) = Y

# For the forward pass the parameters are regrouped into vectors of concrete element type.
# Doing this outside of the differentiated function keeps both the forward pass and `Zygote`
# type stable.
function regroup(get_parameter::Base.Callable)
    (Q=[get_parameter(attention_index(1, l, h)) for l in 1:L for h in 1:n_heads],
        K=[get_parameter(attention_index(2, l, h)) for l in 1:L for h in 1:n_heads],
        V=[get_parameter(attention_index(3, l, h)) for l in 1:L for h in 1:n_heads],
        Wres=[get_parameter(resnet_index(l)) for l in 1:L],
        bres=[get_parameter(resnet_index(l) + 1) for l in 1:L],
        Wclass=get_parameter(classification_index))
end

"""
    regroup_host(v)

Regroup the *flattened* parameters without touching the device. The element type is kept
general so that `check_gradient` can differentiate through it — `ForwardDiff.Dual`s cannot be
handed to `cuBLAS`, so the reference derivative has to be computed on the host.
"""
regroup_host(v::AbstractVector{<:Number}) = regroup(i -> reshape(v[parameter_ranges[i]], parameter_sizes[i]...))

# The parameters are moved to the device in a single transfer and split up there; the 353
# individual parameters are far too small for a transfer each.
const host_parameters = zeros(T, n_parameters)
const device_parameters = to_device(zeros(T, n_parameters))
const device_gradient = to_device(zeros(T, n_parameters))

"""
    flatten_parameters!(v, ps)

Write the parameter `NamedTuple` into the flat vector `v`, in the same order in which
`ParameterHandling.flatten` writes it (which is the order of `parameter_ranges`).
"""
function flatten_parameters!(v::AbstractVector{T}, ps::NamedTuple)
    for (i, p) in pairs(values(ps))
        copyto!(view(v, parameter_ranges[i]), vec(_array(p)))
    end
    v
end

function regroup_device(v::AbstractVector{T})
    copyto!(device_parameters, v)
    regroup(i -> reshape(device_parameters[parameter_ranges[i]], parameter_sizes[i]...))
end

regroup_device(ps::NamedTuple) = regroup_device(flatten_parameters!(host_parameters, ps))

# ---------------------------------------------------------------------------- model ---

"""
    mat_tensor_mul(A, x)

Multiply `A` onto every matrix stored in `x`, i.e. parallelize over the third axis.
"""
function mat_tensor_mul(A::AbstractMatrix, x::AbstractArray{<:Number,3})
    reshape(A * reshape(x, size(x, 1), :), size(A, 1), size(x, 2), size(x, 3))
end

# `Zygote` differentiates `mat_tensor_mul` by pulling the cotangent through the `reshape`, and
# the cotangent of a `Q` that enters `batched_mul` through `batched_transpose` is a *lazy*
# `BatchedTranspose`. Reshaping that produces a `ReshapedArray` which `cuBLAS` does not
# recognize, so the multiplication of the backward pass falls back to the generic method of
# `LinearAlgebra` — which indexes the arrays element by element and therefore errors on the
# device. The rule below writes the backward pass by hand and materializes such cotangents
# first; on the host it avoids the same (there merely slow) fallback.
_dense(Δ::AbstractArray) = Δ
_dense(Δ::BatchedAdjOrTrans) = permutedims(parent(Δ), (2, 1, 3))    # all arrays here are real
_dense(Δ::Union{Adjoint,Transpose}) = permutedims(parent(Δ), (2, 1))

Zygote.@adjoint function mat_tensor_mul(A::AbstractMatrix, x::AbstractArray{<:Number,3})
    function mat_tensor_mul_pullback(Δ)
        Δ₂ = reshape(_dense(Δ), size(A, 1), :)
        x₂ = reshape(x, size(x, 1), :)
        Δ₂ * transpose(x₂), reshape(transpose(A) * Δ₂, size(x))
    end
    mat_tensor_mul(A, x), mat_tensor_mul_pullback
end

"""
    predict(ps, input)

Apply the classification transformer to `input`, a `(dim, seq_length, k)` array, and return
the `(n_classes, k)` matrix of predictions. Here `ps` are *regrouped* parameters that live on
the same device as `input`.
"""
function predict(ps::NamedTuple, input::AbstractArray{<:Number,3})
    x = input
    for l in 1:L
        # the multi head attention layer
        heads = ntuple(n_heads) do h
            i = (l - 1) * n_heads + h
            Q = mat_tensor_mul(transpose(ps.Q[i]), x)
            K = mat_tensor_mul(transpose(ps.K[i]), x)
            V = mat_tensor_mul(transpose(ps.V[i]), x)
            batched_mul(V, softmax(batched_mul(batched_transpose(Q), K) ./ sqrt(T(dim)); dims=1))
        end
        y = add_connection ? x + reduce(vcat, heads) : reduce(vcat, heads)
        # the ResNet layer
        x = y + tanh.(mat_tensor_mul(ps.Wres[l], y) .+ ps.bres[l])
    end
    # the classification layer picks the last column and applies softmax
    softmax(ps.Wclass * x[:, end, :]; dims=1)
end

"""
    network_loss(ps, input, output)

The equivalent of `GeometricMachineLearning.FeedForwardLoss`.
"""
network_loss(ps::NamedTuple, input, output) = norm(predict(ps, input) - output) / norm(output)

"""
    accuracy(ps, input, output)

The ratio of correctly classified images, i.e. the equivalent of
`GeometricMachineLearning.accuracy`. Here `ps` are the parameters as stored by the optimizer
and `input`/`output` live on the host; only one chunk at a time is moved to the device. The
`argmax`es are taken on the host, as an `argmax` per column would be a scalar index on the
device.
"""
function accuracy(ps::NamedTuple, input, output; chunk_size=batch_size)
    regrouped = regroup_device(ps)
    correct = 0
    for k in Iterators.partition(axes(input, 3), chunk_size)
        prediction = to_host(predict(regrouped, to_device(input[:, :, k])))
        for (j, jₖ) in pairs(k)
            correct += argmax(view(prediction, :, j)) == argmax(view(output, :, jₖ))
        end
    end
    correct / size(input, 3)
end

# --------------------------------------------------------------- objective & gradient ---

# The `Optimizer` calls the objective on the parameter `NamedTuple` and `∇F!` on the
# *flattened* parameters, both of which live on the host. The current batch is on the device.
const current_batch = Ref{Tuple{AbstractArray{T,3},AbstractMatrix{T}}}()

function F(ps::NamedTuple)
    input, output = current_batch[]     # the function barrier keeps `network_loss` inferred
    network_loss(regroup_device(ps), input, output)
end

# the cotangent of a `transpose(ps.Q[i])` is a lazy `Transpose`, which `vec` turns into a
# wrapped array that the device cannot copy from — hence `_dense` here as well
_write_gradient!(g, i::Integer, ∂) = copyto!(view(g, parameter_ranges[i]), vec(_dense(∂)))
_write_gradient!(g, i::Integer, ::Nothing) = fill!(view(g, parameter_ranges[i]), zero(T))

function ∇F!(g::AbstractVector{T}, v::AbstractVector{T})
    input, output = current_batch[]
    ∂ps = Zygote.gradient(ps -> network_loss(ps, input, output), regroup_device(v))[1]
    # the gradient is assembled on the device and downloaded in a single transfer
    for l in 1:L, h in 1:n_heads
        i = (l - 1) * n_heads + h
        _write_gradient!(device_gradient, attention_index(1, l, h), ∂ps.Q[i])
        _write_gradient!(device_gradient, attention_index(2, l, h), ∂ps.K[i])
        _write_gradient!(device_gradient, attention_index(3, l, h), ∂ps.V[i])
    end
    for l in 1:L
        _write_gradient!(device_gradient, resnet_index(l), ∂ps.Wres[l])
        _write_gradient!(device_gradient, resnet_index(l) + 1, ∂ps.bres[l])
    end
    _write_gradient!(device_gradient, classification_index, ∂ps.Wclass)
    copyto!(g, device_gradient)

    g
end

"""
    check_gradient(ps)

Compare `∇F!` — which is evaluated on the device — to the *directional derivative* along a
random direction, which is evaluated on the host, and return the relative error. This checks
the gradient and the device at the same time.

Note that a central difference is not a useful comparison here: the objective is evaluated in
`Float32` and the directional derivative is of the order of ``\\|g\\|/\\sqrt{n}``, so the
cancellation error of the difference quotient is of the same order as the quantity itself.
`ForwardDiff.derivative` needs a single dual number instead — and `ForwardDiff.Dual`s cannot
be multiplied by `cuBLAS`, so this part has to run on the host anyway.
"""
function check_gradient(ps::NamedTuple)
    v, _ = ParameterHandling.flatten(ps)
    g = zeros(T, length(v))
    ∇F!(g, v)
    d = Random.randn(Random.Xoshiro(seed), T, length(v))
    d ./= norm(d)
    input, output = current_batch[]
    host_input, host_output = to_host(input), to_host(output)
    directional_derivative = ForwardDiff.derivative(zero(T)) do t
        network_loss(regroup_host(v + t * d), host_input, host_output)
    end
    abs(directional_derivative - sum(g .* d)) / abs(directional_derivative)
end

# ------------------------------------------------------------- checks on the outcome ---

@doc raw"""
    orthonormality_error(ps)

The worst ``\|Y^TY - I\|`` over the attention projections, i.e. how far the parameters have
drifted off the Stiefel manifold. `NaN` if they are not on a manifold to begin with.
"""
function orthonormality_error(ps::NamedTuple)
    ps[1] isa StiefelManifold || return T(NaN)
    worst = zero(T)
    for i in 1:n_attention_parameters
        A = _array(values(ps)[i])
        worst = max(worst, T(norm(A' * A - I)))
    end
    worst
end

"""
    parameters_are_sound(ps)

Every parameter is finite and still stored in `T`. A retraction that produced a `NaN`, or a
step that silently promoted the parameters to `Float64`, shows up here.
"""
function parameters_are_sound(ps::NamedTuple)
    all(values(ps)) do p
        A = _array(p)
        eltype(A) === T && all(isfinite, A)
    end
end

# ------------------------------------------------------------------------ statistics ---

@doc raw"""
    statistics(samples)

The number of `samples`, their mean, their corrected sample standard deviation

```math
s = \sqrt{\frac{1}{n-1}\sum_i (x_i - \bar{x})^2},
```

and their extremes, as a `NamedTuple`.

The deviation of a single sample is `NaN` rather than `0`: one repetition says nothing about
the spread, and a printed `0.0000` would claim that it does. `NaN` samples — the drift of an
unconstrained run — are dropped, so that a configuration without a manifold does not turn the
statistics of the ones with it into `NaN`.

Written out rather than taken from `Statistics` so that this script runs in exactly the
environment `mnist_cuda.jl` does: adding a dependency to `scripts/Project.toml` re-resolves
the manifest on a workstation that already has a working one, which is not a thing to do to a
machine that is about to be occupied for eight hours.
"""
function statistics(samples::AbstractVector{<:Real})
    finite = filter(isfinite, samples)
    n = length(finite)
    n == 0 && return (n=0, mean=NaN, deviation=NaN, minimum=NaN, maximum=NaN)
    μ = sum(finite) / n
    deviation = n < 2 ? oftype(float(μ), NaN) : sqrt(sum(abs2, finite .- μ) / (n - 1))
    (n=n, mean=float(μ), deviation=deviation,
        minimum=float(Base.minimum(finite)), maximum=float(Base.maximum(finite)))
end

"""
    summarize(samples; format)

`samples` as `n = 5   mean 0.8621   deviation 0.0043   min 0.8560   max 0.8702`, with every
number written by `format`. Empty and single-sample sets are printed rather than skipped: what
a run produced is what the report says, including when that is one number.
"""
function summarize(samples::AbstractVector{<:Real}; format=v -> @sprintf("%.4f", v))
    s = statistics(samples)
    s.n == 0 && return "no samples"
    @sprintf("n = %i   mean %s   deviation %s   min %s   max %s   [%s]",
        s.n, format(s.mean), isnan(s.deviation) ? "—" : format(s.deviation),
        format(s.minimum), format(s.maximum), join(map(format, samples), " "))
end

# -------------------------------------------------------------------------- training ---

"""
    train(stiefel, algorithm, input, output, test_input, test_output; label, run_index, run_name, repetition, seed)

Train one repetition of one configuration and return everything worth keeping about it: the
parameters, the per-batch and per-epoch losses, the per-epoch wall clock, the test accuracies
evaluated along the way and why the run ended.

This is `train` of `mnist_cuda.jl` with the seed as a keyword argument instead of a constant —
which is the whole difference between the two scripts below the run loop — and with the
repetition written into the loss CSV.

One line per epoch goes into the report as the epoch finishes, so the file describes a run
that is still in progress just as well as a finished one. `label` is what those lines are
prefixed with.

A non-finite epoch loss ends the run: the parameters cannot come back from it, and the point
of noticing here is not to spend the remaining epochs proving that.
"""
function train(stiefel::Bool, algorithm::GeometricOptimizers.OptimizerMethod, input, output,
    test_input, test_output; label::AbstractString, run_index::Integer, run_name::AbstractString,
    repetition::Integer, seed::Integer=seed, n_epochs=n_epochs, learning_rate=learning_rate)
    rng = Random.Xoshiro(seed)
    ps = initial_parameters(rng, stiefel)

    n_batches = size(input, 3) ÷ batch_size
    current_batch[] = (to_device(input[:, :, 1:batch_size]), to_device(output[:, 1:batch_size]))

    # Note that the learning rate is supplied through the line search: the *methods* only
    # determine the direction. `Static(learning_rate)` is what `Optimizer` defaults to for
    # these three methods anyway; it is written out so that the rate is visible right here.
    #
    # Both are constructed per repetition, so no cache and no iteration counter survives from
    # the previous one — a repetition has to start where the corresponding run of
    # `mnist_cuda.jl` starts, or the statistics below are of something else.
    optimizer = Optimizer(ps, F; (∇F!)=∇F!, algorithm=algorithm, linesearch=Static(learning_rate))
    state = OptimizerState(algorithm, ps)
    initialize_state!(state)

    losses = T[]
    epoch_losses = T[]
    epoch_times = Float64[]
    accuracy_epochs = Int[]
    accuracies = T[]
    orthonormalities = T[]
    stopped = ""
    peak_used[] = 0

    synchronize_device()
    initial_time = time()
    for epoch in 1:n_epochs
        epoch_start = time()
        # `solve!` cannot be used here: it optimizes a *fixed* objective until it converges,
        # whereas the objective changes with every batch.
        batches = Iterators.take(Iterators.partition(Random.shuffle(rng, axes(input, 3)), batch_size), n_batches)
        epoch_loss = zero(T)
        for (i, batch) in pairs(collect(batches))
            # the batch is gathered on the host and uploaded; at 6.4 MB per batch this is
            # negligible next to the forward and backward passes
            current_batch[] = (to_device(input[:, :, batch]), to_device(output[:, batch]))
            increase_iteration_number!(state)
            solver_step!(ps, state, optimizer)
            GeometricOptimizers.update!(state, optimizer, ps)
            loss = F(ps)
            push!(losses, loss)
            epoch_loss += loss / n_batches
            println(losses_io, run_index, ",\"", run_name, "\",", repetition, ",", epoch, ",", i,
                ",", length(losses), ",", loss)
            to_terminal && @printf("\r  epoch %4i/%i, batch %3i/%i, loss %.5f", epoch, n_epochs, i, n_batches, loss)
        end
        synchronize_device()
        flush(losses_io)        # the CSV is complete to the last finished epoch, as the report is
        push!(epoch_losses, epoch_loss)
        push!(epoch_times, time() - epoch_start)
        note_device_memory()

        line = @sprintf("%s  epoch %4i/%-4i  avg loss %8.5f  last %8.5f  %6.1f s  elapsed %s",
            label, epoch, n_epochs, epoch_loss, last(losses), last(epoch_times),
            duration(time() - initial_time))
        # the last epoch is always evaluated, so that every repetition ends on a measured accuracy
        if accuracy_every > 0 && (epoch % accuracy_every == 0 || epoch == n_epochs)
            score = T(accuracy(ps, test_input, test_output))
            push!(accuracy_epochs, epoch)
            push!(accuracies, score)
            line *= @sprintf("  test accuracy %.4f", score)
            # alongside it, how far off the manifold the parameters have drifted by now. A
            # single number at the end cannot distinguish round-off that accumulates with the
            # step count from a retraction that gave up at some point; the series can.
            drift = orthonormality_error(ps)
            push!(orthonormalities, drift)
            isnan(drift) || (line *= @sprintf("  ‖YᵀY-I‖ %.1e", drift))
        end
        clear_progress()
        announce(line)

        if !isfinite(epoch_loss)
            stopped = "the loss is $epoch_loss after epoch $epoch"
            announce("  " * label * "  stopping this repetition: " * stopped)
            break
        end
    end
    synchronize_device()
    total_time = time() - initial_time

    (parameters=ps, losses=losses, epoch_losses=epoch_losses, epoch_times=epoch_times,
        accuracy_epochs=accuracy_epochs, accuracies=accuracies, orthonormalities=orthonormalities,
        total_time=total_time, stopped=stopped, peak_device_bytes=peak_used[])
end

# ------------------------------------------------------------------ configurations ---

# The four configurations of `mnist_cuda.jl`, by key. `learns` is what the configuration is
# expected to do; standard Adam is the non-geometric ablation and is expected to
# collapse onto the trivial prediction (see the header of `mnist_cuda.jl` for why, and why a
# flat loss there is the experiment working rather than a defect).
#
# `Adam` takes the *element type* of the parameters, not a learning rate, and it is not
# converted the way `MomentumMethod` is, so `Adam(T)` is what dispatches to the `Float32`
# cache.
const configurations = Dict(
    "geometric-adam-cayley" => (name="Geometric Adam (Stiefel, Cayley retraction)",
        role="proposed", stiefel=true, learns=true, algorithm=Adam(T),
        retraction="cayley", second_moment="coordinate-wise", transport="global-section"),
    "standard-adam" => (name="Standard Adam (unconstrained)", role="non-geometric-ablation",
        stiefel=false, learns=false, algorithm=Adam(T), retraction="none",
        second_moment="coordinate-wise", transport="none"),
    "gradient" => (name="Riemannian gradient (Stiefel, Cayley retraction)", role="diagnostic",
        stiefel=true, learns=true, algorithm=GradientMethod(), retraction="cayley",
        second_moment="none", transport="none"),
    "momentum" => (name="Riemannian momentum (Stiefel, Cayley retraction)", role="diagnostic",
        stiefel=true, learns=true, algorithm=MomentumMethod(momentum_coefficient),
        retraction="cayley", second_moment="none", transport="global-section"),
)

# in the order of `mnist_cuda.jl`, so that a report of all four is read next to that one
const configuration_order = ["geometric-adam-cayley", "standard-adam", "gradient", "momentum"]
const configuration_aliases = Dict(
    "adam-stiefel" => "geometric-adam-cayley",
    "adam-regular" => "standard-adam",
)

"""
    selected_configurations()

The configurations `MNIST_CONFIGURATIONS` asks for, in the order of `mnist_cuda.jl`. This is
resolved before MNIST is loaded, so that an unknown key is one line now rather than an empty
run eight hours from now.
"""
function selected_configurations()
    requested = get(ENV, "MNIST_CONFIGURATIONS", "geometric-adam-cayley")
    keys_requested = [get(configuration_aliases, key, key) for key in
                      String.(strip.(split(lowercase(requested), ','; keepempty=false)))]
    "all" in keys_requested && return configuration_order
    unknown = filter(k -> !haskey(configurations, k), keys_requested)
    isempty(unknown) ||
        error("unknown configuration(s) $(join(unknown, ", ")) in MNIST_CONFIGURATIONS — " *
              "known are $(join(configuration_order, ", ")) and `all`")
    isempty(keys_requested) && error("MNIST_CONFIGURATIONS is empty")
    filter(in(keys_requested), configuration_order)
end

const selected = selected_configurations()
const dataset_name = lowercase(get(ENV, "MNIST_DATASET", "mnist"))
dataset_name in ("mnist", "fashion-mnist") ||
    error("MNIST_DATASET must be `mnist` or `fashion-mnist`, got `$dataset_name`")

# One entry per training, in the order they are run: all repetitions of a configuration before
# the next one, so that a run which is cut short has complete statistics for the configurations
# it did reach rather than one repetition of each.
const jobs = [(key=key, configuration=configurations[key], repetition=r, seed=repetition_seed(r))
              for key in selected for r in 1:n_repetitions]

# ------------------------------------------------------------------------------ data ---

const script_start = time()

println("loading $dataset_name ...")
dataset = dataset_name == "mnist" ? MLDatasets.MNIST : MLDatasets.FashionMNIST
train_x, train_y = dataset(split=:train)[:]
test_x, test_y = dataset(split=:test)[:]

# the data set stays on the host; the batches are uploaded one at a time
const train_input = split_and_flatten(T.(train_x), patch_length)
const train_output = onehotbatch(train_y)
const test_input = split_and_flatten(T.(test_x), patch_length)
const test_output = onehotbatch(test_y)

@assert size(train_input) == (dim, seq_length, size(train_x, 3))

const n_batches = size(train_input, 3) ÷ batch_size

# ------------------------------------------------------------------- what will be run ---

announce(@sprintf("%i parameters, batch size %i, %i batches per epoch, %i epochs per repetition (%i steps)",
    n_parameters, batch_size, n_batches, n_epochs, n_epochs * n_batches))
announce(@sprintf("%i repetition(s) of %i configuration(s) — %i trainings, %i steps in total",
    n_repetitions, length(selected), length(jobs), length(jobs) * n_epochs * n_batches))
announce(@sprintf("configurations: %s", join((configurations[k].name for k in selected), "; ")))
announce(vary_seed ?
         @sprintf("seeds: %s (one per repetition)", join((repetition_seed(r) for r in 1:n_repetitions), ", ")) :
         @sprintf("seed: %i for every repetition — this measures the nondeterminism, not the spread of the method", seed))
accuracy_every > 0 && announce(@sprintf("the test accuracy is evaluated every %i epoch(s)", accuracy_every))
announce()

# -------------------------------------------------------------------- gradient check ---

# As in `mnist_cuda.jl`, both cases are checked: the unconstrained configuration is expected to
# stall, and the check separates the stall the experiment is about — a vanishing gradient in
# the network — from a wrong `∇F!` on plain arrays, which would look the same from the outside.
current_batch[] = (to_device(train_input[:, :, 1:batch_size]), to_device(train_output[:, 1:batch_size]))

const gradient_errors = map((true, false)) do stiefel
    # not `error`, which would shadow `Base.error` for the rest of the closure
    relative_error = check_gradient(initial_parameters(Random.Xoshiro(seed), stiefel))
    announce(@sprintf("relative error of ∇F! vs a host directional derivative, %s weights: %.2e",
        stiefel ? "Stiefel" : "regular", relative_error))
    relative_error
end

const gradient_ok = all(e -> isfinite(e) && e < gradient_tolerance, gradient_errors)
if !gradient_ok
    announce(@sprintf("*** the gradient disagrees with the host by more than %.0e — the trainings below are not meaningful ***",
        gradient_tolerance))
    @warn "∇F! disagrees with the host directional derivative"
end
announce()

# ------------------------------------------------------------------------------- run ---

"""
    save_results(results)

Write what has been trained so far to `output_path`. Called after every repetition rather than
once at the end: a file that only exists when all of them have finished is worth nothing to a
run that dies in the last one.

The keys are those of `mnist_cuda.jl` — `parameters1`, `losses1`, `total_time1`, `accuracy1`,
… — one group per *training*, plus a `configuration`, a `repetition` and a `seed` per group to
say which training that was, and the `accuracy_samples`/`drift_samples`/`final_loss_samples`/
`time_samples` dictionaries the statistics of the report are computed from. Anything that reads
the output of `mnist_cuda.jl` therefore reads this file as well, and gets the repetitions as if
they were separate configurations.
"""
function save_results(results)
    output = Dict{String,Any}("n_epochs" => n_epochs, "batch_size" => batch_size,
        "n_batches" => n_batches, "gradient_errors" => collect(gradient_errors),
        "n_results" => length(results), "n_repetitions" => n_repetitions,
        "vary_seed" => vary_seed, "seed" => seed,
        "configurations" => [configurations[k].name for k in selected])
    for (i, result) in pairs(results)
        output["name$i"] = result.name
        output["configuration$i"] = result.configuration
        output["optimizer_role$i"] = result.optimizer_role
        output["retraction$i"] = result.retraction
        output["second_moment$i"] = result.second_moment
        output["transport$i"] = result.transport
        output["repetition$i"] = result.repetition
        output["seed$i"] = result.seed
        output["parameters$i"] = result.parameters
        output["losses$i"] = result.losses
        output["epoch_losses$i"] = result.epoch_losses
        output["epoch_times$i"] = result.epoch_times
        output["accuracy_epochs$i"] = result.accuracy_epochs
        output["accuracies$i"] = result.accuracies
        output["orthonormalities$i"] = result.orthonormalities
        output["orthonormality$i"] = result.orthonormality
        output["total_time$i"] = result.total_time
        output["accuracy$i"] = result.accuracy
    end
    for (key, of) in ("accuracy_samples" => (r -> r.accuracy),
        "drift_samples" => (r -> r.orthonormality),
        "final_loss_samples" => (r -> isempty(r.epoch_losses) ? T(NaN) : last(r.epoch_losses)),
        "time_samples" => (r -> r.total_time))
        output[key] = Dict(configurations[k].name =>
            [of(r) for r in results if r.configuration == configurations[k].name] for k in selected)
    end
    JLD2.save(output_path, output)
end

csv_field(value) = "\"" * replace(string(value), '"' => "\"\"") * "\""

function write_run_records(results, failures)
    open(records_path, "w") do io
        println(io, "schema_version,dataset,configuration,optimizer_role,retraction,second_moment,transport,repetition,seed,status,epochs_completed,final_loss,best_loss,test_accuracy,total_seconds,seconds_per_epoch,peak_device_bytes,backend,message")
        for result in results
            epochs_completed = length(result.epoch_losses)
            final_loss = isempty(result.epoch_losses) ? NaN : last(result.epoch_losses)
            best_loss = isempty(result.epoch_losses) ? NaN : minimum(result.epoch_losses)
            result_verdict = verdict(result)
            status = result_verdict == "ok" ? "ok" : "failed_validation"
            println(io, join((2, csv_field(dataset_name), csv_field(result.configuration),
                csv_field(result.optimizer_role), csv_field(result.retraction),
                csv_field(result.second_moment), csv_field(result.transport),
                result.repetition, result.seed, status, epochs_completed, final_loss, best_loss,
                result.accuracy, result.total_time,
                result.total_time / max(epochs_completed, 1), result.peak_device_bytes,
                use_cuda ? "cuda" : "cpu", csv_field(result_verdict)), ','))
        end
        for (name, message) in failures
            println(io, join((2, csv_field(dataset_name), csv_field(name), "", "", "", "", 0, 0, "exception", 0,
                NaN, NaN, NaN, NaN, NaN, peak_used[], use_cuda ? "cuda" : "cpu",
                csv_field(first(split(message, '\n')))), ','))
        end
    end
end

results = []
failures = Tuple{String,String}[]

for (j, job) in pairs(jobs)
    run = job.configuration
    # the repetition is part of the label, so that the epoch lines of the report say which
    # training they belong to — with five repetitions of one configuration they are otherwise
    # 2500 indistinguishable lines
    name = n_repetitions > 1 ? @sprintf("%s #%i", run.name, job.repetition) : run.name
    label = @sprintf("[%i/%i %-28s]", j, length(jobs), name)
    announce(@sprintf("%s starting at %s, seed %i", label, timestamp(), job.seed))
    try
        trained = train(run.stiefel, run.algorithm, train_input, train_output, test_input, test_output;
            label=label, run_index=j, run_name=run.name, repetition=job.repetition, seed=job.seed,
            n_epochs=n_epochs)
        score = T(accuracy(trained.parameters, test_input, test_output))
        push!(results, (name=name, configuration=run.name, repetition=job.repetition, seed=job.seed,
            stiefel=run.stiefel, learns=run.learns, optimizer_role=run.role,
            retraction=run.retraction, second_moment=run.second_moment, transport=run.transport,
            parameters=map(_array, trained.parameters), losses=trained.losses,
            epoch_losses=trained.epoch_losses, epoch_times=trained.epoch_times,
            accuracy_epochs=trained.accuracy_epochs, accuracies=trained.accuracies,
            orthonormalities=trained.orthonormalities,
            total_time=trained.total_time, accuracy=score, stopped=trained.stopped,
            peak_device_bytes=trained.peak_device_bytes,
            orthonormality=orthonormality_error(trained.parameters),
            sound=parameters_are_sound(trained.parameters)))
        announce(@sprintf("%s done in %s (%.2f s/step), test accuracy %.4f", label,
            duration(trained.total_time),
            trained.total_time / max(1, length(trained.losses)), score))
        save_results(results)
        announce(@sprintf("%s written to %s", label, output_path))
    catch e
        # An `InterruptException` is the one exception that means *stop*, not *skip*; anything
        # else is this repetition's problem and the remaining ones still deserve the GPU. A
        # repetition that threw is missing from the statistics, and the verdict says so.
        e isa InterruptException && rethrow()
        message = sprint(showerror, e)
        push!(failures, (name, message))
        clear_progress()
        announce(@sprintf("%s FAILED: %s", label, first(split(message, '\n'))))
        report(message)
        report(sprint(Base.show_backtrace, catch_backtrace()))
        @error "$name failed" exception = (e, catch_backtrace())
    end
    announce()
end

# --------------------------------------------------------------------------- verdict ---

"""
    verdict(result)

Whether one repetition did what its configuration is supposed to do, and if not, why. This is
`verdict` of `mnist_cuda.jl`, applied per repetition.

Two criteria apply to every run — the parameters survived and, on a manifold, stayed on it.
The remaining two depend on `learns`: a configuration that is expected to learn has to reduce
its loss and classify above `accuracy_floor`, while the unconstrained one is expected to
collapse onto the trivial prediction and is checked against that plateau instead.

The bars are those of the *full* run. A short schedule is judged on what it can show: the loss
decrease is read off the per-batch series when there are not two epochs to compare, and the
accuracy is not judged at all below `accuracy_floor_min_epochs`, where a configuration that is
working still has no chance of clearing the floor.
"""
function verdict(result)
    isempty(result.epoch_losses) && return "FAILED: no epoch finished"
    reasons = String[]
    isempty(result.stopped) || push!(reasons, result.stopped)
    result.sound || push!(reasons, "parameters not finite or no longer $T")
    if result.stiefel && !(result.orthonormality ≤ orthonormality_tolerance)
        push!(reasons, @sprintf("off the manifold (‖YᵀY-I‖ = %.1e)", result.orthonormality))
    end
    # With a single epoch `first(epoch_losses)` *is* `last(epoch_losses)`, so the decrease has
    # to be read off the batches instead — otherwise every short run is "flat" by construction.
    first_loss = length(result.epoch_losses) ≥ 2 ? first(result.epoch_losses) : first(result.losses)
    last_loss = last(result.epoch_losses)
    if result.learns
        if !(last_loss < (1 - loss_decrease_floor) * first_loss)
            push!(reasons, @sprintf("loss flat (%.3f → %.3f)", first_loss, last_loss))
        end
        if length(result.epoch_losses) ≥ accuracy_floor_min_epochs && result.accuracy < accuracy_floor
            push!(reasons, @sprintf("accuracy %.4f below the floor of %.2f%s", result.accuracy,
                accuracy_floor, result.accuracy < 2 * chance_accuracy ? ", i.e. at chance" : ""))
        end
    elseif !(abs(last_loss - trivial_loss) ≤ trivial_loss_tolerance)
        push!(reasons, @sprintf("expected the trivial-prediction plateau of %.3f, got %.3f",
            trivial_loss, last_loss))
    end
    isempty(reasons) ? "ok" : "FAILED: " * join(reasons, "; ")
end

const verdicts = map(verdict, results)

announce("=" ^ 100)
announce(@sprintf("finished %s after %s — %i repetition(s) × %i configuration(s), %i epochs (%i steps) each, batch size %i",
    timestamp(), duration(time() - script_start), n_repetitions, length(selected), n_epochs,
    n_epochs * n_batches, batch_size))
announce()
announce(@sprintf("%-29s %6s %8s %8s %10s %10s %10s   %s",
    "training", "seed", "loss 1", "loss N", "accuracy", "‖YᵀY-I‖", "time", "verdict"))
# `epoch_losses` is empty only if a run never finished an epoch; the verdict says so, and the
# table has to stay printable rather than throw at the end of a day of work
loss_cell(losses, which) = isempty(losses) ? "—" : @sprintf("%8.3f", which(losses))

for (result, v) in zip(results, verdicts)
    n_done = length(result.epoch_losses)
    announce(@sprintf("%-29s %6i %8s %8s %10.4f %10s %10s   %s", result.name, result.seed,
        loss_cell(result.epoch_losses, first), loss_cell(result.epoch_losses, last), result.accuracy,
        isnan(result.orthonormality) ? "—" : (@sprintf "%.1e" result.orthonormality),
        duration(result.total_time),
        v * (n_done < n_epochs ? " [$n_done of $n_epochs epochs]" : "")))
end
for (name, message) in failures
    announce(@sprintf("%-29s %6s %8s %8s %10s %10s %10s   THREW: %s", name, "—", "—", "—", "—", "—", "—",
        first(split(message, '\n'))))
end

# ------------------------------------------------------------------------ statistics ---

# What this script exists for. Every quantity is reported over the repetitions that finished,
# with the individual samples in brackets — a mean of five accuracies which is worth quoting
# and a mean of two which is not look identical otherwise.

"""
    samples_of(configuration, of)

`of` applied to every finished repetition of `configuration`, in the order they ran.
"""
samples_of(configuration::AbstractString, of::Base.Callable) =
    [of(r) for r in results if r.configuration == configuration]

announce()
announce(@sprintf("over the repetitions%s —", vary_seed ? "" : " (same seed, so this is the nondeterminism alone)"))
for key in selected
    configuration = configurations[key].name
    accuracies = samples_of(configuration, r -> r.accuracy)
    isempty(accuracies) && continue
    announce()
    announce(configuration)
    announce("  test accuracy     " * summarize(accuracies))
    announce("  final epoch loss  " *
             summarize(samples_of(configuration, r -> isempty(r.epoch_losses) ? T(NaN) : last(r.epoch_losses));
                 format=v -> @sprintf("%.4f", v)))
    drifts = samples_of(configuration, r -> r.orthonormality)
    any(isfinite, drifts) &&
        announce("  ‖YᵀY-I‖          " * summarize(drifts; format=v -> @sprintf("%.2e", v)))
    announce("  time              " *
             summarize(samples_of(configuration, r -> r.total_time); format=duration))
end

# The learning curves of every repetition, so that the shape of each is in the report and not
# only in the `.jld2`. With repetitions this is also where a spread that is present from the
# first evaluation can be told from one that opens up late.
if accuracy_every > 0
    announce()
    announce("test accuracy over the run, as epoch:accuracy —")
    for result in results
        isempty(result.accuracies) && continue
        announce(@sprintf("%-29s %s", result.name,
            join((@sprintf("%i:%.4f", e, a) for (e, a) in zip(result.accuracy_epochs, result.accuracies)), "  ")))
    end

    # The same series for the drift off the manifold. Whether this grows with the step count or
    # settles is the difference between accumulated round-off and a retraction that stopped
    # working, and it cannot be read off the single number in the table above.
    announce()
    announce("‖YᵀY-I‖ over the run, as epoch:drift —")
    for result in results
        (isempty(result.orthonormalities) || all(isnan, result.orthonormalities)) && continue
        announce(@sprintf("%-29s %s", result.name,
            join((@sprintf("%i:%.1e", e, d) for (e, d) in zip(result.accuracy_epochs, result.orthonormalities)), "  ")))
    end
end

const n_passed = count(==("ok"), verdicts)
const threw = isempty(failures) ? "" : @sprintf(", %i threw", length(failures))
const memory_note = use_cuda ? @sprintf(" Peak device memory %i MB of %i MB.",
    peak_used[] ÷ 1024^2, Int(CUDA.totalmem(CUDA.device())) ÷ 1024^2) : ""

announce()
announce(@sprintf("%i of %i trainings behave as expected%s. Gradient check %s.%s",
    n_passed, length(jobs), threw, gradient_ok ? "ok" : "FAILED", memory_note))
announce(@sprintf("bars applied per repetition: ‖YᵀY-I‖ ≤ %.0e, loss down by ≥ %.0f%%, %s.",
    orthonormality_tolerance, 100 * loss_decrease_floor,
    n_epochs ≥ accuracy_floor_min_epochs ?
    @sprintf("accuracy ≥ %.2f", accuracy_floor) :
    @sprintf("accuracy reported but not judged below %i epochs", accuracy_floor_min_epochs)))
announce()

# the one-line answer, for the configurations that have more than one sample
for key in selected
    configuration = configurations[key].name
    s = statistics(samples_of(configuration, r -> r.accuracy))
    s.n == 0 && continue
    announce(s.n < 2 ?
             @sprintf("%s: test accuracy %.4f from a single repetition — not a statistic.",
        configuration, s.mean) :
             @sprintf("%s: test accuracy %.4f ± %.4f over %i repetitions.",
        configuration, s.mean, s.deviation, s.n))
end
if any(r -> !r.learns, results)
    announce()
    announce("`Standard Adam (unconstrained)` is the non-geometric ablation and is expected to stall at the")
    announce(@sprintf("trivial-prediction plateau of %.3f at an accuracy of %.2f; without the Stiefel constraint",
        trivial_loss, chance_accuracy))
    announce("the gradient vanishes through the 16 unnormalized transformer blocks. It is a failure above")
    announce("only if it did something else.")
end

save_results(results)
write_run_records(results, failures)
announce()
announce("parameters:   " * abspath(output_path))
announce("loss curves:  " * abspath(losses_path))
announce("run records:  " * abspath(records_path))
announce("this report:  " * abspath(report_path))
close(losses_io)
close(report_io)

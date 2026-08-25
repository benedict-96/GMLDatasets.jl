using CUDA
using GeometricMachineLearning
using GeometricOptimizers
using Pkg

required_name = get(ENV, "GML_REQUIRED_GPU", "RTX 4090")
allow_any_gpu = parse(Bool, get(ENV, "GML_ALLOW_ANY_GPU", "0"))
allow_no_cuda = parse(Bool, get(ENV, "GML_ALLOW_NO_CUDA", "0"))
functional = CUDA.functional()
functional || allow_no_cuda || error("CUDA.functional() is false")
device_name = functional ? CUDA.name(CUDA.device()) : "none"
functional && !allow_any_gpu && !occursin(required_name, device_name) &&
    error("expected a GPU containing `$required_name`, found `$device_name`; set GML_ALLOW_ANY_GPU=1 only for deliberate testing")

println("julia_version=", VERSION)
println("active_project=", Base.active_project())
println("device=", device_name)
println("driver_version=", functional ? CUDA.driver_version() : "unavailable")
println("runtime_version=", functional ? CUDA.runtime_version() : "unavailable")
println("threads=", Threads.nthreads())
gml_version = pkgversion(GeometricMachineLearning)
go_version = pkgversion(GeometricOptimizers)
println("geometric_machine_learning_version=", gml_version)
println("geometric_optimizers_version=", go_version)

if gml_version ≥ v"0.6.0" && go_version ≥ v"0.5.0"
    error("GeometricMachineLearning $gml_version and GeometricOptimizers $go_version are not a " *
          "released compatible stack: GML 0.6 declares GeometricOptimizers 0.4 and does not " *
          "yet integrate ScalarMomentAdam. Pin a reviewed GML compatibility update before " *
          "running revision experiments.")
end
Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
functional && CUDA.versioninfo()

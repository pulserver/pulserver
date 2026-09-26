# Simulates, for every case directory named after the phantom, the design's
# files, the file exported from its cache and that file in KomaMRI's RF
# convention, and writes the three signals there.
#
# A case directory holds `design.txt`, the design's files in play order, one
# per line, `exported.seq` and `komamri.seq`; it receives `design.sig`,
# `exported.sig` and `komamri.sig`: every ADC sample in play order, as
# little-endian complex128, without the ADC's offsets applied.
using KomaMRICore, KomaMRIFiles

"""The phantom a file holds: the spin count, then x, y, z (m), ρ, T1, T2 (s) and Δw (rad/s)."""
function phantom(path)
    open(path) do io
        n = read(io, Int64)
        x, y, z, ρ, T1, T2, Δw = (read!(io, Vector{Float64}(undef, n)) for _ in 1:7)
        Phantom(; name="pulserver", x, y, z, ρ, T1, T2, Δw)
    end
end

"""
The blocks of `file`, each block's gradients turned by its rotation extension.

KomaMRI turns a block by adding up its rotated gradients and simplifies the
sum, dropping both samples of a corner held twice at one time, or at two times
closer than rounding, such as the peak of a trapezoid without a flat top
(KomaMRIBase 0.14). The gradients are turned here instead, and kept at every
corner of the three.
"""
function turned(file)
    seq = read_seq(file; verbose=false, apply_rotations=false)
    GR = Matrix{Grad}(undef, size(seq.GR))
    GR .= seq.GR
    for block in axes(GR, 2), ext in seq.EXT[block]
        ext isa KomaMRIBase.QuaternionRot || continue
        GR[:, block] = turn(KomaMRIBase.rotation_matrix(ext), GR[:, block])
    end
    return Sequence(GR, seq.RF, seq.ADC, seq.DUR, seq.EXT, seq.DEF)
end

"""The gradients `grads` turned by `R`, joined linearly through all their corners, timed to 1 ns."""
function turn(R, grads)
    samples = map(grads) do g
        s = KomaMRIBase.event_samples(g)
        (t=round.(s.t; digits=9), A=s.A)
    end
    t = KomaMRIBase.merge_sampling_times((s.t for s in samples)...)
    waves = [KomaMRIBase.linear_interpolate_samples(s, t; default=0.0) for s in samples]
    return map(axes(R, 1)) do axis
        a = sum(R[axis, k] .* waves[k] for k in eachindex(waves))
        all(iszero, a) ? Grad(0.0, 0.0) : Grad(a, diff(t), 0.0, 0.0, t[1], a[1], a[end])
    end
end

"""The blocks of `files` in turn, every ADC without its offsets."""
function played(files)
    seq = reduce(+, (turned(file) for file in files))
    for adc in seq.ADC
        adc.Δf = 0.0
        adc.ϕ = 0.0
    end
    return seq
end

# Every RF and gradient sample is a simulation time, so both files are
# simulated through the points they hold.
params() = Dict{String,Any}(
    "return_type" => "mat",
    "precision" => "f64",
    "preserve_samples" => (:rf, :gradients),
    "Δt_rf" => 1e-5,
)

function main(args)
    obj = phantom(args[1])
    sys = Scanner()
    for case in args[2:end]
        design = readlines(joinpath(case, "design.txt"))
        exported = [joinpath(case, "exported.seq")]
        komamri = [joinpath(case, "komamri.seq")]
        for (name, files) in (("design", design), ("exported", exported), ("komamri", komamri))
            elapsed = @elapsed signal = simulate(
                obj, played(files), sys; sim_params=params(), verbose=false
            )
            write(joinpath(case, name * ".sig"), ComplexF64.(vec(signal)))
            println(basename(case), ": ", name, ", ", length(signal), " samples in ",
                    round(elapsed; digits=1), " s")
        end
    end
end

main(ARGS)

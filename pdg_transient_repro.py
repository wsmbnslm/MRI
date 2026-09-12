"""
Repro: PDG (graph solver) loses the inversion-recovery transient in a
phase-cycled bSSFP (TrueFISP) sequence; the time-domain spin simulator
(isochromat_sim) reproduces the expected signed, T1-differential null.

Sequence: 180 deg inversion (TI = 20 ms) followed by 700 TrueFISP spokes,
TR = 4 ms, FA = 60 deg, 64 samples/spoke, 180 deg RF/ADC phase cycling
every TR (Ehses et al., MRM 69:71-81, 2013). On-resonant (B0 = 0),
T2' dephasing disabled (T2dash = 1e3 s), single isochromat per voxel.

Phantom: one voxel per tissue, mean T1/T2/PD of the MNI 1.5 T brain
phantom (as returned by MRzeroCore's load_phantom):
    WM:  T1 = 849 ms,  T2 = 72.9 ms, PD = 0.685
    GM:  T1 = 1389 ms, T2 = 102.3 ms, PD = 0.769
    CSF: T1 = 3032 ms, T2 = 388.6 ms, PD = 0.873

Requirements: MRzeroCore (tested with 1.0.8), pypulseq, torch, numpy.
Runtime: ~5-10 min on CPU. Writes us_trueFISP_2D_Radial.seq to the CWD.
"""
import numpy as np
import torch
import pypulseq as pp
import MRzeroCore as mr0
from MRzeroCore import VoxelGridPhantom

SYSTEM = pp.Opts(
    max_grad=28, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s",
    rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6,
)

N_READOUT, N_X = 700, 64
TR = 4.0e-3


def truefisp_seq(FOV=220e-3, N_x=64, alpha=60.0, slice_thickness=6e-3, TRF=0.8e-3,
                 TBW=2.0, BW=890.0, TR=4.0e-3, pre_dur=0.6e-3, N_readout=700,
                 N_spokes=1, R=1, N_dummy=0, TI_start=20e-3, TI_inter=0.0,
                 TD=2000e-3, write_seq=True):
    system = SYSTEM
    seq = pp.Sequence(system)
    deltak = 1 / FOV
    _gr = seq.grad_raster_time

    rf_inv, gz_inv, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(180), duration=10e-3, slice_thickness=slice_thickness,
        apodization=0.5, time_bw_product=4, system=system,
        return_gz=True, use="inversion",
    )
    rf, gz, _ = pp.make_sinc_pulse(
        apodization=0.5, duration=TRF, flip_angle=np.deg2rad(alpha),
        slice_thickness=slice_thickness, system=system,
        time_bw_product=TBW, return_gz=True, use="excitation",
    )

    _astep = N_x * seq.adc_raster_time
    _k = 1
    while abs(_k * _gr - round(_k * _gr / _astep) * _astep) > 1e-9:
        _k += 1
    readout_flat = int(round((1.0 / BW) / (_k * _gr))) * (_k * _gr)
    gx_base = pp.make_trapezoid(
        channel="x", flat_area=N_x * deltak, flat_time=readout_flat, system=system
    )
    adc = pp.make_adc(
        num_samples=N_x, duration=gx_base.flat_time,
        delay=gx_base.rise_time, system=system,
    )

    pre_dur = int(np.ceil(pre_dur / _gr)) * _gr
    gx_pre_base = pp.make_trapezoid(
        channel="x", area=-gx_base.area / 2, duration=pre_dur, system=system
    )
    gz_reph = pp.make_trapezoid(
        channel="z", area=-gz.area / 2, duration=pre_dur, system=system
    )

    _fixed = (
        TRF + 2 * pre_dur + gx_base.flat_time + gx_base.rise_time + gx_base.fall_time
        + 2 * (gx_pre_base.rise_time + gx_pre_base.fall_time)
    )
    delay_TE = int(np.ceil(max(1e-4, TR - _fixed) / _gr)) * _gr
    TE = pre_dur + TRF / 2 + delay_TE + gx_base.flat_time / 2
    inv_duration = pp.calc_duration(rf_inv, gz_inv)
    delay_TI_start = (
        np.ceil((TI_start - inv_duration / 2 - pp.calc_duration(gz) / 2) / seq.grad_raster_time)
        * seq.grad_raster_time
    )
    assert delay_TI_start > 0, "TI_start too short"

    delay_TD = np.ceil(TD / seq.grad_raster_time) * seq.grad_raster_time
    delay_TI_inter = np.ceil(TI_inter / seq.grad_raster_time) * seq.grad_raster_time

    GA_DEG = 111.246
    spoke_counter = 0
    tr_counter = 0
    for seg in range(N_spokes // R):
        seq.add_block(rf_inv, gz_inv)
        seq.add_block(pp.make_delay(delay_TI_start))
        n_this_seg = N_readout + (N_dummy if seg == 0 else 0)
        for s in range(n_this_seg):
            is_dummy = (seg == 0) and (s < N_dummy)
            # TrueFISP phase cycling: alternate RF/ADC phase by 180 deg every TR
            phase = (tr_counter % 2) * np.pi
            rf.phase_offset = phase
            adc.phase_offset = phase
            angle = 0.0 if is_dummy else np.deg2rad(spoke_counter * GA_DEG)
            if not is_dummy:
                spoke_counter += 1
            kx, ky = np.cos(angle), np.sin(angle)
            gx_pre = pp.make_trapezoid(channel="x", area=gx_pre_base.area * kx, duration=pre_dur, system=system)
            gy_pre = pp.make_trapezoid(channel="y", area=gx_pre_base.area * ky, duration=pre_dur, system=system)
            gx_rew = pp.make_trapezoid(channel="x", area=gx_pre_base.area * kx, duration=pre_dur, system=system)
            gy_rew = pp.make_trapezoid(channel="y", area=gx_pre_base.area * ky, duration=pre_dur, system=system)
            seq.add_block(rf, gz)
            seq.add_block(gx_pre, gy_pre, gz_reph)
            seq.add_block(pp.make_delay(delay_TE))
            gx = pp.make_trapezoid(channel="x", amplitude=gx_base.amplitude * kx,
                                   flat_time=gx_base.flat_time, rise_time=gx_base.rise_time,
                                   fall_time=gx_base.fall_time, system=system)
            gy = pp.make_trapezoid(channel="y", amplitude=gx_base.amplitude * ky,
                                   flat_time=gx_base.flat_time, rise_time=gx_base.rise_time,
                                   fall_time=gx_base.fall_time, system=system)
            if is_dummy:
                seq.add_block(gx, gy)
            else:
                seq.add_block(gx, gy, adc)
            seq.add_block(gx_rew, gy_rew, gz_reph)
            if delay_TI_inter > 0 and s < n_this_seg - 1:
                seq.add_block(pp.make_delay(delay_TI_inter))
            tr_counter += 1
        seq.add_block(pp.make_delay(TD))

    ok, err = seq.check_timing()
    if not ok:
        raise RuntimeError(f"timing FAILED: {err}")
    if write_seq:
        seq.write("us_trueFISP_2D_Radial.seq")
    return seq


def make_phantom(t1, t2, pd):
    ones = lambda: torch.ones((1, 1, 1), dtype=torch.float32)
    ones4 = lambda: torch.ones((1, 1, 1, 1), dtype=torch.float32)
    return VoxelGridPhantom(
        PD=pd * ones(), T1=t1 * ones(), T2=t2 * ones(),
        T2dash=torch.full((1, 1, 1), 1e3, dtype=torch.float32),  # ~no T2' dephasing
        D=torch.zeros((1, 1, 1), dtype=torch.float32),
        B0=torch.zeros((1, 1, 1), dtype=torch.float32),
        B1=ones4(), coil_sens=ones4(),
        affine=torch.eye(4, dtype=torch.float32),  # 1 mm voxel at the FOV centre
    ).build()


def per_spoke_center(signal):
    # signal: (N_readout * N_x, 1) complex -> real part of the centre sample per spoke
    return signal.reshape(N_READOUT, N_X, -1)[:, 32, 0].real.cpu().numpy()


def first_crossing_ms(s):
    # first negative -> positive zero crossing; time approx (spoke + 1) * TR
    idx = np.where((s[:-1] < 0) & (s[1:] > 0))[0]
    return f"{(idx[0] + 1 + 1) * TR * 1e3:.0f} ms" if idx.size else "none"


def main():
    truefisp_seq(write_seq=True)
    mr0_seq = mr0.Sequence.import_file("us_trueFISP_2D_Radial.seq")

    tissues = {
        "WM": (0.849, 0.0729, 0.685),
        "GM": (1.389, 0.1023, 0.769),
        "CSF": (3.032, 0.3886, 0.873),
    }

    print("=== A/B: spin simulator vs PDG (default settings) ===")
    for name, (t1, t2, pd) in tissues.items():
        data = make_phantom(t1, t2, pd)
        torch.manual_seed(0)
        sig_s = mr0.isochromat_sim(mr0_seq, data, spin_count=1, spin_dist="r2",
                                   r2_seed=torch.tensor([0.5, 0.5, 0.5]), print_progress=False)
        graph = mr0.compute_graph(mr0_seq, data)  # defaults: msc=200, min_state_mag=1e-4
        sig_g = mr0.execute_graph(graph, mr0_seq, data, print_progress=False)  # gates: 1e-2
        s, g = per_spoke_center(sig_s), per_spoke_center(sig_g)
        print(f"{name}: spin sim: start={s[0]:+.3f}  null={first_crossing_ms(s)}  SS={s[-1]:+.3f}")
        print(f"{name}: PDG     : start={g[0]:+.3f}  null={first_crossing_ms(g)}  SS={g[-1]:+.3f}")

    print("\n=== PDG max_state_count sweep (GM, default signal gates) ===")
    data = make_phantom(*tissues["GM"])
    for msc in (200, 500, 1000, 1500):
        graph = mr0.compute_graph(mr0_seq, data, max_state_count=msc)
        sig_g = mr0.execute_graph(graph, mr0_seq, data, print_progress=False)
        g = per_spoke_center(sig_g)
        print(f"msc={msc:5d}: null={first_crossing_ms(g)}  SS={g[-1]:+.3f}")

    print("\n=== PDG with the 'accurate' settings from the state-selection docs (GM) ===")
    print("NOTE: the main pass runs at ~1 it/s for this sequence (verified on a CPU),")
    print("i.e. ~2 h per tissue. Abort after ~1 min; the progress rate is the point.")
    graph = mr0.compute_graph(mr0_seq, data, max_state_count=5000, min_state_mag=1e-12)
    sig_g = mr0.execute_graph(graph, mr0_seq, data,
                              min_emitted_signal=1e-3, min_latent_signal=1e-3,
                              print_progress=True)
    g = per_spoke_center(sig_g)
    print(f"msc=5000, gates=1e-3: null={first_crossing_ms(g)}  SS={g[-1]:+.3f}")


if __name__ == "__main__":
    main()

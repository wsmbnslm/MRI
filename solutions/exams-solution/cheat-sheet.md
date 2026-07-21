# MRI II Oral Exam — Coding Reference

This sheet is a **pure syntax reference** for the tools used in the
practical exam (PyPulseq, NumPy) — it exists only to compensate for
not having internet access during the exam. It intentionally does
**not** cover MRI physics, sequence design logic, or any of the
concepts tested in the oral debrief — that is expected knowledge,
not something to look up.

## PyPulseq syntax

```python
rf = pp.make_sinc_pulse(
    flip_angle=np.deg2rad(flip_angle_deg),
    system=system,
    use="excitation",
)

# by total area (1/m)
pp.make_trapezoid(channel="x", area=..., system=system)

# by flat-top area + duration
pp.make_trapezoid(
    channel="x", flat_area=..., flat_time=..., system=system
)

pp.make_adc(
    num_samples=n_x,
    duration=gx.flat_time,
    delay=gx.rise_time,
    system=system,
)

# duration of any block
pp.calc_duration(block)

# time from RF start to RF peak
pp.calc_rf_center(rf)[0]

# single block
seq.add_block(a)

# a and b played simultaneously (same time slot)
seq.add_block(a, b)

seq.add_block(pp.make_delay(t))

# returns (ok: bool, error_report: list)
seq.check_timing()

seq.write(filename, v141_compat=True)
```

**Units:** `area` (or `flat_area`) for a gradient is in **1/m** — a
k-space displacement.

## NumPy syntax used across the notebooks

```python
# degrees -> radians
np.deg2rad(x)

# n evenly spaced values, excluding stop
np.linspace(start, stop, num=n, endpoint=False)

# standard trig functions
np.cos(theta), np.sin(theta)

# round up
np.ceil(x)
```
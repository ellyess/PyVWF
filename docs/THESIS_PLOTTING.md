# Thesis Plotting Reference

## 1. Plot Style

All thesis figures use `vwf.viz.style`, which applies a consistent matplotlib
rcParams configuration via `thesis_plot_style()`. Palettes live in
`vwf.viz.palettes`.

```python
from vwf.viz.style import thesis_plot_style

style = thesis_plot_style()
cm = style['cm']      # 1/2.54 (inch-to-cm conversion)
lw = style['lw']      # 1.2  (default line width)
ms = style['ms']      # 3.5  (default marker size)
dpi = style['dpi']    # 600  (publication DPI)
```

### Style parameters applied globally

| Parameter | Value | Notes |
|-----------|-------|-------|
| Font family | Serif | Print-ready |
| Base font size | 7 pt | Labels, legends |
| Tick label size | 6 pt | Axis ticks |
| DPI | 600 | Publication standard |
| Spines | Bottom + left only | No top/right |
| Axes linewidth | 0.8 pt | Thin, clean |
| Grid | alpha=0.3, linewidth=0.5 | Subtle |
| Font encoding | Type 42 (TrueType) | PDF-safe |

---

## 2. Usage Pattern

```python
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from vwf.viz.style import thesis_plot_style

style = thesis_plot_style()
cm = style['cm']

fig, ax = plt.subplots(figsize=(16 * cm, 10 * cm))

# ... plotting code ...

fig.savefig("output.png", dpi=style['dpi'], bbox_inches='tight')
plt.close(fig)
```

### Colormap conventions

- **Scalar corrections**: `RdBu_r` with `TwoSlopeNorm(vmin, vcenter=1.0, vmax)`
- **Offset corrections**: `RdBu_r` with `TwoSlopeNorm(vmin, vcenter=0.0, vmax)`
- Mark identity values with a black dashed line on colorbars

### Figure sizing

| Type | Size (cm) |
|------|-----------|
| Map panels (2-up) | 17 x 9 |
| Single map | 10 x 10 |
| Bar charts (2 panels) | 14 x 6 |
| 2x4 grid | ncols*5 x 10 |
| Box plots | 17 x 7 |

---

## 3. Checklist for Updating a Script

- [ ] Import `from vwf.viz.style import thesis_plot_style` and palettes from `vwf.viz.palettes`
- [ ] Call `style = thesis_plot_style()` at script start
- [ ] Convert figure sizes to cm: `figsize=(x * cm, y * cm)`
- [ ] Save with `dpi=style['dpi']`
- [ ] Remove all hardcoded `fontsize=` parameters
- [ ] Use `TwoSlopeNorm` for diverging colormaps
- [ ] Use `RdBu_r` for scalar/offset maps
- [ ] Verify output looks correct

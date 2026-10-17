# Third-party notices

## flybody (TuragaLab / Google DeepMind / HHMI Janelia)

3dfly vendors a subset of the [flybody](https://github.com/TuragaLab/flybody) fruit-fly
MuJoCo body (meshes + topology) and applies a documented scale into the 10 m room.

- **Upstream**: https://github.com/TuragaLab/flybody
- **Menagerie copy**: https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody
- **License**: Apache License 2.0 (full text: `assets/fly_meshes/FLYBODY_LICENSE` and `THIRD_PARTY_FLYBODY_LICENSE`)
- **Vendored here**: `assets/fly_meshes/` (thorax, head, abdomen, wings, T2 legs, halteres)
- **Not vendored**: the remaining ~80 high-resolution decorative meshes (~130 MB). Optional fetch:

  ```bash
  python scripts/fetch_flybody.py
  ```

  which sparse-clones menagerie `flybody/` into `assets/flybody/full/` (gitignored).

### Scaling (see `threedfly/sim/scale.py`)

| Quantity | flybody (CGS, life-size) | 3dfly |
| --- | --- | --- |
| Length | ~2.5 mm adult | OBJ millimetres × **0.01** → metres (**2.5 cm**, 10× life-size) |
| Mass | ~1 mg | ~**1.5 g** with flybody's thorax-heavy ratios (~1000×, for SI numerics in a 10 m room) |
| Gravity | 981 cm/s² | 9.81 m/s² |
| Wings | 200 Hz blade-element | stroke-averaged quasi-steady lift + visual flap |
| Legs | articulated walk | **retracted flight pose**, welded |

### Citation

```bibtex
@article{flybody,
  title = {Whole-body physics simulation of fruit fly locomotion},
  author = {Roman Vaxenburg and Igor Siwanowicz and Josh Merel and Alice A Robie and
            Carmen Morrow and Guido Novati and Zinovia Stefanidi and Gert-Jan Both and
            Gwyneth M Card and Michael B Reiser and Matthew M Botvinick and
            Kristin M Branson and Yuval Tassa and Srinivas C Turaga},
  journal = {Nature},
  volume = {643},
  pages = {1312--1320},
  year = {2025},
  doi = {https://doi.org/10.1038/s41586-025-09029-4},
}
```

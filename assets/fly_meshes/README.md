# Fruit Fly 3D Meshes

## Source

These 3D mesh files are derived from the **flybody** project, an anatomically-detailed body model of the fruit fly *Drosophila melanogaster* for MuJoCo physics simulator.

- **Original Repository**: https://github.com/TuragaLab/flybody
- **MuJoCo Menagerie**: https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody
- **Developed by**: Google DeepMind and HHMI Janelia Research Campus

## License

These meshes are released under the **Apache License 2.0** (see `FLYBODY_LICENSE` file).

## Citation

If you use these meshes, please cite the flybody project:

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
  url = {https://www.nature.com/articles/s41586-025-09029-4},
}
```

## Files

- `thorax_body.obj` - Main body (thorax) mesh
- `head_body.obj` - Head mesh  
- `abdomen_1_body.obj` - Abdomen segment mesh
- `wing_left_membrane.obj` - Left wing membrane
- `wing_right_membrane.obj` - Right wing membrane
- `femur_T2_left_body.obj` - Left middle leg femur
- `femur_T2_right_body.obj` - Right middle leg femur
- `tibia_T2_left_body.obj` - Left middle leg tibia
- `tibia_T2_right_body.obj` - Right middle leg tibia

## Usage

These meshes are scaled by 0.01 in the MuJoCo model to match the simulation scale. The original meshes are in millimeters and are scaled down to work with the meter-based physics simulation.

The visual appearance uses these realistic meshes while the physics simulation uses simplified collision geometry for computational efficiency.

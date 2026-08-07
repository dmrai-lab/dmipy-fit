CONSTANTS = dict(
    water_diffusion_constant=3.05e-9,   # m^2/s — free water at 37°C (in vivo)
    water_diffusion_constant_25C=2.299e-9,  # m^2/s — free water at 25°C (lab)
    water_in_axons_diffusion_constant=1.7e-9,  # m^2/s
    naa_in_axons=.00015e-9,  # m^2 / s
    water_gyromagnetic_ratio=267.513e6,   # 1/(sT)
)

# Parameter-scaling factors — the single source of truth for the optimizer's parameter
# normalization (each fitted parameter is carried internally in units of its scale so the
# solver sees O(1) magnitudes). Import these; do NOT redefine per module.
DIFFUSIVITY_SCALING = 1e-9   # m^2/s   (diffusivities carried in units of 1e-9)
DIAMETER_SCALING = 1e-6      # m       (diameters carried in micrometres)
A_SCALING = 1e-12            # m^2     (cross-sectional areas)
BETA_SCALING = 1e-6          # m       (Gamma diameter-distribution scale, 1 um)

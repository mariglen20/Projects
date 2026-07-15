# Boundary conditions for velocity field
def apply_bc_velocity(u, v):
    # No-slip on the borders
    for field in [u, v]:        # field is pointing at both u and v (same BCs)
        field[0, :] = 0.0       # lower border
        field[:, 0] = 0.0       # left border
        field[:, -1] = 0.0      # right border
        field[-1, :] = 0.0      # upper border (needed for v)

    # Lid
    u[-1, :] = 1.0

    return u, v

# # Boundary conditions for pressure field (needed for projection method)
# def apply_bc_pressure(p):
#     # dp_dn = 0
#     p[0, :] = p[1, :]       # lower border
#     p[:, 0] = p[:, 1]       # left border
#     p[-1, :] = p[-2, :]     # upper border
#     p[:, -1] = p[:, -2]     # right border

#     return p
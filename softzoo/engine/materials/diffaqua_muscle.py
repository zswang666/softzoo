from typing import Any
import numpy as np
import taichi as ti

from .. import I_DTYPE, F_DTYPE
from . import default


def get_material_model(solver: Any):
    dim = solver.dim
    dx = solver.dx
    inv_dx = solver.inv_dx
    p_vol = solver.p_vol

    # Get muscle direction for all particle group (only used in SimpleMuscle material)
    use_actuator_to_specify_muscle_direction = getattr(solver, 'use_actuator_to_specify_muscle_direction', False)
    f_dtype_np = np.float64 if F_DTYPE == ti.f64 else np.float32
    if use_actuator_to_specify_muscle_direction:
        n_actuators = solver.n_actuators
        solver.muscle_direction = ti.Matrix.field(solver.dim, solver.dim, dtype=F_DTYPE, shape=(n_actuators,))
        for i in range(n_actuators):
            AAt = solver.base_muscle_direction_matrix
            solver.muscle_direction[i] = AAt.astype(f_dtype_np).tolist()
    else:
        max_particle_group_id = max(solver.particle_group_info.keys()) + 1
        solver.muscle_direction = ti.Matrix.field(solver.dim, solver.dim, dtype=F_DTYPE)
        ti.root.dynamic(ti.i, max(2, max_particle_group_id)).place(solver.muscle_direction) # cannot instantiate dynamic S-node with shape (1,)
        for k, v in solver.particle_group_info.items():
            if v.muscle_direction is None:
                AAt = np.zeros((solver.dim, solver.dim))
            else:
                AAt = np.array(v.muscle_direction).reshape(-1, 1) @ np.array(v.muscle_direction).reshape(1, -1)
            solver.muscle_direction[k] = AAt.astype(f_dtype_np).tolist()
    
    compute_F_tmp, p2g = default.get_material_model(solver)

    @ti.func
    def get_muscle_direction_matrix(p, id):
        return solver.muscle_direction[id]

    if not hasattr(solver, 'get_muscle_direction_matrix'):
        solver.get_muscle_direction_matrix = get_muscle_direction_matrix

    @ti.func
    def p2g_corotated(
        p: I_DTYPE,
        s: I_DTYPE,
        next_s: I_DTYPE,
        dt: F_DTYPE,
        p_mass: F_DTYPE,
        x: ti.template(),
        v: ti.template(),
        F: ti.template(),
        C: ti.template(),
        Jp: ti.template(),
        F_tmp: ti.template(),
        U: ti.template(),
        sig: ti.template(),
        V: ti.template(),
        actuation: ti.template(),
        mu_: ti.template(),
        lambd_: ti.template(),
        particle_ids: ti.template(),
        grid_v_in: ti.template(),
        grid_m: ti.template(),
    ) -> ti.template():
        # Get Lame parameters
        h = ti.cast(0.3, F_DTYPE) # hardening coefficient
        mu = mu_[p] * h
        lambd = lambd_[p] * h

        # Compute determinant
        J = ti.cast(1., F_DTYPE)
        new_sig = ti.Matrix.zero(F_DTYPE, dim, dim)
        for d in ti.static(range(dim)):
            new_sig[d, d] = sig[s, p][d, d]
            J *= new_sig[d, d]

        Jp[next_s, p] = J

        # Compute deformation gradient
        new_F = F_tmp[s, p]
        F[next_s, p] = new_F

        # Compute stress
        # looks like corotational material:
        # P = 2 \mu (F - U V^t) F^T + \lambda I J (J - 1)
        stress = 2 * mu * (new_F - U[s, p] @ V[s, p].transpose()) @ new_F.transpose() + \
                ti.Matrix.identity(F_DTYPE, dim) * lambd * J * (J - 1)

        id = particle_ids[p]
        muscle_stiffness = mu / ti.cast(solver.base_muscle_mu, F_DTYPE) * ti.cast(solver.base_muscle_stiffness, F_DTYPE)
        AAt = solver.get_muscle_direction_matrix(p, id)
        stress += muscle_stiffness * actuation[s, p] * new_F @ AAt @ new_F.transpose()

        stress = (-dt * p_vol * 4 * inv_dx**2) * stress

        return stress

    @ti.func
    def p2g_neohooken(
        p: I_DTYPE,
        s: I_DTYPE,
        next_s: I_DTYPE,
        dt: F_DTYPE,
        p_mass: F_DTYPE,
        x: ti.template(),
        v: ti.template(),
        F: ti.template(),
        C: ti.template(),
        Jp: ti.template(),
        F_tmp: ti.template(),
        U: ti.template(),
        sig: ti.template(),
        V: ti.template(),
        actuation: ti.template(),
        mu_: ti.template(),
        lambd_: ti.template(),
        particle_ids: ti.template(),
        grid_v_in: ti.template(),
        grid_m: ti.template(),
    ) -> ti.template():
        # Get Lame parameters
        h = ti.cast(0.3, F_DTYPE) # hardening coefficient
        mu = mu_[p] * h
        lambd = lambd_[p] * h

        # Compute determinant
        J = ti.cast(1., F_DTYPE)
        new_sig = ti.Matrix.zero(F_DTYPE, dim, dim)
        for d in ti.static(range(dim)):
            new_sig[d, d] = sig[s, p][d, d]
            J *= new_sig[d, d]

        Jp[next_s, p] = J

        # Compute deformation gradient
        new_F = F_tmp[s, p]
        F[next_s, p] = new_F

        # Compute stress
        stress = mu * (new_F @ new_F.transpose()) + ti.Matrix.identity(F_DTYPE, dim) * (lambd * ti.log(J) - mu)

        id = particle_ids[p]
        muscle_stiffness = mu / ti.cast(solver.base_muscle_mu, F_DTYPE) * ti.cast(solver.base_muscle_stiffness, F_DTYPE)
        AAt = solver.get_muscle_direction_matrix(p, id)
        stress += muscle_stiffness * actuation[s, p] * new_F @ AAt @ new_F.transpose()

        stress = (-dt * p_vol * 4 * inv_dx**2) * stress

        return stress

    p2g = p2g_neohooken

    return compute_F_tmp, p2g

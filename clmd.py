import os
import warnings

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.algorithm import RadixSort
from pyopencl.reduction import ReductionKernel

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
warnings.filterwarnings("ignore", category=cl.CompilerWarning)

__doc__ = r'''A simple MD example in opencl.
'''


class MDCodeGenerator:
    def __init__(self):
        self.injections = {
            "USER_DEFINED_FUNCTIONS": "",
            "INJECT_PAIR_FORCE": "",
            "INJECT_SINGLE_PARTICLE_FORCE": ""
        }

        # =========================================================================
        # 1. UTILITIES (Hash, PBC, RNG)
        # =========================================================================
        self.utils_code = """
        {USER_DEFINED_FUNCTIONS}

        inline unsigned int expandBits(unsigned int v) {
            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;
            return v;
        }

        inline unsigned int get_morton_hash(int cx, int cy, int cz) {
            cx = clamp(cx, 0, 1023); cy = clamp(cy, 0, 1023); cz = clamp(cz, 0, 1023);
            return (expandBits(cx)) | (expandBits(cy) << 1) | (expandBits(cz) << 2);
        }

        inline float apply_pbc(float dx, float box) { return dx - box * rint(dx / box); }

        inline uint hash32(uint state) {
            state ^= state >> 16; state *= 0x85ebca6bu;
            state ^= state >> 13; state *= 0xc2b2ae35u;
            state ^= state >> 16; return state;
        }

        inline float rand_normal(uint seed, int id, int step) {
            uint state = seed + (uint)id * 1999999973u + (uint)step * 283970195u;
            uint s1 = hash32(state); uint s2 = hash32(s1);
            float u1 = clamp((float)s1 / 4294967296.0f, 1e-6f, 0.999999f);
            float u2 = clamp((float)s2 / 4294967296.0f, 1e-6f, 0.999999f);
            return sqrt(-2.0f * log(u1)) * cos(2.0f * 3.14159265359f * u2);
        }

        __kernel void calc_cell_hash(
            __global const float* restrict x, __global const float* restrict y, __global const float* restrict z,
            __global int* restrict cell_hash, __global int* restrict indices,
            const int grid_dim_x, const int grid_dim_y, const int grid_dim_z,
            const float box_x, const float box_y, const float box_z,
            const int n_atoms
        ) {
            int i = get_global_id(0);
            if (i >= n_atoms) return;
            int cx = clamp((int)(x[i] / box_x * grid_dim_x), 0, grid_dim_x - 1);
            int cy = clamp((int)(y[i] / box_y * grid_dim_y), 0, grid_dim_y - 1);
            int cz = clamp((int)(z[i] / box_z * grid_dim_z), 0, grid_dim_z - 1);
            cell_hash[i] = get_morton_hash(cx, cy, cz);
            indices[i] = i;
        }

        __kernel void build_reverse_map(
            __global const int* restrict original_id,
            __global int* restrict reverse_map, const int n_atoms
        ) {
            int i = get_global_id(0);
            if (i >= n_atoms) return;
            reverse_map[original_id[i]] = i; 
        }
        """

        # =========================================================================
        # 2. NEIGHBOR LIST
        # =========================================================================
        self.nlist_code = """
        __kernel void build_cell_boundaries(
            __global const int* restrict cell_id, __global int* restrict cell_start,
            __global int* restrict cell_end, const int n_atoms
        ) {
            int i = get_global_id(0);
            if (i >= n_atoms) return;
            int my_cell = cell_id[i];
            if (i == 0) { cell_start[my_cell] = 0; }
            else {
                int prev_cell = cell_id[i - 1];
                if (my_cell != prev_cell) { cell_start[my_cell] = i; cell_end[prev_cell] = i; }
            }
            if (i == n_atoms - 1) { cell_end[my_cell] = n_atoms; }
        }

        __kernel void build_verlet_list(
            __global const float* restrict x, __global const float* restrict y, __global const float* restrict z,
            __global float* restrict x_ref, __global float* restrict y_ref, __global float* restrict z_ref,
            __global const int* restrict cell_start, __global const int* restrict cell_end,
            __global int* restrict nlist, __global int* restrict nlist_counts,
            const int grid_dim_x, const int grid_dim_y, const int grid_dim_z,
            const float cell_size, const float r_list_sq, const int max_neighbors, const int n_atoms,
            const float box_x, const float box_y, const float box_z
        ) {
            int i = get_global_id(0);
            if (i >= n_atoms) return;
            float xi = x[i], yi = y[i], zi = z[i];
            x_ref[i] = xi; y_ref[i] = yi; z_ref[i] = zi;
            int cx = clamp((int)(x[i] / box_x * grid_dim_x), 0, grid_dim_x - 1);
            int cy = clamp((int)(y[i] / box_y * grid_dim_y), 0, grid_dim_y - 1);
            int cz = clamp((int)(z[i] / box_z * grid_dim_z), 0, grid_dim_z - 1);
            int count = 0, base_idx = i * max_neighbors;

            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = (cx + dx % grid_dim_x + grid_dim_x) % grid_dim_x;
                        int ny = (cy + dy % grid_dim_y + grid_dim_y) % grid_dim_y;
                        int nz = (cz + dz % grid_dim_z + grid_dim_z) % grid_dim_z;
                        int n_cell_id = get_morton_hash(nx, ny, nz);
                        int start = cell_start[n_cell_id], end = cell_end[n_cell_id];
                        for (int j = start; j < end; ++j) {
                            if (i == j) continue;
                            float dx_vec = apply_pbc(xi - x[j], box_x);
                            float dy_vec = apply_pbc(yi - y[j], box_y);
                            float dz_vec = apply_pbc(zi - z[j], box_z);
                            if (dx_vec*dx_vec + dy_vec*dy_vec + dz_vec*dz_vec < r_list_sq) {
                                if (count < max_neighbors) nlist[base_idx + count] = j; 
                                count++;
                            }
                        }
                    }
                }
            }
            nlist_counts[i] = count;
        }
        """

        # =========================================================================
        # 3. INTERACTION ENGINES (Non-Bonded & Topologies)
        # =========================================================================
        self.non_bond_code = """
        inline void calc_lj_coul_core(float r_sq, float sig_i, float sig_j, float eps_i, float eps_j, float q_i,
            float q_j, float coulomb_const, float s_vdw, float s_coul, float* f_mag_vdw, float* f_mag_coul,
            float* u_vdw, float* u_coul
        ) {
            if (s_vdw > 0.001f) {
                float r2inv = 1.0f / r_sq; float sig_mix = sqrt(sig_i * sig_j); float eps_mix = sqrt(eps_i * eps_j);
                float r6inv = (sig_mix*sig_mix*r2inv) * (sig_mix*sig_mix*r2inv) * (sig_mix*sig_mix*r2inv);
                *f_mag_vdw = (48.0f * eps_mix * r6inv * r6inv - 24.0f * eps_mix * r6inv) * r2inv * s_vdw;
                *u_vdw = 4.0f * eps_mix * (r6inv * r6inv - r6inv) * s_vdw;
            }
            if (s_coul > 0.001f) {
                float r2inv = 1.0f / r_sq; float r_inv = sqrt(r2inv);
                *f_mag_coul = coulomb_const * (q_i * q_j) * r_inv * r2inv * s_coul;
                *u_coul = coulomb_const * (q_i * q_j) * r_inv * s_coul;
            }
            *f_mag_vdw = clamp(*f_mag_vdw, -20000.0f, 20000.0f); *f_mag_coul = clamp(*f_mag_coul, -20000.0f, 20000.0f);
        }

        inline void compute_non_bonded(
            int orig_i, __global const int* restrict original_id, __global const int* restrict excl_count,
            __global const int* restrict excl_partners, 
            __global const float* restrict excl_scale_vdw, __global const float* restrict excl_scale_coul, const int max_excl, 
            int i, int gi, int count, int max_neighbors, __global const int* nlist,
            float xi, float yi, float zi, float sig_i, float eps_i, float q_i, float d_i,
            __global const float* restrict x_in, __global const float* restrict y_in, __global const float* restrict z_in,
            __global const float* restrict sigma, __global const float* restrict epsilon, __global const float* restrict charge,
            __global const float* restrict diameter, __global const int* restrict group_id, 
            float box_x, float box_y, float box_z, float r_cut_sq, float coulomb_const,
            float* fx_vdw, float* fy_vdw, float* fz_vdw, float* fx_coul, float* fy_coul, float* fz_coul, 
            float* u_vdw, float* u_coul, float* vir_vdw, float* vir_coul
        ) {
            int n_excl = excl_count[orig_i];
            for (int k = 0; k < count; ++k) {
                int j = nlist[i * max_neighbors + k];
                int orig_j = original_id[j]; int gj = group_id[j];

                float dx = apply_pbc(xi - x_in[j], box_x); float dy = apply_pbc(yi - y_in[j], box_y); float dz = apply_pbc(zi - z_in[j], box_z);
                float r_sq = dx*dx + dy*dy + dz*dz;

                if (r_sq < r_cut_sq && r_sq > 0.01f) {
                    float f_vdw = 0.0f, f_coul = 0.0f, u_v = 0.0f, u_c = 0.0f, s_vdw = 1.0f, s_coul = 1.0f;
                    float sig_j = sigma[j], eps_j = epsilon[j], q_j = charge[j], d_j = diameter[j];

                    for (int e = 0; e < n_excl; ++e) {
                        int idx = orig_i * max_excl + e;
                        if (excl_partners[idx] == orig_j) { s_vdw = excl_scale_vdw[idx]; s_coul = excl_scale_coul[idx]; break; }
                    }

                    calc_lj_coul_core(r_sq, sig_i, sig_j, eps_i, eps_j, q_i, q_j, coulomb_const, s_vdw, s_coul, &f_vdw, &f_coul, &u_v, &u_c);

                    float custom_f_mag = 0.0f; float custom_u = 0.0f;
                    {INJECT_PAIR_FORCE}
                    f_vdw += custom_f_mag; u_v += custom_u;

                    // Write separated forces
                    *fx_vdw += f_vdw * dx; *fy_vdw += f_vdw * dy; *fz_vdw += f_vdw * dz;
                    *fx_coul += f_coul * dx; *fy_coul += f_coul * dy; *fz_coul += f_coul * dz;
                    *u_vdw += 0.5f * u_v; *u_coul += 0.5f * u_c;

                    vir_vdw[0] -= 0.5f*(f_vdw*dx)*dx; vir_vdw[1] -= 0.5f*(f_vdw*dy)*dy; vir_vdw[2] -= 0.5f*(f_vdw*dz)*dz;
                    vir_vdw[3] -= 0.5f*(f_vdw*dx)*dy; vir_vdw[4] -= 0.5f*(f_vdw*dx)*dz; vir_vdw[5] -= 0.5f*(f_vdw*dy)*dz;
                    vir_coul[0] -= 0.5f*(f_coul*dx)*dx; vir_coul[1] -= 0.5f*(f_coul*dy)*dy; vir_coul[2] -= 0.5f*(f_coul*dz)*dz;
                    vir_coul[3] -= 0.5f*(f_coul*dx)*dy; vir_coul[4] -= 0.5f*(f_coul*dx)*dz; vir_coul[5] -= 0.5f*(f_coul*dy)*dz;
                }
            }
        }
        """

        self.bond_code = """
        inline void compute_bonds(
            int orig_i, __global const int* restrict reverse_map, __global const int* restrict bond_count, __global const int* restrict bond_partners, 
            __global const float* restrict bond_k, __global const float* restrict bond_r0, const int max_bonds, float xi, float yi, float zi,
            __global const float* restrict x_in, __global const float* restrict y_in, __global const float* restrict z_in,
            float box_x, float box_y, float box_z, float* fx, float* fy, float* fz, float* u_bond, float* vir_bond
        ) {
            int count = bond_count[orig_i];
            for (int b = 0; b < count; ++b) {
                int idx = orig_i * max_bonds + b; int j = reverse_map[bond_partners[idx]];
                float dx = apply_pbc(xi - x_in[j], box_x); float dy = apply_pbc(yi - y_in[j], box_y); float dz = apply_pbc(zi - z_in[j], box_z);
                float r = sqrt(dx*dx + dy*dy + dz*dz);
                if (r > 0.0001f) {
                    float k_b = bond_k[idx]; float r0 = bond_r0[idx]; float dr = r - r0;
                    float f_mag = -2.0f * k_b * dr / r; 
                    *fx += f_mag * dx; *fy += f_mag * dy; *fz += f_mag * dz; *u_bond += 0.5f * (k_b * dr * dr); 
                    vir_bond[0] -= 0.5f*(f_mag*dx)*dx; vir_bond[1] -= 0.5f*(f_mag*dy)*dy; vir_bond[2] -= 0.5f*(f_mag*dz)*dz;
                    vir_bond[3] -= 0.5f*(f_mag*dx)*dy; vir_bond[4] -= 0.5f*(f_mag*dx)*dz; vir_bond[5] -= 0.5f*(f_mag*dy)*dz;
                }
            }
        }
        """

        self.angle_code = """
        inline void compute_angles(
            int orig_i, __global const int* restrict reverse_map, __global const int* restrict angle_count,
            __global const int* restrict angle_A, 
            __global const int* restrict angle_B, __global const int* restrict angle_C, __global const float* restrict angle_k,
            __global const float* restrict angle_theta0, const int max_angles,
            __global const float* restrict x_in, __global const float* restrict y_in, __global const float* restrict z_in,
            float box_x, float box_y, float box_z, float* fx, float* fy, float* fz, float* u_angle, float* vir_angle
        ) {
            int count = angle_count[orig_i];
            for (int a = 0; a < count; ++a) {
                int idx = orig_i * max_angles + a; int orig_A = angle_A[idx], orig_B = angle_B[idx], orig_C = angle_C[idx];
                int id_A = reverse_map[orig_A], id_B = reverse_map[orig_B], id_C = reverse_map[orig_C];
                float rBA_x = apply_pbc(x_in[id_A] - x_in[id_B], box_x);
                float rBA_y = apply_pbc(y_in[id_A] - y_in[id_B], box_y);
                float rBA_z = apply_pbc(z_in[id_A] - z_in[id_B], box_z);
                float rBC_x = apply_pbc(x_in[id_C] - x_in[id_B], box_x);
                float rBC_y = apply_pbc(y_in[id_C] - y_in[id_B], box_y);
                float rBC_z = apply_pbc(z_in[id_C] - z_in[id_B], box_z);
                float dBA = sqrt(rBA_x*rBA_x + rBA_y*rBA_y + rBA_z*rBA_z);
                float dBC = sqrt(rBC_x*rBC_x + rBC_y*rBC_y + rBC_z*rBC_z);

                if (dBA > 0.001f && dBC > 0.001f) {
                    float cos_theta = clamp((rBA_x*rBC_x + rBA_y*rBC_y + rBA_z*rBC_z) / (dBA * dBC), -0.9999f, 0.9999f);
                    float theta = acos(cos_theta);
                    float sin_theta = max(sqrt(1.0f - cos_theta*cos_theta), 1e-6f);
                    float dU_dtheta = 2.0f * angle_k[idx] * (theta - angle_theta0[idx]); 

                    float termA = -dU_dtheta / (dBA * sin_theta); float termC = -dU_dtheta / (dBC * sin_theta);
                    float fA_x = termA * (rBC_x/dBC - cos_theta*rBA_x/dBA);
                    float fA_y = termA * (rBC_y/dBC - cos_theta*rBA_y/dBA);
                    float fA_z = termA * (rBC_z/dBC - cos_theta*rBA_z/dBA);
                    float fC_x = termC * (rBA_x/dBA - cos_theta*rBC_x/dBC);
                    float fC_y = termC * (rBA_y/dBA - cos_theta*rBC_y/dBC);
                    float fC_z = termC * (rBA_z/dBA - cos_theta*rBC_z/dBC);
                    float fB_x = -(fA_x + fC_x); float fB_y = -(fA_y + fC_y); float fB_z = -(fA_z + fC_z);

                    if (orig_i == orig_A) { *fx += fA_x; *fy += fA_y; *fz += fA_z; }
                    else if (orig_i == orig_C) { *fx += fC_x; *fy += fC_y; *fz += fC_z; }
                    else if (orig_i == orig_B) { 
                        *fx += fB_x; *fy += fB_y; *fz += fB_z;
                        *u_angle += angle_k[idx] * (theta - angle_theta0[idx])*(theta - angle_theta0[idx]); 
                        vir_angle[0] -= (fA_x*rBA_x + fC_x*rBC_x); vir_angle[1] -= (fA_y*rBA_y + fC_y*rBC_y);
                        vir_angle[2] -= (fA_z*rBA_z + fC_z*rBC_z); vir_angle[3] -= (fA_x*rBA_y + fC_x*rBC_y);
                        vir_angle[4] -= (fA_x*rBA_z + fC_x*rBC_z); vir_angle[5] -= (fA_y*rBA_z + fC_y*rBC_z);
                    }
                }
            }
        }
        """

        self.torsion_code = """
        inline void compute_torsions(
            int orig_i, __global const int* restrict reverse_map, __global const int* restrict dihe_count, 
            __global const int* restrict dihe_A, __global const int* restrict dihe_B, __global const int* restrict dihe_C,
            __global const int* restrict dihe_D, __global const float* restrict dihe_k1, __global const float* restrict dihe_k2,
            __global const float* restrict dihe_k3, __global const float* restrict dihe_k4, const int max_dihes,
            __global const float* restrict x_in, __global const float* restrict y_in, __global const float* restrict z_in,
            float box_x, float box_y, float box_z, float* fx, float* fy, float* fz, float* u_dihe, float* vir_dihe
        )
        {
            int count = dihe_count[orig_i];
            for (int a = 0; a < count; ++a) {
                int idx = orig_i * max_dihes + a;
                int oA = dihe_A[idx], oB = dihe_B[idx], oC = dihe_C[idx], oD = dihe_D[idx];
                int id_A = reverse_map[oA], id_B = reverse_map[oB], id_C = reverse_map[oC], id_D = reverse_map[oD];

                float rABx = apply_pbc(x_in[id_B] - x_in[id_A], box_x);
                float rABy = apply_pbc(y_in[id_B] - y_in[id_A], box_y);
                float rABz = apply_pbc(z_in[id_B] - z_in[id_A], box_z);
                float rBCx = apply_pbc(x_in[id_C] - x_in[id_B], box_x);
                float rBCy = apply_pbc(y_in[id_C] - y_in[id_B], box_y);
                float rBCz = apply_pbc(z_in[id_C] - z_in[id_B], box_z);
                float rCDx = apply_pbc(x_in[id_D] - x_in[id_C], box_x);
                float rCDy = apply_pbc(y_in[id_D] - y_in[id_C], box_y);
                float rCDz = apply_pbc(z_in[id_D] - z_in[id_C], box_z);

                float mx = rABy*rBCz - rABz*rBCy; float my = rABz*rBCx - rABx*rBCz; float mz = rABx*rBCy - rABy*rBCx;
                float nx = rBCy*rCDz - rBCz*rCDy; float ny = rBCz*rCDx - rBCx*rCDz; float nz = rBCx*rCDy - rBCy*rCDx;
                float m_sq = max(mx*mx + my*my + mz*mz, 1e-8f); float n_sq = max(nx*nx + ny*ny + nz*nz, 1e-8f);
                float bc_norm = max(sqrt(rBCx*rBCx + rBCy*rBCy + rBCz*rBCz), 1e-6f);

                float cos_phi = clamp((mx*nx + my*ny + mz*nz) / sqrt(m_sq * n_sq), -0.9999f, 0.9999f);
                float sin_phi = ((mx*rCDx + my*rCDy + mz*rCDz)*bc_norm) / sqrt(m_sq * n_sq); float phi = atan2(sin_phi, cos_phi);

                float k1 = dihe_k1[idx], k2 = dihe_k2[idx], k3 = dihe_k3[idx], k4 = dihe_k4[idx];
                float energy = 0.5f*k1*(1.0f+cos(phi)) + 0.5f*k2*(1.0f-cos(2.0f*phi)) + 0.5f*k3*(1.0f+cos(3.0f*phi)) + 0.5f*k4*(1.0f-cos(4.0f*phi));
                float dU_dphi = -0.5f*k1*sin(phi) + k2*sin(2.0f*phi) - 1.5f*k3*sin(3.0f*phi) + 2.0f*k4*sin(4.0f*phi);

                float fA_scale = dU_dphi * bc_norm / m_sq; float fD_scale = -dU_dphi * bc_norm / n_sq;
                float fAx = fA_scale * mx; float fAy = fA_scale * my; float fAz = fA_scale * mz;
                float fDx = fD_scale * nx; float fDy = fD_scale * ny; float fDz = fD_scale * nz;

                float B_scaleA = (rABx*rBCx + rABy*rBCy + rABz*rBCz) / (bc_norm*bc_norm);
                float B_scaleD = (rBCx*rCDx + rBCy*rCDy + rBCz*rCDz) / (bc_norm*bc_norm);
                float fBx = fAx*(B_scaleA - 1.0f) - fDx*B_scaleD; float fBy = fAy*(B_scaleA - 1.0f) - fDy*B_scaleD;
                float fBz = fAz*(B_scaleA - 1.0f) - fDz*B_scaleD;
                float fCx = fDx*(B_scaleD - 1.0f) - fAx*B_scaleA; float fCy = fDy*(B_scaleD - 1.0f) - fAy*B_scaleA;
                float fCz = fDz*(B_scaleD - 1.0f) - fAz*B_scaleA;

                if (orig_i == oA) { *fx += fAx; *fy += fAy; *fz += fAz; }
                else if (orig_i == oB) { *fx += fBx; *fy += fBy; *fz += fBz; }
                else if (orig_i == oD) { *fx += fDx; *fy += fDy; *fz += fDz; }
                else if (orig_i == oC) { 
                    *fx += fCx; *fy += fCy; *fz += fCz; *u_dihe += energy; 
                    float rCAx = -rABx-rBCx, rCAy = -rABy-rBCy, rCAz = -rABz-rBCz;
                    float rCBx = -rBCx, rCBy = -rBCy, rCBz = -rBCz;
                    vir_dihe[0] -= (fAx*rCAx + fBx*rCBx + fDx*rCDx); vir_dihe[1] -= (fAy*rCAy + fBy*rCBy + fDy*rCDy);
                    vir_dihe[2] -= (fAz*rCAz + fBz*rCBz + fDz*rCDz);
                    vir_dihe[3] -= (fAx*rCAy + fBx*rCBy + fDx*rCDy); vir_dihe[4] -= (fAx*rCAz + fBx*rCBz + fDx*rCDz);
                    vir_dihe[5] -= (fAy*rCAz + fBy*rCBz + fDy*rCDz);
                }
            }
        }
        """

        self.improper_code = """
        inline void compute_impropers(
            int orig_i, __global const int* restrict reverse_map, __global const int* restrict impr_count, 
            __global const int* restrict impr_A, __global const int* restrict impr_B,
            __global const int* restrict impr_C, __global const int* restrict impr_D,
            __global const float* restrict impr_k, __global const float* restrict impr_theta0, const int max_imprs,
            __global const float* restrict x_in, __global const float* restrict y_in, __global const float* restrict z_in,
            float box_x, float box_y, float box_z, float* fx, float* fy, float* fz, float* u_impr, float* vir_impr
        ) {
            int count = impr_count[orig_i];
            for (int a = 0; a < count; ++a) {
                int idx = orig_i * max_imprs + a; int oA = impr_A[idx], oB = impr_B[idx], oC = impr_C[idx], oD = impr_D[idx];
                int id_A = reverse_map[oA], id_B = reverse_map[oB], id_C = reverse_map[oC], id_D = reverse_map[oD];

                float rBAx = apply_pbc(x_in[id_A] - x_in[id_B], box_x);
                float rBAy = apply_pbc(y_in[id_A] - y_in[id_B], box_y);
                float rBAz = apply_pbc(z_in[id_A] - z_in[id_B], box_z);
                float rBCx = apply_pbc(x_in[id_C] - x_in[id_B], box_x);
                float rBCy = apply_pbc(y_in[id_C] - y_in[id_B], box_y);
                float rBCz = apply_pbc(z_in[id_C] - z_in[id_B], box_z);
                float rBDx = apply_pbc(x_in[id_D] - x_in[id_B], box_x);
                float rBDy = apply_pbc(y_in[id_D] - y_in[id_B], box_y);
                float rBDz = apply_pbc(z_in[id_D] - z_in[id_B], box_z);

                float mx = rBAy*rBCz - rBAz*rBCy; float my = rBAz*rBCx - rBAx*rBCz; float mz = rBAx*rBCy - rBAy*rBCx;
                float nx = rBCy*rBDz - rBCz*rBDy; float ny = rBCz*rBDx - rBCx*rBDz; float nz = rBCx*rBDy - rBCy*rBDx;
                float m_sq = max(mx*mx + my*my + mz*mz, 1e-8f); float n_sq = max(nx*nx + ny*ny + nz*nz, 1e-8f);
                float bc_norm = max(sqrt(rBCx*rBCx + rBCy*rBCy + rBCz*rBCz), 1e-6f);

                float cos_phi = clamp((mx*nx + my*ny + mz*nz) / sqrt(m_sq * n_sq), -0.9999f, 0.9999f);
                float sin_phi = ((mx*rBDx + my*rBDy + mz*rBDz)*bc_norm) / sqrt(m_sq * n_sq);float phi = atan2(sin_phi, cos_phi);
                float dU_dphi = 2.0f * impr_k[idx] * (phi - impr_theta0[idx]);

                float fA_scale = dU_dphi * bc_norm / m_sq; float fD_scale = -dU_dphi * bc_norm / n_sq;
                float fAx = fA_scale * mx; float fAy = fA_scale * my; float fAz = fA_scale * mz;
                float fDx = fD_scale * nx; float fDy = fD_scale * ny; float fDz = fD_scale * nz;

                float B_scaleA = (rBAx*rBCx + rBAy*rBCy + rBAz*rBCz) / (bc_norm*bc_norm);
                float B_scaleD = (rBCx*rBDx + rBCy*rBDy + rBCz*rBDz) / (bc_norm*bc_norm);
                float fCx = fAx*(B_scaleA - 1.0f) - fDx*B_scaleD;
                float fCy = fAy*(B_scaleA - 1.0f) - fDy*B_scaleD;
                float fCz = fAz*(B_scaleA - 1.0f) - fDz*B_scaleD;
                float fBx = -(fAx + fCx + fDx); float fBy = -(fAy + fCy + fDy); float fBz = -(fAz + fCz + fDz);

                if (orig_i == oA) { *fx += fAx; *fy += fAy; *fz += fAz; }
                else if (orig_i == oC) { *fx += fCx; *fy += fCy; *fz += fCz; }
                else if (orig_i == oD) { *fx += fDx; *fy += fDy; *fz += fDz; }
                else if (orig_i == oB) { 
                    *fx += fBx; *fy += fBy; *fz += fBz; *u_impr += impr_k[idx] * (phi - impr_theta0[idx])*(phi - impr_theta0[idx]); 
                    vir_impr[0] -= (fAx*rBAx + fCx*rBCx + fDx*rBDx); vir_impr[1] -= (fAy*rBAy + fCy*rBCy + fDy*rBDy);
                    vir_impr[2] -= (fAz*rBAz + fCz*rBCz + fDz*rBDz); vir_impr[3] -= (fAx*rBAy + fCx*rBCy + fDx*rBDy);
                    vir_impr[4] -= (fAx*rBAz + fCx*rBCz + fDx*rBDz); vir_impr[5] -= (fAy*rBAz + fCy*rBCz + fDy*rBDz);
                }
            }
        }
        """

        # =========================================================================
        # 4. THERMODYNAMICS & MAIN FORCE KERNEL
        # =========================================================================
        self.thermo_code = """
        inline void calc_particle_thermo(
            float* vir_vdw, float* vir_coul, float* vir_bond, float* vir_angle, float* vir_dihe, float* vir_impr, float* vir_all
        ) {
            for (int d = 0; d < 6; ++d) {
                vir_all[d] = vir_vdw[d] + vir_coul[d] + vir_bond[d] + vir_angle[d] + vir_dihe[d] + vir_impr[d];
            }
        }
        """

        self.force_code = """
        __kernel void compute_forces(
            // 1. coord input
            __global const float* restrict x_in, __global const float* restrict y_in,
            __global const float* restrict z_in,
            // 2. output forces for integrator
            __global float* restrict fx_out, __global float* restrict fy_out, __global float* restrict fz_out,
            // 3. trigger of nlist
            __global const float* restrict x_ref, __global const float* restrict y_ref,
            __global const float* restrict z_ref,
            __global int* restrict nlist_trigger, const float skin_sq,
            // 4. identities, sigma, charge...
            __global const int* restrict original_id, __global const int* restrict reverse_map,
            __global const float* restrict diameter, __global const float* restrict sigma,
            __global const float* restrict epsilon, 
            __global const float* restrict charge, __global const int* restrict group_id,
            // 5. nlist, bonded table
            __global const int* restrict nlist, __global const int* restrict nlist_counts,
            __global const int* restrict excl_count, __global const int* restrict excl_partners,
            __global const float* restrict excl_scale_vdw, __global const float* restrict excl_scale_coul,
            __global const int* restrict bond_count, __global const int* restrict bond_partners,
            __global const float* restrict bond_k, __global const float* restrict bond_r0,
            __global const int* restrict angle_count, __global const int* restrict angle_A,
            __global const int* restrict angle_B, __global const int* restrict angle_C,
            __global const float* restrict angle_k, __global const float* restrict angle_theta0,
            __global const int* restrict dihe_count, __global const int* restrict dihe_A,
            __global const int* restrict dihe_B, __global const int* restrict dihe_C,
            __global const int* restrict dihe_D, __global const float* restrict dihe_k1,
            __global const float* restrict dihe_k2, __global const float* restrict dihe_k3,
            __global const float* restrict dihe_k4,
            __global const int* restrict impr_count, __global const int* restrict impr_A,
            __global const int* restrict impr_B, __global const int* restrict impr_C,
            __global const int* restrict impr_D, __global const float* restrict impr_k,
            __global const float* restrict impr_theta0,
            // 6. energies and virials
            __global float* restrict u_tot, __global float* restrict u_vdw_arr, __global float* restrict u_coul_arr,
            __global float* restrict u_bond_arr, __global float* restrict u_angle_arr, __global float* restrict u_dihe_arr,
            __global float* restrict u_impr_arr,
            __global float* restrict fx_vdw_arr, __global float* restrict fy_vdw_arr, __global float* restrict fz_vdw_arr,
            __global float* restrict fx_coul_arr, __global float* restrict fy_coul_arr, __global float* restrict fz_coul_arr,
            __global float* restrict fx_bond_arr, __global float* restrict fy_bond_arr, __global float* restrict fz_bond_arr,
            __global float* restrict fx_angle_arr, __global float* restrict fy_angle_arr, __global float* restrict fz_angle_arr,
            __global float* restrict fx_dihe_arr, __global float* restrict fy_dihe_arr, __global float* restrict fz_dihe_arr,
            __global float* restrict fx_impr_arr, __global float* restrict fy_impr_arr, __global float* restrict fz_impr_arr,
            __global float* restrict vir_all_arr, __global float* restrict vir_vdw_arr, __global float* restrict vir_coul_arr,
            __global float* restrict vir_bond_arr, __global float* restrict vir_angle_arr, __global float* restrict vir_dihe_arr,
            __global float* restrict vir_impr_arr,
            // 7. control params
            const int max_neighbors, const int max_excl, const int max_bonds, const int max_angles, const int max_dihes,
            const int max_imprs, const float box_x, const float box_y, const float box_z, const float r_cut_sq,
            const float coulomb_const
        )
        {
            int i = get_global_id(0);
            if (i >= get_global_size(0)) return;

            float xi = x_in[i], yi = y_in[i], zi = z_in[i];
            int orig_i = original_id[i]; int gi = group_id[i]; float d_i = diameter[i];

            // Local accumulators for forces
            float fx_vdw = 0.0f, fy_vdw = 0.0f, fz_vdw = 0.0f;
            float fx_coul = 0.0f, fy_coul = 0.0f, fz_coul = 0.0f;
            float fx_bond = 0.0f, fy_bond = 0.0f, fz_bond = 0.0f;
            float fx_angle = 0.0f, fy_angle = 0.0f, fz_angle = 0.0f;
            float fx_dihe = 0.0f, fy_dihe = 0.0f, fz_dihe = 0.0f;
            float fx_impr = 0.0f, fy_impr = 0.0f, fz_impr = 0.0f;

            // Local accumulators for energies and virials
            float e_vdw=0.0f, e_coul=0.0f, e_bond=0.0f, e_angle=0.0f, e_dihe=0.0f, e_impr=0.0f;
            float v_vdw[6]={0}, v_coul[6]={0}, v_bond[6]={0}, v_angle[6]={0}, v_dihe[6]={0}, v_impr[6]={0};

            // Calculate specific interactions
            compute_non_bonded(
                orig_i, original_id, excl_count, excl_partners, excl_scale_vdw, excl_scale_coul, max_excl,
                i, gi, nlist_counts[i], max_neighbors, nlist, xi, yi, zi, sigma[i], epsilon[i], charge[i],
                d_i, x_in, y_in, z_in, sigma, epsilon, charge, diameter, group_id, box_x, box_y, box_z,
                r_cut_sq, coulomb_const, &fx_vdw, &fy_vdw, &fz_vdw, &fx_coul, &fy_coul, &fz_coul, &e_vdw,
                &e_coul, v_vdw, v_coul
            );
            compute_bonds(
                orig_i, reverse_map, bond_count, bond_partners, bond_k, bond_r0, max_bonds,
                xi, yi, zi, x_in, y_in, z_in, box_x, box_y, box_z, &fx_bond, &fy_bond, &fz_bond, &e_bond, v_bond
            );
            compute_angles(
                orig_i, reverse_map, angle_count, angle_A, angle_B, angle_C, angle_k, angle_theta0,
                max_angles, x_in, y_in, z_in, box_x, box_y, box_z, &fx_angle, &fy_angle, &fz_angle, &e_angle, v_angle
            );
            compute_torsions(
                orig_i, reverse_map, dihe_count, dihe_A, dihe_B, dihe_C, dihe_D, dihe_k1, dihe_k2, dihe_k3, dihe_k4,
                max_dihes, x_in, y_in, z_in, box_x, box_y, box_z, &fx_dihe, &fy_dihe, &fz_dihe, &e_dihe, v_dihe
            );
            compute_impropers(
                orig_i, reverse_map, impr_count, impr_A, impr_B, impr_C, impr_D, impr_k, impr_theta0, max_imprs,
                x_in, y_in, z_in, box_x, box_y, box_z, &fx_impr, &fy_impr, &fz_impr, &e_impr, v_impr
            );

            // INJECT_SINGLE, use orig_i, gi, xi, yi, zi, box_x, box_y, box_z
            // modify fx,y,z_custom
            float fx_custom = 0.0f, fy_custom = 0.0f, fz_custom = 0.0f;
            
            {INJECT_SINGLE_PARTICLE_FORCE}

            // Accumulate total force for integrator
            float fxi = fx_vdw + fx_coul + fx_bond + fx_angle + fx_dihe + fx_impr + fx_custom;
            float fyi = fy_vdw + fy_coul + fy_bond + fy_angle + fy_dihe + fy_impr + fy_custom;
            float fzi = fz_vdw + fz_coul + fz_bond + fz_angle + fz_dihe + fz_impr + fz_custom;

            // Write outputs to global memory arrays
            fx_out[i] = fxi; fy_out[i] = fyi; fz_out[i] = fzi;
            fx_vdw_arr[i] = fx_vdw; fy_vdw_arr[i] = fy_vdw; fz_vdw_arr[i] = fz_vdw;
            fx_coul_arr[i] = fx_coul; fy_coul_arr[i] = fy_coul; fz_coul_arr[i] = fz_coul;
            fx_bond_arr[i] = fx_bond; fy_bond_arr[i] = fy_bond; fz_bond_arr[i] = fz_bond;
            fx_angle_arr[i] = fx_angle; fy_angle_arr[i] = fy_angle; fz_angle_arr[i] = fz_angle;
            fx_dihe_arr[i] = fx_dihe; fy_dihe_arr[i] = fy_dihe; fz_dihe_arr[i] = fz_dihe;
            fx_impr_arr[i] = fx_impr; fy_impr_arr[i] = fy_impr; fz_impr_arr[i] = fz_impr;

            u_tot[i] = e_vdw + e_coul + e_bond + e_angle + e_dihe + e_impr;
            u_vdw_arr[i] = e_vdw; u_coul_arr[i] = e_coul; u_bond_arr[i] = e_bond;
            u_angle_arr[i] = e_angle; u_dihe_arr[i] = e_dihe; u_impr_arr[i] = e_impr;

            float vir_all[6] = {0};
            calc_particle_thermo(v_vdw, v_coul, v_bond, v_angle, v_dihe, v_impr, vir_all);

            for(int d=0; d<6; ++d) {
                vir_all_arr[i*6+d] = vir_all[d];
                vir_vdw_arr[i*6+d] = v_vdw[d]; vir_coul_arr[i*6+d] = v_coul[d]; vir_bond_arr[i*6+d] = v_bond[d];
                vir_angle_arr[i*6+d] = v_angle[d]; vir_dihe_arr[i*6+d] = v_dihe[d]; vir_impr_arr[i*6+d] = v_impr[d];
            }

            // Trigger Check
            float dx_ref = apply_pbc(xi - x_ref[i], box_x);
            float dy_ref = apply_pbc(yi - y_ref[i], box_y);
            float dz_ref = apply_pbc(zi - z_ref[i], box_z);
            if (dx_ref*dx_ref + dy_ref*dy_ref + dz_ref*dz_ref >= skin_sq) {
                atomic_xchg((volatile __global int*)nlist_trigger, 1);
            }
        }
        """

        # =========================================================================
        # 5. INTEGRATORS
        # =========================================================================
        self.integrator_code = """
        // 1. NVE (Velocity Verlet)
        __kernel void nve_step1(
            __global float* x, __global float* y, __global float* z,
            __global float* vx, __global float* vy, __global float* vz,
            __global const float* fx, __global const float* fy, __global const float* fz,
            __global const float* mass, const float dt,
            const float box_x, const float box_y, const float box_z
        )
        {
            int i = get_global_id(0); float m_i = mass[i];
            vx[i] += (fx[i] / m_i) * 0.5f * dt;
            vy[i] += (fy[i] / m_i) * 0.5f * dt;
            vz[i] += (fz[i] / m_i) * 0.5f * dt;
            x[i] += vx[i] * dt; y[i] += vy[i] * dt; z[i] += vz[i] * dt;
            x[i] -= box_x * floor(x[i] / box_x);
            y[i] -= box_y * floor(y[i] / box_y);
            z[i] -= box_z * floor(z[i] / box_z);
        }
        __kernel void nve_step2(
            __global float* vx, __global float* vy, __global float* vz,
            __global const float* fx, __global const float* fy, __global const float* fz,
            __global const float* mass, const float dt
        )
        {
            int i = get_global_id(0); float m_i = mass[i];
            vx[i] += (fx[i] / m_i) * 0.5f * dt;
            vy[i] += (fy[i] / m_i) * 0.5f * dt;
            vz[i] += (fz[i] / m_i) * 0.5f * dt;
        }

        // 2. NVT (Nose-Hoover)
        __kernel void nh_step1(
            __global float* x, __global float* y, __global float* z,
            __global float* vx, __global float* vy, __global float* vz,
            __global const float* fx, __global const float* fy, __global const float* fz,
            __global const float* mass, const float dt, const float zeta,
            const float box_x, const float box_y, const float box_z
        )
        {
            int i = get_global_id(0); float m_i = mass[i];
            vx[i] += (fx[i] / m_i - zeta * vx[i]) * 0.5f * dt;
            vy[i] += (fy[i] / m_i - zeta * vy[i]) * 0.5f * dt;
            vz[i] += (fz[i] / m_i - zeta * vz[i]) * 0.5f * dt;
            x[i] += vx[i] * dt; y[i] += vy[i] * dt; z[i] += vz[i] * dt;
            x[i] -= box_x * floor(x[i] / box_x);
            y[i] -= box_y * floor(y[i] / box_y);
            z[i] -= box_z * floor(z[i] / box_z);
        }
        __kernel void nh_step2(__global float* vx, __global float* vy, __global float* vz,
            __global const float* fx, __global const float* fy, __global const float* fz,
            __global const float* mass, const float dt, const float zeta
        ) {
            int i = get_global_id(0); float m_i = mass[i];
            vx[i] += (fx[i] / m_i - zeta * vx[i]) * 0.5f * dt;
            vy[i] += (fy[i] / m_i - zeta * vy[i]) * 0.5f * dt;
            vz[i] += (fz[i] / m_i - zeta * vz[i]) * 0.5f * dt;
        }

        // 3. Langevin (VV Formulation)
        __kernel void langevin_step1(
            __global float* x, __global float* y, __global float* z,
            __global float* vx, __global float* vy, __global float* vz,
            __global const float* fx, __global const float* fy, __global const float* fz,
            __global const float* mass, const float dt, const float gamma, const float temp,
            const uint seed, const int step,
            const float box_x, const float box_y, const float box_z
        )
        {
            int i = get_global_id(0); float m_i = mass[i];
            float noise_std = sqrt(2.0f * gamma * temp * m_i / dt);
            float rand_x = noise_std * rand_normal(seed, i, step * 2 + 0);
            float rand_y = noise_std * rand_normal(seed, i, step * 2 + 1);
            float rand_z = noise_std * rand_normal(seed, i, step * 2 + 2);

            vx[i] += ((fx[i] + rand_x) / m_i - gamma * vx[i]) * 0.5f * dt;
            vy[i] += ((fy[i] + rand_y) / m_i - gamma * vy[i]) * 0.5f * dt;
            vz[i] += ((fz[i] + rand_z) / m_i - gamma * vz[i]) * 0.5f * dt;

            x[i] += vx[i] * dt; y[i] += vy[i] * dt; z[i] += vz[i] * dt;
            x[i] -= box_x * floor(x[i] / box_x);
            y[i] -= box_y * floor(y[i] / box_y);
            z[i] -= box_z * floor(z[i] / box_z);
        }
        __kernel void langevin_step2(
            __global float* vx, __global float* vy, __global float* vz,
            __global const float* fx, __global const float* fy, __global const float* fz,
            __global const float* mass, const float dt, const float gamma, const float temp,
            const uint seed, const int step
        )
        {
            int i = get_global_id(0); float m_i = mass[i];
            float noise_std = sqrt(2.0f * gamma * temp * m_i / dt);
            float rand_x = noise_std * rand_normal(seed, i, step * 2 + 10);
            float rand_y = noise_std * rand_normal(seed, i, step * 2 + 11);
            float rand_z = noise_std * rand_normal(seed, i, step * 2 + 12);

            vx[i] += ((fx[i] + rand_x) / m_i - gamma * vx[i]) * 0.5f * dt;
            vy[i] += ((fy[i] + rand_y) / m_i - gamma * vy[i]) * 0.5f * dt;
            vz[i] += ((fz[i] + rand_z) / m_i - gamma * vz[i]) * 0.5f * dt;
        }
        """

    def generate(self):
        code = self.utils_code + self.nlist_code + self.non_bond_code + \
               self.bond_code + self.angle_code + self.torsion_code + \
               self.improper_code + self.thermo_code + self.force_code + self.integrator_code

        for tag, val in self.injections.items():
            code = code.replace(f"{{{tag}}}", val)
        return code

    def inject(self, tag, code):
        if tag in self.injections:
            self.injections[tag] += code + "\n"


class MDEngine:
    def __init__(self, n_atoms, box_size, r_cut=2.5, skin=0.5, dt=0.002, safety_factor=1.2, temperature=120):
        r"""
        :param n_atoms: Number of atoms
        :param box_size: list of box dimensions
        :param r_cut: cut radius
        :param skin: cut buffer
        :param dt: integration time interval
        :param safety_factor: to estimate max_neighbor: \rho * V_cut * safety_factor
        :param temperature: 120K ~ 0.24 kcal/mol
        """
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        self.codegen = MDCodeGenerator()

        self.n_atoms = np.int32(n_atoms)
        self.dt = np.float32(dt)
        self.box_x = np.float32(box_size[0])
        self.box_y = np.float32(box_size[1])
        self.box_z = np.float32(box_size[2])

        self.r_cut = np.float32(r_cut)
        self.r_cut_sq = np.float32(r_cut * r_cut)
        self.skin = np.float32(skin)
        self.skin_sq = np.float32((skin / 2.0) ** 2)
        self.r_list = np.float32(r_cut + skin)
        self.r_list_sq = np.float32(self.r_list * self.r_list)

        # integrator settings
        self.gamma = np.float32(1)
        self.target_temp = np.float32(temperature)
        self.boltzmann = np.float32(0.0019872)  # in system of kcal/mol, A, fs
        self.internal_temp = self.target_temp * self.boltzmann
        self.seed = np.uint32(np.random.randint(0, 1000000))
        self.coulomb_const = np.float32(332.0637)
        self.zeta = np.float32(1.0)  # Nose-Hoover zeta0
        self.tau = np.float32(0.5)  # Nose-Hoover time

        volume = box_size[0] * box_size[1] * box_size[2]
        density = n_atoms / volume
        search_vol = (4.0 / 3.0) * np.pi * (self.r_list ** 3)
        self.max_neighbors = np.int32(max(100, int(np.ceil(density * search_vol * safety_factor))))

        self.max_excl = np.int32(32)
        self.max_bonds = np.int32(6)
        self.max_angles = np.int32(12)
        self.max_dihes = np.int32(24)
        self.max_imprs = np.int32(12)

        self.cell_size = np.float32(self.r_list)
        self.grid_dim = np.floor(np.array(box_size) / self.cell_size).astype(np.int32)
        max_dim = int(np.max(self.grid_dim))
        self.max_morton_id = np.int32((1 << (max_dim - 1).bit_length() if max_dim > 0 else 1) ** 3)

        self.radix_sort = RadixSort(self.ctx, arguments="int *keys, int *values", key_expr="keys[i]",
                                    sort_arg_names=("keys", "values"))

        self._allocate_buffers()
        self._num_nl_update = 0

    def _allocate_buffers(self):
        # =====================================================================
        # 1. coord and identities (Ping-Pong Buffer)
        # =====================================================================
        self.d_x = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_y = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_z = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_vx = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_vy = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_vz = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)

        self.d_mass = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_diameter = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_sigma = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_epsilon = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_charge = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_group_id = cl_array.zeros(self.queue, self.n_atoms, dtype=np.int32)
        self.d_original_id = cl_array.arange(self.queue, self.n_atoms, dtype=np.int32)

        # Ping-Pong Alts
        self.d_x_alt = cl_array.empty_like(self.d_x)
        self.d_y_alt = cl_array.empty_like(self.d_y)
        self.d_z_alt = cl_array.empty_like(self.d_z)
        self.d_vx_alt = cl_array.empty_like(self.d_vx)
        self.d_vy_alt = cl_array.empty_like(self.d_vy)
        self.d_vz_alt = cl_array.empty_like(self.d_vz)
        self.d_mass_alt = cl_array.empty_like(self.d_mass)
        self.d_diameter_alt = cl_array.empty_like(self.d_diameter)
        self.d_sigma_alt = cl_array.empty_like(self.d_sigma)
        self.d_epsilon_alt = cl_array.empty_like(self.d_epsilon)
        self.d_charge_alt = cl_array.empty_like(self.d_charge)
        self.d_group_id_alt = cl_array.empty_like(self.d_group_id)
        self.d_original_id_alt = cl_array.empty_like(self.d_original_id)

        # =====================================================================
        # 2. thermo quantities, write in-place, no need for permutation
        # =====================================================================
        # forces and energy
        self.d_fx = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fy = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fz = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_u_tot = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)

        # Per-particle separated energies
        self.d_u_vdw = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_u_coul = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_u_bond = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_u_angle = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_u_dihe = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_u_impr = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)

        # Per-particle separated forces
        self.d_fx_vdw = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fy_vdw = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fz_vdw = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fx_coul = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fy_coul = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fz_coul = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fx_bond = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fy_bond = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fz_bond = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fx_angle = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fy_angle = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fz_angle = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fx_dihe = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fy_dihe = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fz_dihe = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fx_impr = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fy_impr = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_fz_impr = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)

        # Per-particle separated virials, 6-components N*6 arrays
        self.d_vir_all = cl_array.zeros(self.queue, self.n_atoms * 6, dtype=np.float32)
        self.d_vir_vdw = cl_array.zeros(self.queue, self.n_atoms * 6, dtype=np.float32)
        self.d_vir_coul = cl_array.zeros(self.queue, self.n_atoms * 6, dtype=np.float32)
        self.d_vir_bond = cl_array.zeros(self.queue, self.n_atoms * 6, dtype=np.float32)
        self.d_vir_angle = cl_array.zeros(self.queue, self.n_atoms * 6, dtype=np.float32)
        self.d_vir_dihe = cl_array.zeros(self.queue, self.n_atoms * 6, dtype=np.float32)
        self.d_vir_impr = cl_array.zeros(self.queue, self.n_atoms * 6, dtype=np.float32)

        # =====================================================================
        # 3. nlist and mappings
        # =====================================================================
        self.d_hash = cl_array.zeros(self.queue, self.n_atoms, dtype=np.int32)
        self.d_indices = cl_array.zeros(self.queue, self.n_atoms, dtype=np.int32)
        self.d_reverse_map = cl_array.zeros(self.queue, self.n_atoms, dtype=np.int32)
        self.d_x_ref = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_y_ref = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_z_ref = cl_array.zeros(self.queue, self.n_atoms, dtype=np.float32)
        self.d_nlist_trigger = cl_array.zeros(self.queue, 1, dtype=np.int32)
        self.d_cell_start = cl_array.zeros(self.queue, self.max_morton_id, dtype=np.int32)
        self.d_cell_end = cl_array.zeros(self.queue, self.max_morton_id, dtype=np.int32)
        self.d_nlist = cl_array.zeros(self.queue, self.n_atoms * self.max_neighbors, dtype=np.int32)
        self.d_counts = cl_array.zeros(self.queue, self.n_atoms, dtype=np.int32)

        # =====================================================================
        # 4. topologies
        # =====================================================================
        self.d_excl_count = cl_array.zeros(self.queue, self.n_atoms, dtype=np.int32)
        self.d_excl_partners = cl_array.zeros(self.queue, self.n_atoms * self.max_excl, dtype=np.int32)
        self.d_excl_scale_vdw = cl_array.zeros(self.queue, self.n_atoms * self.max_excl, dtype=np.float32)
        self.d_excl_scale_coul = cl_array.zeros(self.queue, self.n_atoms * self.max_excl, dtype=np.float32)
        self.d_bond_count = cl_array.zeros(self.queue, self.n_atoms, dtype=np.int32)
        self.d_bond_partners = cl_array.zeros(self.queue, self.n_atoms * self.max_bonds, dtype=np.int32)
        self.d_bond_k = cl_array.zeros(self.queue, self.n_atoms * self.max_bonds, dtype=np.float32)
        self.d_bond_r0 = cl_array.zeros(self.queue, self.n_atoms * self.max_bonds, dtype=np.float32)
        self.d_angle_count = cl_array.zeros(self.queue, self.n_atoms, dtype=np.int32)
        self.d_angle_A = cl_array.zeros(self.queue, self.n_atoms * self.max_angles, dtype=np.int32)
        self.d_angle_B = cl_array.zeros(self.queue, self.n_atoms * self.max_angles, dtype=np.int32)
        self.d_angle_C = cl_array.zeros(self.queue, self.n_atoms * self.max_angles, dtype=np.int32)
        self.d_angle_k = cl_array.zeros(self.queue, self.n_atoms * self.max_angles, dtype=np.float32)
        self.d_angle_t0 = cl_array.zeros(self.queue, self.n_atoms * self.max_angles, dtype=np.float32)
        self.d_dihe_count = cl_array.zeros(self.queue, self.n_atoms, dtype=np.int32)
        self.d_dihe_A = cl_array.zeros(self.queue, self.n_atoms * self.max_dihes, dtype=np.int32)
        self.d_dihe_B = cl_array.zeros(self.queue, self.n_atoms * self.max_dihes, dtype=np.int32)
        self.d_dihe_C = cl_array.zeros(self.queue, self.n_atoms * self.max_dihes, dtype=np.int32)
        self.d_dihe_D = cl_array.zeros(self.queue, self.n_atoms * self.max_dihes, dtype=np.int32)
        self.d_dihe_k1 = cl_array.zeros(self.queue, self.n_atoms * self.max_dihes, dtype=np.float32)
        self.d_dihe_k2 = cl_array.zeros(self.queue, self.n_atoms * self.max_dihes, dtype=np.float32)
        self.d_dihe_k3 = cl_array.zeros(self.queue, self.n_atoms * self.max_dihes, dtype=np.float32)
        self.d_dihe_k4 = cl_array.zeros(self.queue, self.n_atoms * self.max_dihes, dtype=np.float32)
        self.d_impr_count = cl_array.zeros(self.queue, self.n_atoms, dtype=np.int32)
        self.d_impr_A = cl_array.zeros(self.queue, self.n_atoms * self.max_imprs, dtype=np.int32)
        self.d_impr_B = cl_array.zeros(self.queue, self.n_atoms * self.max_imprs, dtype=np.int32)
        self.d_impr_C = cl_array.zeros(self.queue, self.n_atoms * self.max_imprs, dtype=np.int32)
        self.d_impr_D = cl_array.zeros(self.queue, self.n_atoms * self.max_imprs, dtype=np.int32)
        self.d_impr_k = cl_array.zeros(self.queue, self.n_atoms * self.max_imprs, dtype=np.float32)
        self.d_impr_t0 = cl_array.zeros(self.queue, self.n_atoms * self.max_imprs, dtype=np.float32)

    def load_initial_aos(self, pos, vel, mass, diameter, sigma, epsilon, charge, group_id):
        self.d_x.set(pos[:, 0].astype(np.float32))
        self.d_y.set(pos[:, 1].astype(np.float32))
        self.d_z.set(pos[:, 2].astype(np.float32))
        self.d_vx.set(vel[:, 0].astype(np.float32))
        self.d_vy.set(vel[:, 1].astype(np.float32))
        self.d_vz.set(vel[:, 2].astype(np.float32))
        self.d_mass.set(mass.astype(np.float32))
        self.d_diameter.set(diameter.astype(np.float32))
        self.d_sigma.set(sigma.astype(np.float32))
        self.d_epsilon.set(epsilon.astype(np.float32))
        self.d_charge.set(charge.astype(np.float32))
        self.d_group_id.set(group_id.astype(np.int32))

    def load_exclusions(self, excl_list):
        counts = np.zeros(self.n_atoms, dtype=np.int32)
        partners = np.zeros(self.n_atoms * self.max_excl, dtype=np.int32)
        sv = np.zeros(self.n_atoms * self.max_excl, dtype=np.float32)
        sc = np.zeros(self.n_atoms * self.max_excl, dtype=np.float32)

        for i, j, v, c in excl_list:
            # J IN I'S EXCL
            if counts[i] < self.max_excl:
                idx = i * self.max_excl + counts[i]
                partners[idx] = j
                sv[idx] = v
                sc[idx] = c
                counts[i] += 1
            # I IN J'S EXCL
            if counts[j] < self.max_excl:
                idx = j * self.max_excl + counts[j]
                partners[idx] = i
                sv[idx] = v
                sc[idx] = c
                counts[j] += 1

        self.d_excl_count.set(counts)
        self.d_excl_partners.set(partners)
        self.d_excl_scale_vdw.set(sv)
        self.d_excl_scale_coul.set(sc)

    def load_bonds(self, bond_list):
        counts, partners, ks, r0s = np.zeros(self.n_atoms, dtype=np.int32), np.zeros(self.n_atoms * self.max_bonds,
                                                                                     dtype=np.int32), np.zeros(
            self.n_atoms * self.max_bonds, dtype=np.float32), np.zeros(self.n_atoms * self.max_bonds, dtype=np.float32)
        for i, j, k, r0 in bond_list:
            if counts[i] < self.max_bonds:
                idx = i * self.max_bonds + counts[i]
                partners[idx] = j
                ks[idx] = k
                r0s[idx] = r0
                counts[i] += 1
            if counts[j] < self.max_bonds:
                idx = j * self.max_bonds + counts[j]
                partners[idx] = i
                ks[idx] = k
                r0s[idx] = r0
                counts[j] += 1
        self.d_bond_count.set(counts)
        self.d_bond_partners.set(partners)
        self.d_bond_k.set(ks)
        self.d_bond_r0.set(r0s)

    def load_angles(self, angle_list):
        counts, A, B, C, ks, t0s = np.zeros(self.n_atoms, dtype=np.int32), np.zeros(self.n_atoms * self.max_angles,
                                                                                    dtype=np.int32), np.zeros(
            self.n_atoms * self.max_angles, dtype=np.int32), np.zeros(self.n_atoms * self.max_angles,
                                                                      dtype=np.int32), np.zeros(
            self.n_atoms * self.max_angles, dtype=np.float32), np.zeros(self.n_atoms * self.max_angles,
                                                                        dtype=np.float32)
        for a, b, c, k, t0 in angle_list:
            for node in (a, b, c):
                if counts[node] < self.max_angles:
                    idx = node * self.max_angles + counts[node]
                    A[idx] = a
                    B[idx] = b
                    C[idx] = c
                    ks[idx] = k
                    t0s[idx] = t0
                    counts[node] += 1
        self.d_angle_count.set(counts)
        self.d_angle_A.set(A)
        self.d_angle_B.set(B)
        self.d_angle_C.set(C)
        self.d_angle_k.set(ks)
        self.d_angle_t0.set(t0s)

    def load_dihedrals(self, dihe_list):
        counts, A, B, C, D, k1, k2, k3, k4 = (
            np.zeros(self.n_atoms, dtype=np.int32),
            np.zeros(self.n_atoms * self.max_dihes, dtype=np.int32),
            np.zeros(self.n_atoms * self.max_dihes, dtype=np.int32),
            np.zeros(self.n_atoms * self.max_dihes, dtype=np.int32),
            np.zeros(self.n_atoms * self.max_dihes, dtype=np.int32),
            np.zeros(self.n_atoms * self.max_dihes, dtype=np.float32),
            np.zeros(self.n_atoms * self.max_dihes, dtype=np.float32),
            np.zeros(self.n_atoms * self.max_dihes, dtype=np.float32),
            np.zeros(self.n_atoms * self.max_dihes, dtype=np.float32)
        )
        for a, b, c, d, p1, p2, p3, p4 in dihe_list:
            for node in (a, b, c, d):
                if counts[node] < self.max_dihes:
                    idx = node * self.max_dihes + counts[node]
                    A[idx] = a
                    B[idx] = b
                    C[idx] = c
                    D[idx] = d
                    k1[idx] = p1
                    k2[idx] = p2
                    k3[idx] = p3
                    k4[idx] = p4
                    counts[node] += 1
        self.d_dihe_count.set(counts)
        self.d_dihe_A.set(A)
        self.d_dihe_B.set(B)
        self.d_dihe_C.set(C)
        self.d_dihe_D.set(D)
        self.d_dihe_k1.set(k1)
        self.d_dihe_k2.set(k2)
        self.d_dihe_k3.set(k3)
        self.d_dihe_k4.set(k4)

    def load_impropers(self, impr_list):
        counts, A, B, C, D, ks, t0s = (
            np.zeros(self.n_atoms, dtype=np.int32), np.zeros(self.n_atoms * self.max_imprs, dtype=np.int32),
            np.zeros(self.n_atoms * self.max_imprs, dtype=np.int32),
            np.zeros(self.n_atoms * self.max_imprs, dtype=np.int32),
            np.zeros(self.n_atoms * self.max_imprs, dtype=np.int32),
            np.zeros(self.n_atoms * self.max_imprs, dtype=np.float32),
            np.zeros(self.n_atoms * self.max_imprs, dtype=np.float32)
        )
        for a, b, c, d, k, t0 in impr_list:
            for node in (a, b, c, d):
                if counts[node] < self.max_imprs:
                    idx = node * self.max_imprs + counts[node]
                    A[idx] = a
                    B[idx] = b
                    C[idx] = c
                    D[idx] = d
                    ks[idx] = k
                    t0s[idx] = t0
                    counts[node] += 1
        self.d_impr_count.set(counts)
        self.d_impr_A.set(A)
        self.d_impr_B.set(B)
        self.d_impr_C.set(C)
        self.d_impr_D.set(D)
        self.d_impr_k.set(ks)
        self.d_impr_t0.set(t0s)

    def compile(self):
        source = self.codegen.generate()
        self.prg = cl.Program(self.ctx, source).build()
        self.knl_calc_cell_hash = self.prg.calc_cell_hash
        self.knl_build_reverse_map = self.prg.build_reverse_map
        self.knl_build_bounds = self.prg.build_cell_boundaries
        self.knl_build_verlet = self.prg.build_verlet_list
        self.knl_compute_forces = self.prg.compute_forces

        # Integrator kernels
        self.knl_nve_step1, self.knl_nve_step2 = self.prg.nve_step1, self.prg.nve_step2
        self.knl_nh_step1, self.knl_nh_step2 = self.prg.nh_step1, self.prg.nh_step2
        self.knl_langevin_step1, self.knl_langevin_step2 = self.prg.langevin_step1, self.prg.langevin_step2

        # Only permute r and idendities
        self.knl_permute = cl.Program(self.ctx, """
        __kernel void permute_data(
            __global const int* restrict sorted_indices,
            __global const float* restrict x_in, __global float* restrict x_out,
            __global const float* restrict y_in, __global float* restrict y_out,
            __global const float* restrict z_in, __global float* restrict z_out,
            __global const float* restrict vx_in, __global float* restrict vx_out,
            __global const float* restrict vy_in, __global float* restrict vy_out,
            __global const float* restrict vz_in, __global float* restrict vz_out,
            __global const float* restrict mass_in, __global float* restrict mass_out,
            __global const float* restrict diameter_in, __global float* restrict diameter_out,
            __global const float* restrict sigma_in, __global float* restrict sigma_out,
            __global const float* restrict epsilon_in, __global float* restrict epsilon_out,
            __global const float* restrict charge_in, __global float* restrict charge_out,
            __global const int* restrict group_in, __global int* restrict group_out,
            __global const int* restrict id_in, __global int* restrict id_out,
            const int n_atoms
        ) {
            int i = get_global_id(0);
            if (i >= n_atoms) return;
            int old = sorted_indices[i];
            x_out[i] = x_in[old]; y_out[i] = y_in[old]; z_out[i] = z_in[old];
            vx_out[i] = vx_in[old]; vy_out[i] = vy_in[old]; vz_out[i] = vz_in[old];
            mass_out[i] = mass_in[old]; diameter_out[i] = diameter_in[old];
            sigma_out[i] = sigma_in[old]; epsilon_out[i] = epsilon_in[old]; charge_out[i] = charge_in[old];
            group_out[i] = group_in[old]; id_out[i] = id_in[old];
        }
        """).build().permute_data

        # ------------------------------------------------------------------
        # Reduction Kernels, avoiding any atomic operations
        # ------------------------------------------------------------------
        self._red_max = ReductionKernel(
            self.ctx, np.int32, neutral="-1", reduce_expr="a > b ? a : b", map_expr="x[i]",
            arguments="__global const int *x"
        )
        self._red_sum = ReductionKernel(
            self.ctx, np.float32, neutral="0", reduce_expr="a+b", map_expr="x[i]",
            arguments="__global const float *x"
        )
        self._red_ke = ReductionKernel(
            self.ctx, np.float32, neutral="0", reduce_expr="a+b",
            map_expr="0.5f * m[i] * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i])",
            arguments="__global const float *m, __global const float *vx,"
                      "__global const float *vy, __global const float *vz"
        )
        self._red_virial = ReductionKernel(
            self.ctx, np.float32, neutral="0", reduce_expr="a+b",
            map_expr="vir[i*6 + offset]",
            arguments="__global const float *dummy, __global const float *vir, int offset"
        )

    def rebuild_neighbor_list_on_gpu(self):
        global_size = (int(self.n_atoms),)
        self._num_nl_update += 1

        self.knl_calc_cell_hash(self.queue, global_size, None, self.d_x.data, self.d_y.data, self.d_z.data,
                                self.d_hash.data, self.d_indices.data, self.grid_dim[0], self.grid_dim[1],
                                self.grid_dim[2], self.box_x, self.box_y, self.box_z, self.n_atoms)
        (sorted_hash, sorted_indices), _ = self.radix_sort(self.d_hash, self.d_indices)

        self.knl_permute(self.queue, global_size, None, sorted_indices.data,
                         self.d_x.data, self.d_x_alt.data, self.d_y.data, self.d_y_alt.data, self.d_z.data,
                         self.d_z_alt.data,
                         self.d_vx.data, self.d_vx_alt.data, self.d_vy.data, self.d_vy_alt.data, self.d_vz.data,
                         self.d_vz_alt.data,
                         self.d_mass.data, self.d_mass_alt.data, self.d_diameter.data, self.d_diameter_alt.data,
                         self.d_sigma.data, self.d_sigma_alt.data, self.d_epsilon.data, self.d_epsilon_alt.data,
                         self.d_charge.data, self.d_charge_alt.data,
                         self.d_group_id.data, self.d_group_id_alt.data, self.d_original_id.data,
                         self.d_original_id_alt.data, self.n_atoms)

        # ping-pong exchage
        self.d_x, self.d_x_alt = self.d_x_alt, self.d_x
        self.d_y, self.d_y_alt = self.d_y_alt, self.d_y
        self.d_z, self.d_z_alt = self.d_z_alt, self.d_z
        self.d_vx, self.d_vx_alt = self.d_vx_alt, self.d_vx
        self.d_vy, self.d_vy_alt = self.d_vy_alt, self.d_vy
        self.d_vz, self.d_vz_alt = self.d_vz_alt, self.d_vz
        self.d_mass, self.d_mass_alt = self.d_mass_alt, self.d_mass
        self.d_diameter, self.d_diameter_alt = self.d_diameter_alt, self.d_diameter
        self.d_sigma, self.d_sigma_alt = self.d_sigma_alt, self.d_sigma
        self.d_epsilon, self.d_epsilon_alt = self.d_epsilon_alt, self.d_epsilon
        self.d_charge, self.d_charge_alt = self.d_charge_alt, self.d_charge
        self.d_group_id, self.d_group_id_alt = self.d_group_id_alt, self.d_group_id
        self.d_original_id, self.d_original_id_alt = self.d_original_id_alt, self.d_original_id

        self.knl_build_reverse_map(self.queue, global_size, None, self.d_original_id.data, self.d_reverse_map.data,
                                   self.n_atoms)

        while True:
            self.d_cell_start.fill(0)
            self.d_cell_end.fill(0)
            self.knl_build_bounds(self.queue, global_size, None, sorted_hash.data, self.d_cell_start.data,
                                  self.d_cell_end.data, self.n_atoms)
            self.knl_build_verlet(
                self.queue, global_size, None, self.d_x.data, self.d_y.data, self.d_z.data,
                self.d_x_ref.data, self.d_y_ref.data, self.d_z_ref.data, self.d_cell_start.data, self.d_cell_end.data,
                self.d_nlist.data, self.d_counts.data, self.grid_dim[0], self.grid_dim[1], self.grid_dim[2],
                self.cell_size, self.r_list_sq, self.max_neighbors, self.n_atoms, self.box_x, self.box_y, self.box_z
            )
            found_max = int(self._red_max(self.d_counts).get())
            if found_max <= self.max_neighbors: break
            print(f"Max num of neighbors: {found_max} exceeds {self.max_neighbors}, "
                  f"expanding nlist to {found_max + 16}")
            self.max_neighbors = np.int32(found_max + 16)
            self.d_nlist = cl_array.zeros(self.queue, self.n_atoms * self.max_neighbors, dtype=np.int32)

        self.d_nlist_trigger.fill(0)

    def _execute_compute_forces(self, global_size):
        # 参数排布非常壮观，严格按照内核逻辑分块，绝无任何冗余错乱！
        self.knl_compute_forces(
            self.queue, global_size, None,
            self.d_x.data, self.d_y.data, self.d_z.data,
            self.d_fx.data, self.d_fy.data, self.d_fz.data,
            self.d_x_ref.data, self.d_y_ref.data, self.d_z_ref.data, self.d_nlist_trigger.data, self.skin_sq,
            self.d_original_id.data, self.d_reverse_map.data,
            self.d_diameter.data, self.d_sigma.data, self.d_epsilon.data, self.d_charge.data, self.d_group_id.data,
            self.d_nlist.data, self.d_counts.data,
            self.d_excl_count.data, self.d_excl_partners.data, self.d_excl_scale_vdw.data, self.d_excl_scale_coul.data,
            self.d_bond_count.data, self.d_bond_partners.data, self.d_bond_k.data, self.d_bond_r0.data,
            self.d_angle_count.data, self.d_angle_A.data, self.d_angle_B.data, self.d_angle_C.data, self.d_angle_k.data,
            self.d_angle_t0.data,
            self.d_dihe_count.data, self.d_dihe_A.data, self.d_dihe_B.data, self.d_dihe_C.data, self.d_dihe_D.data,
            self.d_dihe_k1.data, self.d_dihe_k2.data, self.d_dihe_k3.data, self.d_dihe_k4.data,
            self.d_impr_count.data, self.d_impr_A.data, self.d_impr_B.data, self.d_impr_C.data, self.d_impr_D.data,
            self.d_impr_k.data, self.d_impr_t0.data,
            # Outputs
            self.d_u_tot.data, self.d_u_vdw.data, self.d_u_coul.data, self.d_u_bond.data, self.d_u_angle.data,
            self.d_u_dihe.data, self.d_u_impr.data,
            self.d_fx_vdw.data, self.d_fy_vdw.data, self.d_fz_vdw.data,
            self.d_fx_coul.data, self.d_fy_coul.data, self.d_fz_coul.data,
            self.d_fx_bond.data, self.d_fy_bond.data, self.d_fz_bond.data,
            self.d_fx_angle.data, self.d_fy_angle.data, self.d_fz_angle.data,
            self.d_fx_dihe.data, self.d_fy_dihe.data, self.d_fz_dihe.data,
            self.d_fx_impr.data, self.d_fy_impr.data, self.d_fz_impr.data,
            self.d_vir_all.data, self.d_vir_vdw.data, self.d_vir_coul.data, self.d_vir_bond.data, self.d_vir_angle.data,
            self.d_vir_dihe.data, self.d_vir_impr.data,
            # Constants
            self.max_neighbors, self.max_excl, self.max_bonds, self.max_angles, self.max_dihes, self.max_imprs,
            self.box_x, self.box_y, self.box_z, self.r_cut_sq, self.coulomb_const
        )

    # =========================================================================
    # Extracting per particle data
    # =========================================================================
    def get_per_particle_data(self, property_name="u_vdw"):
        """
        get per particle data: (Array shape: (N,) or (N, 3) or (N, 6))
        available property_name prefixes:
        - Energy: u_tot, u_vdw, u_coul, u_bond, u_angle, u_dihe, u_impr
        - Forces: f_tot, f_vdw, f_coul, f_bond, f_angle, f_dihe, f_impr
        - Virial: vir_all, vir_vdw, vir_coul, vir_bond, vir_angle, vir_dihe, vir_impr
        """
        self.queue.finish()
        restore_indices = np.argsort(self.d_original_id.get())

        if property_name.startswith("u_"):
            # energies are 1-D arrays
            d_arr = getattr(self, f"d_{property_name}", None)
            if d_arr is None: raise ValueError(f"Unknown property {property_name}")
            return d_arr.get()[restore_indices]

        elif property_name.startswith("f_"):
            # assemble fx, fy, fz to (N, 3)
            suffix = property_name.split("_")[1]
            if suffix == "tot":
                x, y, z = self.d_fx.get(), self.d_fy.get(), self.d_fz.get()
            else:
                x = getattr(self, f"d_fx_{suffix}").get()
                y = getattr(self, f"d_fy_{suffix}").get()
                z = getattr(self, f"d_fz_{suffix}").get()
            return np.column_stack((x, y, z))[restore_indices]

        elif property_name.startswith("vir_"):
            # virials are (N, 6)
            d_arr = getattr(self, f"d_{property_name}", None)
            if d_arr is None: raise ValueError(f"Unknown property {property_name}")
            return d_arr.get().reshape((self.n_atoms, 6))[restore_indices]

        else:
            raise ValueError("Supported properties start with u_, f_, or vir_")

    def dump_debug_info(self, num_particles=125):
        self.rebuild_neighbor_list_on_gpu()
        self.queue.finish()
        id_h = self.d_original_id.get()
        restore_indices = np.argsort(id_h)

        # 1. Neighbor List Dump
        nlist_h = self.d_nlist.get()
        counts_h = self.d_counts.get()
        print(f"\n========== Verlet List Mapping ==========")
        xyz = np.empty((num_particles, 3))
        X, Y, Z = self.d_x.get(), self.d_y.get(), self.d_z.get()
        counts = []
        cnlist = []
        for orig_id in range(num_particles):
            gpu_idx = restore_indices[orig_id]
            count = counts_h[gpu_idx]
            raw_neighbors = nlist_h[gpu_idx * self.max_neighbors: gpu_idx * self.max_neighbors + count]
            real_neighbors = [int(id_h[n]) for n in raw_neighbors]
            # print(f"Original Atom {orig_id} (GPU addr: {gpu_idx}) has {count} neighbors -> {real_neighbors}")
            xyz[orig_id] = [X[gpu_idx], Y[gpu_idx], Z[gpu_idx]]
            counts.append(count)
            cnlist.append(real_neighbors)
        return xyz, counts, cnlist

    def compute_thermo(self):
        pe_total = float(self._red_sum(self.d_u_tot).get())
        ke_total = float(self._red_ke(self.d_mass, self.d_vx, self.d_vy, self.d_vz).get())

        virial = np.zeros(6, dtype=np.float32)
        for d in range(6):
            virial[d] = float(self._red_virial(self.d_mass, self.d_vir_all, np.int32(d)).get())

        temp_K = (2.0 * ke_total) / (3.0 * self.n_atoms * self.boltzmann)
        vol = self.box_x * self.box_y * self.box_z
        virial_trace = virial[0] + virial[1] + virial[2]
        pressure_atm = ((2.0 * ke_total - virial_trace) / (3.0 * vol)) * 68568.4

        return {
            "Total_Energy": pe_total + ke_total, "Potential_Energy": pe_total, "Kinetic_Energy": ke_total,
            "Temperature": temp_K, "Pressure_atm": pressure_atm, "Virial_Tensor": virial
        }

    def run(self, steps, ensemble="NVT"):
        global_size = (int(self.n_atoms),)

        # Force at t=0
        self.rebuild_neighbor_list_on_gpu()
        self._execute_compute_forces(global_size)

        for step in range(steps):
            # integrate step 1
            if ensemble == "NVE":
                self.knl_nve_step1(self.queue, global_size, None, self.d_x.data, self.d_y.data, self.d_z.data,
                                   self.d_vx.data, self.d_vy.data, self.d_vz.data, self.d_fx.data, self.d_fy.data,
                                   self.d_fz.data, self.d_mass.data, self.dt, self.box_x, self.box_y, self.box_z)
            elif ensemble == "NVT":
                self.knl_nh_step1(self.queue, global_size, None, self.d_x.data, self.d_y.data, self.d_z.data,
                                  self.d_vx.data, self.d_vy.data, self.d_vz.data, self.d_fx.data, self.d_fy.data,
                                  self.d_fz.data, self.d_mass.data, self.dt, self.zeta, self.box_x, self.box_y,
                                  self.box_z)
            elif ensemble == "LANGEVIN":
                self.knl_langevin_step1(self.queue, global_size, None, self.d_x.data, self.d_y.data, self.d_z.data,
                                        self.d_vx.data, self.d_vy.data, self.d_vz.data, self.d_fx.data, self.d_fy.data,
                                        self.d_fz.data, self.d_mass.data, self.dt, self.gamma, self.internal_temp,
                                        self.seed, np.int32(step), self.box_x, self.box_y, self.box_z)

            # ===== TRIGGER CHECK & NEIGHBOR LIST =====
            if self.d_nlist_trigger.get()[0] > 0:
                self.rebuild_neighbor_list_on_gpu()

            # ===== FORCE EVALUATION =====
            self._execute_compute_forces(global_size)

            # ===== THERMOSTAT UPDATE (Nose-Hoover specific) =====
            if ensemble == "NVT":
                ke_half = float(self._red_ke(self.d_mass, self.d_vx, self.d_vy, self.d_vz).get())
                t_curr = (2.0 * ke_half) / (3.0 * self.n_atoms * self.boltzmann)
                zeta_dot = np.float32((t_curr - self.target_temp) / ((self.tau ** 2) * self.target_temp))
                self.zeta += zeta_dot * self.dt

            # ===== integration step 2 =====
            if ensemble == "NVE":
                self.knl_nve_step2(self.queue, global_size, None, self.d_vx.data, self.d_vy.data, self.d_vz.data,
                                   self.d_fx.data, self.d_fy.data, self.d_fz.data, self.d_mass.data, self.dt)
            elif ensemble == "NVT":
                self.knl_nh_step2(self.queue, global_size, None, self.d_vx.data, self.d_vy.data, self.d_vz.data,
                                  self.d_fx.data, self.d_fy.data, self.d_fz.data, self.d_mass.data, self.dt, self.zeta)
            elif ensemble == "LANGEVIN":
                self.knl_langevin_step2(self.queue, global_size, None, self.d_vx.data, self.d_vy.data, self.d_vz.data,
                                        self.d_fx.data, self.d_fy.data, self.d_fz.data, self.d_mass.data, self.dt,
                                        self.gamma, self.internal_temp, self.seed, np.int32(step))

            if step % 20000 == 0:
                thermo = self.compute_thermo()
                print(
                    f"[{ensemble}] Step {step}: T={thermo['Temperature']:.2f}K, "
                    f"P={thermo['Pressure_atm']:.2f}atm, E_pot={thermo['Potential_Energy']:.4f}, "
                    f"E_tot={thermo['Total_Energy']:.4f}"
                )

        self.queue.finish()


def generate_cubic_lattice(size=5, spacing=1.0):
    axis = np.arange(0, size * spacing, spacing)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')

    coords = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    return coords


if __name__ == "__main__":
    import time

    BOX_SIZE = [50.0, 50.0, 50.0]
    # CG system
    pos_aos = generate_cubic_lattice(size=47, spacing=1.0) + 0.3
    N_ATOMS = pos_aos.shape[0]
    vel_aos = np.random.random((N_ATOMS, 3))
    vel_aos = vel_aos - vel_aos.mean(axis=0)
    mass_aos = np.ones(N_ATOMS) * 39.948  # e.g., Argon mass
    diameter_aos = np.ones(N_ATOMS) * 3.4
    sigma_aos = np.ones(N_ATOMS) * 1.0
    epsilon_aos = np.ones(N_ATOMS) * 0.24  # 120K
    charge_aos = np.zeros(N_ATOMS)
    charge_aos[0] = 1
    charge_aos[1] = 1
    charge_aos[3] = 0
    group_aos = np.zeros(N_ATOMS)
    group_aos[:500] = 0

    # Topology
    bonds = [[0, 1, 500.0, 1.5], [1, 2, 500.0, 1.5], [2, 3, 500, 1.5]]
    angles = [[0, 1, 2, 50.0, 109 / 180 * np.pi], [1, 2, 3, 50, 109 / 180 * np.pi]]
    dihes = [[0, 1, 2, 3, 1.5, 0.0, 0.0, 0.0]]
    excls = [[0, 1, 0.0, 0.0], [1, 2, 0.0, 0.0], [0, 2, 0.0, 0.0], [2, 3, 0.0, 0.0], [1, 3, 0.0, 0.0], [0, 3, 0.5, 0.5]]

    engine = MDEngine(n_atoms=N_ATOMS, box_size=BOX_SIZE, r_cut=2.5, skin=0.25, dt=0.002, temperature=120)

    # API OF PAIRWISE FORCE INJECTION
    custom_force_code = """
    if (gi == 1 && gj == 1) {
        float r_mag = sqrt(r_sq);
        float sigma_ij = 0.5f * (d_i + d_j); // Custom mix rule
        if (r_mag < sigma_ij) {
            float soft_repulse = 0.0f * (1.0f - r_mag/sigma_ij);
            custom_f_mag = 0 * soft_repulse / r_mag; 
            custom_u = 0 * 25.0f * pow(1.0f - r_mag/sigma_ij, 2);
        }
    }
    """
    engine.codegen.inject("INJECT_PAIR_FORCE", custom_force_code)

    engine.compile()
    engine.load_initial_aos(pos_aos, vel_aos, mass_aos, diameter_aos, sigma_aos, epsilon_aos, charge_aos, group_aos)

    engine.load_exclusions(excls)
    engine.load_bonds(bonds)
    engine.load_angles(angles)
    engine.load_dihedrals(dihes)

    print("Running...")
    s = time.time()
    Nstep = 50001
    engine.run(steps=Nstep, ensemble="LANGEVIN")
    print(f"Time for {Nstep} steps: {time.time() - s:.4f} seconds tps {Nstep / (time.time() - s):.4f} steps/s")
    print(f"Avg steps per nl update: {Nstep / engine._num_nl_update}")

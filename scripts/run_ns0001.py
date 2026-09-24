"""Generate the complete NS-0001 verification result package.

This script is the reproducible source for the committed NS-0001 evidence.
"""
from pathlib import Path
import sys

# Running ``python scripts/run_ns0001.py`` places ``scripts/`` rather than the
# repository root on sys.path. Add the root explicitly so the script is
# reproducible locally and in GitHub Actions without relying on PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import matplotlib.pyplot as plt
from src.physics.axisymmetric_fvm import FVMConfig, solve, inlet_profile
from src.reporting.run_package import write_residual_history

OUT=ROOT/'generated/NS-0001'; FIG=OUT/'figures'; FIG.mkdir(parents=True,exist_ok=True)
pressure_ref=736.92179; vmax_ref=0.26663
cfg=FVMConfig(nz=81,nr=81,max_iterations=5000,tolerance=1e-7)
r=solve(cfg); exact=inlet_profile(cfg,r.r_m)
p_err=abs(r.pressure_drop_pa-pressure_ref)/pressure_ref
v_err=abs(r.max_velocity_m_s-vmax_ref)/vmax_ref

# Release evidence must be produced by the same iterative solver used for NS-0001.
mesh_levels=(21,41,81,161)
mesh=[]
for nr in mesh_levels:
    mr=solve(FVMConfig(nz=41,nr=nr,max_iterations=5000,tolerance=1e-7))
    mesh.append({'nr':nr,'converged':bool(mr.converged),'pressure_drop_relative_error':abs(mr.pressure_drop_pa-pressure_ref)/pressure_ref,'velocity_profile_relative_L2_error':mr.velocity_profile_relative_l2_error})
pmesh=[x['pressure_drop_relative_error'] for x in mesh]; umesh=[x['velocity_profile_relative_L2_error'] for x in mesh]
mesh_pass=all(x['converged'] for x in mesh) and all(b<a for a,b in zip(pmesh,pmesh[1:])) and all(b<a for a,b in zip(umesh,umesh[1:])) and pmesh[-1]<5e-5 and umesh[-1]<5e-5

release={'analytical_pressure_drop':'pass' if p_err<1e-3 else 'fail','analytical_velocity_profile':'pass' if r.velocity_profile_relative_l2_error<5e-4 else 'fail','no_slip_walls':'pass' if max(abs(r.uz_m_s[:,0]).max(),abs(r.uz_m_s[:,-1]).max())<1e-12 else 'fail','mass_conservation':'pass' if abs(r.relative_mass_imbalance)<1e-8 else 'fail','residual_convergence':'pass' if r.converged else 'fail','mesh_convergence':'pass' if mesh_pass else 'fail'}
verification={'schema_version':'1.1','simulation_id':'NS-0001','status':'verified-straight-annulus' if all(v=='pass' for v in release.values()) else 'numerical-verification','reference':{'pressure_drop_Pa':pressure_ref,'max_axial_velocity_m_s':vmax_ref},'numerical_result':{'pressure_drop_Pa':r.pressure_drop_pa,'pressure_drop_relative_error':p_err,'max_axial_velocity_m_s':r.max_velocity_m_s,'max_velocity_relative_error':v_err,'velocity_profile_relative_L2_error':r.velocity_profile_relative_l2_error,'relative_mass_imbalance':r.relative_mass_imbalance,'final_continuity_residual':r.residuals.continuity[-1],'final_axial_momentum_residual':r.residuals.axial_momentum[-1],'final_radial_momentum_residual':r.residuals.radial_momentum[-1],'iterations':r.iterations,'runtime_seconds':r.runtime_seconds},'mesh_convergence':{'solver':'src.physics.axisymmetric_fvm.solve','levels':mesh,'acceptance':{'monotonic_pressure_error_reduction':True,'monotonic_velocity_error_reduction':True,'finest_pressure_relative_error_lt':5e-5,'finest_velocity_relative_L2_error_lt':5e-5}},'release_gates':release,'overall_verification':'STRAIGHT-ANNULUS NUMERICAL VERIFICATION PASSED' if all(v=='pass' for v in release.values()) else 'NOT YET VERIFIED','scope_warning':'This verifies only the fully developed straight concentric annulus numerical benchmark. The Entry-009 solver is not a general 2-D tapered-geometry CFD solver and does not model tip effects, radial flow, axial development, or recirculation.'}
(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n'); write_residual_history(r,OUT/'residual_history.csv')
Z,R=np.meshgrid(r.z_m*1000,r.r_m*1000,indexing='ij')
def save(fig,name): fig.tight_layout(); fig.savefig(FIG/name,dpi=140,bbox_inches='tight'); plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4)); c=ax.contourf(Z,R,r.uz_m_s,30); fig.colorbar(c,ax=ax,label='Axial velocity (m/s)'); ax.set(xlabel='Axial position (mm)',ylabel='Radius (mm)',title='NS-0001 Axial Velocity Field — Fully Developed Benchmark'); save(fig,'velocity-field.png')
fig,ax=plt.subplots(figsize=(9,4)); c=ax.contourf(Z,R,r.pressure_pa,30); fig.colorbar(c,ax=ax,label='Gauge pressure (Pa)'); ax.set(xlabel='Axial position (mm)',ylabel='Radius (mm)',title='NS-0001 Pressure Field — Straight Annulus'); save(fig,'pressure-field.png')
fig,ax=plt.subplots(figsize=(6,5)); ax.plot(r.uz_m_s[len(r.z_m)//2],r.r_m*1000,label='Numerical'); ax.plot(exact,r.r_m*1000,'--',label='Analytical'); ax.set(xlabel='Axial velocity (m/s)',ylabel='Radius (mm)',title='Velocity Profile Verification'); ax.legend(); ax.grid(True,alpha=.25); save(fig,'velocity-profile.png')
fig,ax=plt.subplots(figsize=(7,5)); it=np.arange(1,r.iterations+1); ax.semilogy(it,r.residuals.continuity,label='Flow/continuity'); ax.semilogy(it,r.residuals.axial_momentum,label='Axial momentum'); ax.set(xlabel='Iteration',ylabel='Normalized residual',title='NS-0001 Residual Convergence'); ax.legend(); ax.grid(True,which='both',alpha=.25); save(fig,'residual-convergence.png')
print(json.dumps({'numerical_result':verification['numerical_result'],'mesh_convergence':verification['mesh_convergence'],'release_gates':release},indent=2))

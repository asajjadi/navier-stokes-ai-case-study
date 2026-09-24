"""Write traceable CFD run packages. Numerical values come only from solver output."""
from __future__ import annotations
import csv,json
from pathlib import Path

def verification_dict(result):
    return {"schema_version":"1.0","simulation_id":"NS-0001","status":"coupled-solver-result","numerical_result":{"pressure_drop_Pa":result.pressure_drop_pa,"max_axial_velocity_m_s":result.max_velocity_m_s,"velocity_profile_relative_L2_error":result.velocity_profile_relative_l2_error,"relative_mass_imbalance":result.relative_mass_imbalance,"final_continuity_residual":result.residuals.continuity[-1],"final_axial_momentum_residual":result.residuals.axial_momentum[-1],"final_radial_momentum_residual":result.residuals.radial_momentum[-1],"iterations":result.iterations,"runtime_seconds":result.runtime_seconds},"overall_verification":"PENDING FULL RELEASE GATES"}

def write_residual_history(result,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['iteration','continuity','axial_momentum','radial_momentum'])
        for i,v in enumerate(zip(result.residuals.continuity,result.residuals.axial_momentum,result.residuals.radial_momentum),1):w.writerow([i,*v])

def write_verification(result,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(verification_dict(result),indent=2)+'\n')

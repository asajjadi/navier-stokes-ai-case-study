"""Axisymmetric incompressible Navier-Stokes verification solver core.

The straight concentric annulus is the mandatory verification geometry. Entry
009 adds an iterative pressure-gradient/velocity coupling for NS-0001. Tapered
geometry remains gated until straight-annulus verification is complete.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .annular_baseline import analytical_velocity, pressure_gradient_from_flow

@dataclass(frozen=True)
class FVMConfig:
    rho: float=1060.; mu: float=.0035; length_m: float=.1
    vessel_radius_m: float=.002; catheter_radius_m: float=.001
    flow_rate_m3_s: float=100e-6/60.; nz: int=81; nr: int=41
    max_iterations: int=5000; tolerance: float=1e-7
    velocity_relaxation: float=.7; pressure_relaxation: float=.3
    def validate(self):
        if self.rho<=0 or self.mu<=0: raise ValueError('Density and viscosity must be positive.')
        if self.flow_rate_m3_s<=0: raise ValueError('Flow rate must be positive.')
        if not 0<self.catheter_radius_m<self.vessel_radius_m: raise ValueError('Catheter radius must lie inside vessel radius.')
        if self.length_m<=0: raise ValueError('Domain length must be positive.')
        if self.nz<5 or self.nr<5: raise ValueError('At least five nodes are required per direction.')

@dataclass
class ResidualHistory:
    continuity:list[float]=field(default_factory=list); axial_momentum:list[float]=field(default_factory=list); radial_momentum:list[float]=field(default_factory=list)

@dataclass
class FVMResult:
    z_m:np.ndarray; r_m:np.ndarray; uz_m_s:np.ndarray; ur_m_s:np.ndarray; pressure_pa:np.ndarray
    residuals:ResidualHistory; inlet_flow_m3_s:float; outlet_flow_m3_s:float; pressure_drop_pa:float
    iterations:int; converged:bool; runtime_seconds:float=0.; max_velocity_m_s:float=0.; velocity_profile_relative_l2_error:float=0.
    @property
    def relative_mass_imbalance(self): return float((self.outlet_flow_m3_s-self.inlet_flow_m3_s)/self.inlet_flow_m3_s)

def build_grid(config=FVMConfig()):
    config.validate(); return np.linspace(0,config.length_m,config.nz),np.linspace(config.catheter_radius_m,config.vessel_radius_m,config.nr)

def inlet_profile(config,r):
    G=pressure_gradient_from_flow(config.flow_rate_m3_s,config.catheter_radius_m,config.vessel_radius_m,config.mu)
    return analytical_velocity(r,config.catheter_radius_m,config.vessel_radius_m,config.mu,G)

def axisymmetric_flow_rate(r,uz): return float(2*np.pi*np.trapezoid(uz*r,r))
def continuity_residual(z,r,uz,ur): return np.gradient(uz,z,axis=0,edge_order=2)+np.gradient(ur*r[None,:],r,axis=1,edge_order=2)/r[None,:]

def _scalar_axisymmetric_laplacian(z,r,f):
    dz=z[1]-z[0]; dr=r[1]-r[0]; lap=np.zeros_like(f,float)
    lap[1:-1,1:-1]=(f[2:,1:-1]-2*f[1:-1,1:-1]+f[:-2,1:-1])/dz**2+(f[1:-1,2:]-2*f[1:-1,1:-1]+f[1:-1,:-2])/dr**2+(f[1:-1,2:]-f[1:-1,:-2])/(2*dr*r[None,1:-1])
    return lap

def axial_momentum_residual(config,z,r,uz,ur,p):
    return config.rho*(uz*np.gradient(uz,z,axis=0,edge_order=2)+ur*np.gradient(uz,r,axis=1,edge_order=2))+np.gradient(p,z,axis=0,edge_order=2)-config.mu*_scalar_axisymmetric_laplacian(z,r,uz)

def radial_momentum_residual(config,z,r,uz,ur,p):
    vl=_scalar_axisymmetric_laplacian(z,r,ur)-ur/(r[None,:]**2)
    return config.rho*(uz*np.gradient(ur,z,axis=0,edge_order=2)+ur*np.gradient(ur,r,axis=1,edge_order=2))+np.gradient(p,r,axis=1,edge_order=2)-config.mu*vl

def rms(a,interior=True):
    x=a[1:-1,1:-1] if interior and min(a.shape)>2 else a; return float(np.linalg.norm(x)/np.sqrt(x.size))

def pressure_poisson_matrix(z,r):
    nz,nr=len(z),len(r); iz=nz-2; ir=nr-2; n=iz*ir; dz=z[1]-z[0]; dr=r[1]-r[0]; A=sparse.lil_matrix((n,n))
    def k(i,j): return (i-1)*ir+(j-1)
    for i in range(1,nz-1):
      for j in range(1,nr-1):
        row=k(i,j); rr=r[j]; az=1/dz**2; ap=1/dr**2; ar=1/(2*rr*dr); west=east=az; south=ap-ar; north=ap+ar; diag=-(2*az+2*ap)
        if i>1:A[row,k(i-1,j)]=west
        else:diag+=west
        if i<nz-2:A[row,k(i+1,j)]=east
        else:diag+=east
        if j>1:A[row,k(i,j-1)]=south
        else:diag+=south
        if j<nr-2:A[row,k(i,j+1)]=north
        else:diag+=north
        A[row,row]=diag
    A[0,:]=0; A[0,0]=1; return A.tocsr()

def solve_pressure_correction(z,r,source):
    A=pressure_poisson_matrix(z,r); b=np.asarray(source[1:-1,1:-1],float).ravel(); b[0]=0; x=spsolve(A,b); pc=np.zeros_like(source); pc[1:-1,1:-1]=x.reshape((len(z)-2,len(r)-2)); pc[0]=pc[1]; pc[-1]=pc[-2]; pc[:,0]=pc[:,1]; pc[:,-1]=pc[:,-2]; return pc

def initialize_straight_annulus(config=FVMConfig()):
    z,r=build_grid(config); profile=inlet_profile(config,r); uz=np.repeat(profile[None,:],config.nz,axis=0); ur=np.zeros_like(uz); G=pressure_gradient_from_flow(config.flow_rate_m3_s,config.catheter_radius_m,config.vessel_radius_m,config.mu); p=G*(config.length_m-z)[:,None]*np.ones((1,config.nr)); div=continuity_residual(z,r,uz,ur); ax=axial_momentum_residual(config,z,r,uz,ur,p); rad=radial_momentum_residual(config,z,r,uz,ur,p); hist=ResidualHistory([rms(div)],[rms(ax)],[rms(rad)]); q0=axisymmetric_flow_rate(r,uz[0]); q1=axisymmetric_flow_rate(r,uz[-1]); return FVMResult(z,r,uz,ur,p,hist,q0,q1,float(G*config.length_m),0,False)

def solve(config=FVMConfig()):
    """Iteratively recover the straight-annulus solution from a non-exact field.

    For this fully developed verification problem, pressure-velocity coupling
    reduces to finding the pressure gradient whose discrete annular momentum
    solution carries the prescribed flow. This is deliberately narrower than
    the later general 2-D SIMPLE solver and must not be used for tapered cases.
    """
    config.validate(); t0=time.perf_counter(); z,r=build_grid(config); dr=r[1]-r[0]; n=config.nr-2
    # Radial cylindrical diffusion matrix for interior axial velocity.
    A=np.zeros((n,n))
    for j in range(1,config.nr-1):
        k=j-1; rr=r[j]; am=1/dr**2-1/(2*rr*dr); ap=1/dr**2+1/(2*rr*dr); A[k,k]=-2/dr**2
        if k>0:A[k,k-1]=am
        if k<n-1:A[k,k+1]=ap
    # Non-exact initial state: zero interior velocity and half analytical gradient.
    G_exact=pressure_gradient_from_flow(config.flow_rate_m3_s,config.catheter_radius_m,config.vessel_radius_m,config.mu); G=.5*G_exact
    u=np.zeros(config.nr); hist=ResidualHistory(); converged=False
    for it in range(1,config.max_iterations+1):
        rhs=np.full(n,-G/config.mu); u_star=np.zeros_like(u); u_star[1:-1]=np.linalg.solve(A,rhs)
        u_new=(1-config.velocity_relaxation)*u+config.velocity_relaxation*u_star; u_new[0]=u_new[-1]=0.
        q=axisymmetric_flow_rate(r,u_new); flow_err=(config.flow_rate_m3_s-q)/config.flow_rate_m3_s
        # Pressure correction for this verification geometry: Q is linear in G.
        Gcorr=G*(config.flow_rate_m3_s/max(q,1e-30)-1.); G_new=G+config.pressure_relaxation*Gcorr
        uz=np.repeat(u_new[None,:],config.nz,axis=0); ur=np.zeros_like(uz); p=G_new*(config.length_m-z)[:,None]*np.ones((1,config.nr))
        div=continuity_residual(z,r,uz,ur); ax=axial_momentum_residual(config,z,r,uz,ur,p); rad=radial_momentum_residual(config,z,r,uz,ur,p)
        hist.continuity.append(abs(float(flow_err))); hist.axial_momentum.append(rms(ax)/max(abs(G_new),1e-30)); hist.radial_momentum.append(rms(rad))
        u=u_new; G=G_new
        if abs(flow_err)<config.tolerance and hist.axial_momentum[-1]<1e-5:
            converged=True; break
    uz=np.repeat(u[None,:],config.nz,axis=0); ur=np.zeros_like(uz); p=G*(config.length_m-z)[:,None]*np.ones((1,config.nr)); q0=axisymmetric_flow_rate(r,uz[0]); q1=axisymmetric_flow_rate(r,uz[-1]); exact=inlet_profile(config,r); rel=float(np.linalg.norm(u-exact)/np.linalg.norm(exact)); runtime=time.perf_counter()-t0
    return FVMResult(z,r,uz,ur,p,hist,q0,q1,float(G*config.length_m),it,converged,runtime,float(np.max(u)),rel)

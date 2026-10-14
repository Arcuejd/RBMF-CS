"""Bounded 2-D quadrature: integrate gamma and delta analytically."""
import numpy as np
from scipy.special import roots_legendre,logsumexp,gammaln

def collapsed_log_joint(alpha,rho,basis,low,h,y,config,tau):
    c=config;n=len(low);a=np.asarray(alpha);r=np.asarray(rho)
    energy=a*a*float(basis@basis)-2*a*r*float(basis@low)+r*r*float(low@low)
    obs=np.sum(y*y)-2*a*float(h@y)+a*a*float(h@h)
    gamma=c.a_gamma*np.log(c.b_gamma)-gammaln(c.a_gamma)+gammaln(c.a_gamma+.5)-.5*np.log(2*np.pi)-(c.a_gamma+.5)*np.log(c.b_gamma+.5*a*a)
    delta=c.a_delta*np.log(c.b_delta)-gammaln(c.a_delta)+gammaln(c.a_delta+n/2)-n/2*np.log(2*np.pi)-(c.a_delta+n/2)*np.log(c.b_delta+.5*np.maximum(energy,0))
    return gamma+delta+.5*(np.log(c.kappa_rho0/(2*np.pi))-c.kappa_rho0*(r-c.rho0)**2)+len(y)/2*np.log(tau/(2*np.pi))-.5*tau*obs

def integrate(basis,low,h,y,config,tau,nodes=512,alpha_limit=16.,rho_limit=8.,unbounded=False):
    x,w=roots_legendre(nodes)
    if unbounded:
        aa=np.tan(np.pi*x/2);rr=aa.copy()
        lwa=np.log(w*np.pi/2*(1+aa**2));lwr=lwa.copy()
    else:
        aa=x*alpha_limit;rr=x*rho_limit
        lwa=np.log(w*alpha_limit);lwr=np.log(w*rho_limit)
    # Two passes in small blocks, no full node^2 arrays persisted.
    logz=-np.inf
    for start in range(0,nodes,64):
        a=aa[start:start+64,None];r=rr[None,:]
        z=collapsed_log_joint(a,r,basis,low,h,y,config,tau)+lwa[start:start+64,None]+lwr[None,:]
        logz=np.logaddexp(logz,logsumexp(z))
    moments=np.zeros(5);origin_mass=0.
    for start in range(0,nodes,64):
        a=aa[start:start+64,None];r=rr[None,:]
        z=collapsed_log_joint(a,r,basis,low,h,y,config,tau)+lwa[start:start+64,None]+lwr[None,:]
        p=np.exp(z-logz)
        moments+=np.array([np.sum(p*a),np.sum(p*r),np.sum(p*a*a),np.sum(p*r*r),np.sum(p*a*r)])
        origin_mass+=float(np.sum(p*(abs(a)<.5)))
    return {'nodes':nodes,'alpha_limit':None if unbounded else alpha_limit,'rho_limit':None if unbounded else rho_limit,'unbounded':unbounded,'log_evidence':float(logz),
        'moments':moments.tolist(),'mean_alpha':float(moments[0]),'mean_rho':float(moments[1]),
        'cov_alpha_rho':float(moments[4]-moments[0]*moments[1]),'prob_abs_alpha_below_0p5':origin_mass}

def adaptive_reference(basis,low,h,y,config,tau):
    from scipy.integrate import quad_vec
    offset=float(collapsed_log_joint(0.,0.,basis,low,h,y,config,tau))
    def outer(r):
        def inner(a):
            p=np.exp(collapsed_log_joint(a,r,basis,low,h,y,config,tau)-offset)
            return p*np.array([1.,a,r,a*a,r*r,a*r])
        value,error=quad_vec(inner,-np.inf,np.inf,epsabs=1e-8,epsrel=1e-8,limit=200)
        return value
    value,error=quad_vec(outer,-np.inf,np.inf,epsabs=1e-8,epsrel=1e-8,limit=200)
    return {'moments':(value[1:]/value[0]).tolist(),'log_evidence':float(np.log(value[0])+offset),'outer_error_estimate':float(error)}

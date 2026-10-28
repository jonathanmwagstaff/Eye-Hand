"""Monte-Carlo error budget for the eye->finger->screen ray (see docs/ARCHITECTURE.md).

Screen plane z=0, two cameras ~70 cm apart at the screen, f=1000 px. Prints cursor
error for keypoint noise, calibration errors, time skew and choosing the wrong eye.
"""
import numpy as np
rng=np.random.default_rng(0)
# Screen frame: screen in z=0 plane, +z toward user, mm.
def look(C,T):
    z=(T-C);z/=np.linalg.norm(z);x=np.cross([0,-1,0],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    R=np.stack([x,y,z]);return R,-R@C
K=np.array([[1000,0,640],[0,1000,360],[0,0,1.]])
cams=[look(np.array([-350,-50,0.]),np.array([0,0,450.])),look(np.array([350,-50,0.]),np.array([0,0,450.]))]
def proj(K,R,t,X): x=K@(R@X+t);return x[:2]/x[2]
def tri(Ps,uvs):
    A=[]
    for (K,R,t),uv in zip(Ps,uvs):
        P=K@np.hstack([R,t[:,None]]);A+= [uv[0]*P[2]-P[0],uv[1]*P[2]-P[1]]
    X=np.linalg.svd(np.array(A))[2][-1];return X[:3]/X[3]
def cursor(E,F): s=E[2]/(E[2]-F[2]);return (E+s*(F-E))[:2]
E=np.array([30.,-80,650]);F=np.array([60.,-40,200])  # eye 65cm, finger 45cm in front of eye
P=[(K,R,t) for R,t in cams]
c0=cursor(E,F);print("lever s=",E[2]/(E[2]-F[2]))
def mc(sf,se,Pr=P,n=3000):
    errs=[]
    for _ in range(n):
        Ee=tri(Pr,[proj(K,R,t,E)+rng.normal(0,se,2) for K,R,t in P])
        Fe=tri(Pr,[proj(K,R,t,F)+rng.normal(0,sf,2) for K,R,t in P])
        errs.append(cursor(Ee,Fe)-c0)
    e=np.array(errs);return np.sqrt((e**2).sum(1).mean()),np.linalg.norm(e.mean(0))
for sf,se in [(1,0.5),(2,1),(4,1.5)]:
    r,b=mc(sf,se);print(f"keypoint noise finger {sf}px eye {se}px -> cursor RMS {r:.1f} mm")
# systematic: calibration errors (true cams P, assumed wrong)
def biased(Pw): return np.linalg.norm(cursor(tri(Pw,[proj(K,R,t,E) for K,R,t in P]),tri(Pw,[proj(K,R,t,F) for K,R,t in P]))-c0)
K2=K.copy();K2[0,0]*=1.03;K2[1,1]*=1.03
print("3%% focal error cam1 -> bias %.1f mm"%biased([(K2,cams[0][0],cams[0][1]),P[1]]))
K3=K.copy();K3[0,2]+=280
print("principal point off 280px (old crop bug) -> bias %.0f mm"%biased([(K3,cams[0][0],cams[0][1]),(K3,cams[1][0],cams[1][1])]))
a=np.deg2rad(0.5);Rz=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
print("0.5deg extrinsic rotation error cam1 -> bias %.1f mm"%biased([(K,Rz@cams[0][0],cams[0][1]),P[1]]))
# screen pose error: 1deg tilt barely matters near center; 10mm offset of plane location -> direct shift
# timing: finger moving 300mm/s laterally, cam1 frame 16ms older
for dt in [0.008,0.016,0.033]:
    Fold=F-np.array([300*dt,0,0])
    Fe=tri(P,[proj(K,*cams[0],Fold),proj(K,*cams[1],F)])
    print(f"time skew {dt*1000:.0f}ms @0.3m/s -> cursor err {np.linalg.norm(cursor(E,Fe)-c0):.1f} mm, finger depth err {abs(Fe[2]-F[2]):.1f} mm")
# wrong eye (cyclops vs dominant): 32mm lateral
print("using wrong eye (32mm off) -> %.1f mm"%np.linalg.norm(cursor(E+[32,0,0],F)-c0))

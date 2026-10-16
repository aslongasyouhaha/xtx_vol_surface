"""Fixed-prefix diagnostics: Jensen lower bound versus free convex-order LP."""
import _bootstrap  # project paths for direct script execution
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from xtx_inputs import load_inputs, core_mask
from sanos_generalized_grid import GridProduct
from sanos_generalized import ExactProduct


class FixedKernel(ExactProduct):
    def __init__(self,A):self.A=A
    def kernel(self,anchors,increments,strikes):return self.A
    def advance(self,anchors,increments,q):pass


def fit_kernel(A,anchors,items,hard_earlier=False):
    if not hard_earlier:return FixedKernel(A).fit_step(anchors,np.zeros(len(anchors)),
        np.concatenate([it['market_k'] for it in items]),
        np.concatenate([it['pure_bid'] for it in items]),
        np.concatenate([it['pure_ask'] for it in items]),
        np.concatenate([it['pure_mid'] for it in items]))
    bid,ask,mid=[np.concatenate([it[x] for it in items]) for x in ['pure_bid','pure_ask','pure_mid']]
    n,m=A.shape;I=sparse.eye(n);Z=sparse.csr_matrix((n,n))
    bounds=[(0,None)]*(m+2*n)
    for j in range(len(items[0]['market_k'])):bounds[m+j]=(0,0)
    r=linprog(np.r_[np.zeros(m),1/(ask-bid),1e-8/(ask-bid)],
        A_ub=sparse.vstack([sparse.hstack([A,-I,Z]),sparse.hstack([-A,-I,Z]),
            sparse.hstack([A,Z,-I]),sparse.hstack([-A,Z,-I])]),
        b_ub=np.r_[ask,-bid,mid,-mid],
        A_eq=np.c_[np.array([np.ones(m),anchors]),np.zeros((2,2*n))],
        b_eq=[1,1],bounds=bounds,method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not r.success:raise RuntimeError(r.message)
    return dict(q=r.x[:m])


def model_with(weights):
    m=GridProduct(16385);m.weights=weights.copy();return m


def pair_lookback(items,saved,hard_earlier=False):
    """Local block coordinate experiment, no global optimality claim."""
    a,b=list(items.values())[1:3]
    prefix=saved['density_0'];q1=saved['q_1'].copy();q2=saved['q_2'].copy()
    v1,v2=saved['increment_1'],saved['increment_2']
    A1=model_with(prefix).kernel(a['anchor_k'],v1,a['market_k'])
    trace=[]
    for step in range(7):
        first=model_with(prefix);first.advance(a['anchor_k'],v1,q1)
        second=model_with(first.weights)
        result=second.fit_step(b['anchor_k'],v2,b['market_k'],b['pure_bid'],b['pure_ask'],b['pure_mid'])
        q2=result['q'];f1=A1@q1;f2=result['fit']
        def stats(f,it):
            bid,ask,mid=[np.asarray(it[x]) for x in ['pure_bid','pure_ask','pure_mid']]
            return dict(coverage=float(np.mean((f>=bid-1e-9)&(f<=ask+1e-9))),
                objective=float(np.sum((np.maximum.reduce([bid-f,f-ask,np.zeros(len(f))])+1e-8*abs(f-mid))/(ask-bid))))
        trace.append(dict(iteration=step,earlier=stats(f1,a),later=stats(f2,b)))
        # Independent multiplicative kernels commute; use the same full grid.
        swapped=model_with(prefix);swapped.advance(b['anchor_k'],v2,q2)
        A2=swapped.kernel(a['anchor_k'],v1,b['market_k'])
        q1=fit_kernel(np.vstack([A1,A2]),a['anchor_k'],[a,b],hard_earlier=hard_earlier)['q']
    return dict(method='local alternating two-expiry LP, not globally certified',hard_earlier=hard_earlier,trace=trace)


def relaxed_marginal(previous,item):
    """Free next marginal on same grid, constrained to dominate fixed previous.

    Call-value variables make full-grid convex-order constraints sparse.
    This relaxes independent-increment restrictions, not calendar order.
    """
    s=previous.states;n=len(s);k=np.asarray(item['market_k']);m=len(k)
    d=np.diff(s)
    slope=sparse.diags([-1/d,1/d],[0,1],shape=(n-1,n),format='csr')
    convex=slope[:-1]-slope[1:]
    j=np.searchsorted(s,k)-1;t=(k-s[j])/(s[j+1]-s[j])
    G=sparse.coo_matrix((np.r_[1-t,t],(np.r_[np.arange(m),np.arange(m)],np.r_[j,j+1])),shape=(m,n)).tocsr()
    I=sparse.eye(m);Z=sparse.csr_matrix((m,m))
    def padded(a):return sparse.hstack([a,sparse.csr_matrix((a.shape[0],2*m))])
    A=sparse.vstack([padded(slope),padded(-slope),padded(convex),
        sparse.hstack([G,-I,Z]),sparse.hstack([-G,-I,Z]),
        sparse.hstack([G,Z,-I]),sparse.hstack([-G,Z,-I])],format='csr')
    bid,ask,mid=[np.asarray(item[x]) for x in ['pure_bid','pure_ask','pure_mid']]
    b=np.r_[np.zeros(n-1),np.ones(n-1),np.zeros(n-2),ask,-bid,mid,-mid]
    lower=previous.calls(s)
    bounds=[(max(0,float(x)),None) for x in lower]+[(0,None)]*(2*m)
    bounds[0]=(1-s[0],1-s[0]);bounds[n-1]=(0,0)
    c=np.r_[np.zeros(n),1/(ask-bid),1e-8/(ask-bid)]
    result=linprog(c,A_ub=A,b_ub=b,bounds=bounds,method='highs',
        options={'time_limit':30,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:return dict(success=False,message=result.message)
    fit=G@result.x[:n]
    residual=max(float(np.max(A@result.x-b)),float(np.max(lower-result.x[:n])),0.)
    return dict(success=residual<=2e-7,objective=float(result.fun),
        actual_outside_objective=float(np.sum(np.maximum.reduce([bid-fit,fit-ask,np.zeros(m)])/(ask-bid))),
        max_residual=residual,coverage=float(np.mean((fit>=bid-1e-9)&(fit<=ask+1e-9))))


def main():
    root=Path('data/outputs/generalized_grid')
    items=load_inputs('2025-04-01',.1)
    saved=np.load(root/'2025-04-01_16385.npz')
    previous=GridProduct(16385);rows=[]
    for j,(e,it) in enumerate(items.items()):
        floor=previous.calls(it['market_k']);ask=np.asarray(it['pure_ask']);spread=ask-it['pure_bid']
        unavoidable=floor>ask+1e-9
        row=dict(dte=int(it['dte']),jensen_blocked=int(unavoidable.sum()),
            core_jensen_blocked=int(unavoidable[core_mask(it['quotes'])].sum()),
            jensen_loss_lower_bound=float(np.sum(np.maximum(floor-ask,0)/spread)))
        if it['dte'] in [3,6,7,10]:
            row['relaxed_marginal']=relaxed_marginal(previous,it)
            for name,model,dv in [('fresh_same_increment',GridProduct(16385),saved[f'increment_{j}']),
                    ('fixed_prefix_near_zero_increment',model_with(previous.weights),np.full(len(it['anchor_k']),1e-7))]:
                result=model.fit_step(it['anchor_k'],dv,it['market_k'],it['pure_bid'],it['pure_ask'],it['pure_mid'])
                row[name]=dict(objective=result['objective'],coverage=float(np.mean(
                    (result['fit']>=it['pure_bid']-1e-9)&(result['fit']<=it['pure_ask']+1e-9))))
        rows.append(row);print(json.dumps(row),flush=True)
        previous.weights=saved[f'density_{j}'].copy()
    (root/'april_prefix_diagnosis.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    pair=[pair_lookback(items,saved),pair_lookback(items,saved,hard_earlier=True)]
    (root/'april_pair_lookback.json').write_text(json.dumps(pair,indent=2),encoding='utf8')
    print(json.dumps(pair),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()

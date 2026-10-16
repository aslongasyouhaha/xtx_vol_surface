"""Build the short, saved-result project walkthrough notebook."""
import _bootstrap
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]

def main():
    nb=nbf.v4.new_notebook();cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('# XTX / SANOS：从报价到曲面\n\n本 notebook 展示基础复现的每一步，读取已保存结果，不自动运行校准。'
       '默认日期为2025-06-02。所有市场点均来自实际校准输入。研究参数和限制见根目录 README。')
    code('''from pathlib import Path
import sys, json
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/sanos_core.py').exists())
sys.path[:0] = [str(ROOT/'src')]
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import norm
from IPython.display import display, Image
from xtx_plotting import otm, iv
pio.renderers.default='plotly_mimetype+notebook'
DATE='2025-06-02'
BLOCK='june'
MODEL_DIR=ROOT/'data/outputs/daily_validation'/BLOCK/DATE/'lp'
if not (MODEL_DIR/'quotes.csv').exists():
    raise FileNotFoundError('需要本地校准结果。请先按 README.md 生成结果，再设置 MODEL_DIR。')
quotes=pd.read_csv(MODEL_DIR/'quotes.csv')
model=np.load(MODEL_DIR/'model.npz')
summary=json.loads((MODEL_DIR/'summary.json').read_text(encoding='utf8'))
print(DATE, 'quotes:',len(quotes),'expiries:',quotes.exdate.nunique())
display(quotes[['exdate','dte','strike','forward','k','best_bid','best_offer','volume']].head())''')
    md('## 1. 标准化\n\n$k=K/F(T)$，价格除以 $DF(T)F(T)$。左侧 OTM put 通过平价转为 call。下图检查标准化后的执行价范围。')
    code('''assert np.allclose(quotes.k,quotes.strike/quotes.forward)
fig,ax=plt.subplots(figsize=(10,3))
for e,g in quotes.groupby('exdate'):ax.scatter(g.k,g.dte,s=5)
ax.axvline(1,color='gray',ls='--');ax.set(xlabel='k = K/F',ylabel='Days to expiry',title='Actual calibration observations');plt.show()''')
    md('## 2. 报价筛选\n\n成交量至少100、Vega/√T至少0.001、每期限至少20个有效点；项目另限制1–120日、k≥0.45。'
       '这里只展示筛选后的真实样本，不将被排除的报价画成拟合目标。下表不是原始数据到筛选数据的流失统计。')
    code('''selection=quotes.groupby(['exdate','dte']).agg(quotes=('k','size'),min_k=('k','min'),max_k=('k','max'),min_volume=('volume','min'))
display(selection)
assert quotes.volume.ge(100).all()
assert quotes.source_vega_over_sqrt_t.ge(.001).all()
assert quotes.groupby('exdate').size().ge(20).all()''')
    md('## 3. 方差骨架与平滑\n\n骨架来自 ATM call bid IV，必要时单调化；核方差为 eta 乘骨架方差。eta 是诊断设置。')
    code('''backbone=quotes.groupby('dte')[['atm_total_variance_raw','atm_total_variance']].first()
backbone.plot(marker='o',figsize=(9,3),title=f'ATM variance backbone; eta={summary["eta"]}')
plt.xlabel('Days to expiry');plt.show()''')
    md('## 4. 联合 LP 的价格拟合\n\n求非负混合权重 q，保持总质量、均值为1，并施加期限间凸序。'
       '目标主要惩罚 bid–ask 外误差，而不是要求穿过全部中间价。下面选择当前最长有效期限。')
    code('''EXPIRY=quotes.sort_values('dte').exdate.iloc[-1]
g=quotes[quotes.exdate==EXPIRY].sort_values('k')
fig,ax=plt.subplots(figsize=(10,3))
ax.fill_between(g.k,g.pure_bid,g.pure_ask,alpha=.3,label='bid/ask')
ax.scatter(g.k,g.pure_mid,s=8,label='market mid');ax.plot(g.k,g.fit,label='joint LP')
ax.set(title=f'{DATE} / {EXPIRY}: normalized call',xlabel='k');ax.legend();plt.show()''')
    md('## 5. 隐含波动率切片\n\n每个期限只画参与拟合的报价，模型线限制在该期限的报价范围内。'
       'IV 由 OTM 价格稳定反解；无效数值保留为缺口，不用 Vega 阈值掩盖曲线。')
    code('''expiries=sorted(quotes.exdate.unique())
fig,axs=plt.subplots(int(np.ceil(len(expiries)/3)),3,figsize=(14,3*np.ceil(len(expiries)/3)),squeeze=False)
for ax,e in zip(axs.flat,expiries):
    f=quotes[quotes.exdate==e].sort_values('k');T=float(model[e+'__T'])
    k=np.linspace(f.k.min(),f.k.max(),501)
    v=iv(k,otm(k,model[e+'__anchors'],float(model[e+'__smooth_variance']),model[e+'__q']),T)
    intrinsic=np.maximum(1-f.k.to_numpy(),0)
    ax.fill_between(f.k,iv(f.k.to_numpy(),f.pure_bid.to_numpy()-intrinsic,T),iv(f.k.to_numpy(),f.pure_ask.to_numpy()-intrinsic,T),alpha=.25)
    ax.scatter(f.k,f.source_otm_iv,s=6,c='black');ax.plot(k,v,c='red')
    ax.set_title(f'{int(f.dte.iloc[0])}D');ax.set_xlabel('k');ax.grid(alpha=.2)
for ax in list(axs.flat)[len(expiries):]:ax.set_visible(False)
fig.tight_layout();plt.show()''')
    md('### 多期限 IV 叠加对比\n\n将市场报价点与拟合线放在同一坐标中，方便比较不同期限。'
       '默认选择最短、中间、最长三个有效期限，可修改 `overlay_expiries`。'
       '只使用当前校准样本，因此与历史截图的期限可能不同。'
       '拖动可框选放大，双击恢复；点击图例可隐藏或显示对应期限。')
    code('''from xtx_plotting import calibration_iv_figure
overlay_expiries=[expiries[j] for j in sorted({0,len(expiries)//2,len(expiries)-1})]
overlay_quotes=quotes[quotes.exdate.isin(overlay_expiries)]
iv_overlay=calibration_iv_figure(overlay_quotes,model)
iv_overlay.show(config={'responsive':True,'displaylogo':False})''')
    md('## 6. 混合权重与平滑后的密度\n\n节点权重 q 不是最终连续密度。对数正态核把离散状态平滑成连续分布。'
       '下图限制在观察执行价范围；尾部没有展示完全。')
    code('''e=EXPIRY;a=model[e+'__anchors'];q=model[e+'__q'];w=float(model[e+'__smooth_variance'])
k=np.linspace(g.k.min(),g.k.max(),600);s=np.sqrt(w)
d2=(np.log(a[None,:]/k[:,None])-.5*w)/s
density=(norm.pdf(d2)/(k[:,None]*s))@q
fig,axs=plt.subplots(1,2,figsize=(12,3))
axs[0].stem(a,q,markerfmt=' ');axs[0].set(title='Latent state weights',xlabel='state a')
axs[1].plot(k,density);axs[1].set(title='Smoothed density within quote range',xlabel='k');plt.show()
print('mass:',q.sum(),'mean:',a@q,'minimum q:',q.min())''')
    md('## 7. 连续价格曲面\n\n在最短与最长期限之间插值，不做时间外推。'
       '下图只取所有有效期限共同的执行价范围；连续查询不是新的市场观察。'
       '按住鼠标左键拖动可旋转，滚轮缩放；三维图在 VS Code / Jupyter 中可交互，GitHub 预览不执行交互脚本。')
    code('''from sanos_surface import SanosSurface
surface=SanosSurface.from_directory(MODEL_DIR)
support=quotes.groupby('exdate').k.agg(['min','max'])
kg=np.linspace(support['min'].max(),support['max'].min(),65)
days=np.linspace(quotes.dte.min(),quotes.dte.max(),35)
kk,dd=np.meshgrid(kg,days)
prices=surface.pure_call_days(dd,kk)
price_surface=go.Figure(go.Surface(x=kk,y=dd,z=prices,colorscale='Viridis',
    colorbar=dict(title='Call price'),
    hovertemplate='k=%{x:.4f}<br>DTE=%{y:.2f}<br>Call=%{z:.6f}<extra></extra>'))
price_surface.update_layout(title='Continuous fitted price surface',height=600,
    scene=dict(xaxis_title='k',yaxis_title='DTE',zaxis_title='Normalized call',
               dragmode='orbit',camera=dict(eye=dict(x=1.5,y=1.5,z=1.1))),
    margin=dict(l=0,r=0,t=55,b=0))
price_surface.show(config={'responsive':True,'scrollZoom':True,'displaylogo':False})''')
    md('## 8. 无套利数值审计与多日覆盖率\n\n保存结果包含中间期限、执行价网格的数值审计。'
       '核心定义为 OTM 绝对远期 delta≥0.1，尚非业务批准的报价区域。数值检查和报价覆盖率不等于 Greeks 验收。')
    code('''display(pd.Series(summary['surface_audit']['residuals'],name='violation'))
rows=[]
for p in (ROOT/'data/outputs/daily_validation').glob('*/*/lp/summary.json'):
    r=json.loads(p.read_text(encoding='utf8'))
    rows.append({k:r[k] for k in ['date','eta','expiries','quotes','coverage','core_coverage']})
results=pd.DataFrame(rows).sort_values('date');display(results)
results.set_index('date')[['coverage','core_coverage']].plot.bar(figsize=(11,3),ylim=(.97,1.005));plt.show()''')
    md('## 9. 翼部边界对照\n\n固定报价和 eta，只增加共同左尾状态0.2。已检查的3处回头消失，核心拟合保持。'
       '这是诊断对照，尚未修改默认状态范围，也不保证更极端翼部。')
    code('''wing=ROOT/'data/outputs/wing_boundary'
if (wing/'summary.csv').exists():
    display(pd.read_csv(wing/'summary.csv'))
    display(Image(filename=str(wing/'2025-04-03_43D.png')))
else:
    print('可选翼部对照尚未生成：python scripts/research_xtx.py wing-boundary')''')
    md('## 10. 延伸探索：广义模型、QP 与 DLV\n\n'
       '广义模型按同一观察日的到期期限递推；核心优先是可选的两阶段 LP，先保护核心误差，再拟合翼部。'
       '跨日 QP 以先前交易日曲面为参考，研究稳定更新；DLV 研究离散局部波动率参数化及直接校准。'
       '以下只比较广义模型的目标函数，不代表基础 LP 结果，也不表示 QP 或 DLV 使用相同方法。'
       '实验范围和未解决的问题见 docs/SUMMARY.md。')
    code('''priority_path=ROOT/'data/outputs/research_exploration/core_priority/results.csv'
if priority_path.exists():
    priority=pd.read_csv(priority_path)
    comparison=priority[priority.experiment.eq('entire_recursion')].pivot(
        index='date',columns='priority',values=['core_coverage','coverage'])
    display(comparison)
    fig,axs=plt.subplots(1,2,figsize=(12,3))
    for ax,metric,title in zip(axs,['core_coverage','coverage'],['Generalized model: core coverage','Generalized model: all quotes']):
        comparison[metric].plot.bar(ax=ax,ylim=(0,1.05),title=title,rot=0)
        ax.set_ylabel('Fraction inside bid/ask')
    fig.tight_layout();plt.show()
else:
    print('可选目标对照尚未生成：python scripts/research_xtx.py core-priority')''')
    nb.cells=cells;nb.metadata.kernelspec=dict(name='python3',display_name='Python 3',language='python')
    target=ROOT/'notebooks/project_overview.ipynb';nbf.write(nb,target);print(target)

if __name__=='__main__':main()

# SANOS/XTX Vol Surface Reflections

这份笔记是这个项目的主观总结。它不是论文翻译，而是这次复现、诊断和工程化之后，我对这个方法的理解。

## 这个方法到底解决什么

SANOS/XTX surface 不是 alpha 模型，也不是判断某个 option quote 真假的工具。它更像是一个：

```text
market data cleaning
+ arbitrage-free projection
+ interpolation engine
```

它把离散、噪声、不规则的市场报价，变成一张可用于定价、风险和回测的连续曲面。

这个定位很重要。因为一旦把它当成“真值模型”，就会误解很多结果。比如某个点不在 bid/ask 里，不代表市场一定错，也不代表模型完全失败。它只说明在当前约束下，模型选择牺牲了这个局部点。

## 价格空间比 IV 空间更可靠

这次最明显的教训是：price surface 看起来规整时，短端 IV surface 仍然可能很吓人。尤其是极短期限、深 OTM、低 vega 的位置，小的价格误差会被 IV 反解放大。

所以主诊断应该永远先看：

```text
pure call price
bid/ask outside error
calendar difference
q density stability
```

IV 图是交易员更熟悉的语言，但不是这个 LP 真正拟合的对象。

## Expiry case 的意义

2025-06-02 的 74D expiry 是这个项目里最有价值的默认示例。它一开始看起来像坏报价，因为 joint LP coverage 只有 0.10。

但拆开以后发现：

```text
74D 自己没有明显 monotonicity 问题
74D 自己没有明显 convexity 问题
邻近 59D/88D 的 ATM calendar 也没有明显冲突
```

所以更准确的结论不是“74D 市场错了”，而是：

```text
当前 model grid、ETA、joint calendar 和稀疏有效 quote 之间有冲突。
```

这也是工程化规则出现的原因：不是人工盯 74D，而是让诊断自动发现类似 74D 的 weak expiry。

## ETA=0.25 的重新定位

我们一开始沿用论文里的 `ETA=0.25`。这作为 benchmark 是合理的，因为它给了一个简单、平滑、可复现的起点。

但多日 sweep 后，我不再把它当生产默认。更好的理解是：

```text
market-only anchors: ETA=0.25 有时可以作为合理平滑。
dense anchors: ETA=0.25 往往偏大。
```

anchor 密度和 ETA 是一对互相替代的自由度：

```text
anchor 越密，ETA 越应该小。
anchor 越稀，ETA 可以稍大。
```

这不是纯理论推导出来的，而是数据上跑出来的工程规律。

## 我们额外做的事情

论文给了无套利曲面的数学结构。这个项目真正多做的是工程化：

```text
1. 明确 quote filtering 和 OTM call-equivalent 口径。
2. 建立 price-space 诊断，而不是只看 IV。
3. 把 single-expiry 和 joint fit 拆开看。
4. 做通用 expiry-case forensic diagnostics，74D 只是默认示例。
5. 做多日 ETA/anchor regime study。
6. 做自动 remediation：只修坏 expiry，不全局加密。
7. 做 wing-weighted QP 稳定化，让 q 不要在翼部跳。
```

其中 wing-weighted QP 不是论文原公式，而是一个工程 regularization。它的作用是：当 LP 有很多差不多好的解时，优先选择更平滑、更不尖、更稳定的 q。

## 最终理解

我会把这个项目的最终结论写成一句话：

```text
SANOS/XTX 是一个优秀的无套利 surface construction framework；
但要把它用于真实市场，需要诊断、过滤、局部补救和稳定化。
```

它的难点不在“能不能跑通 LP”，而在：

```text
哪些 quote 应该进入模型；
哪些 expiry 应该局部加密；
ETA 应该按什么 regime 变化；
什么时候 coverage 低是模型问题，什么时候可能是市场问题；
怎么让 q 稳定而不是只让 price 表面好看。
```

这次项目的收获是：我们不只是复现了论文方法，而是把它推进到一个更接近生产使用的形态。


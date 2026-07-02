# Along-track and cross-track bias effects on DCPA and t\_CPA

## CNS position model

Each aircraft's reported position is truth plus noise plus a systematic bias:

$$\mathbf{p}_i^{\text{obs}} = \mathbf{p}_i^{\text{true}} + \boldsymbol{\epsilon}_i + \mathbf{b}_i$$

where $\boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0},\, \sigma^2 \mathbf{I})$ with $\sigma = \sigma_p / 2.448$ ($\sigma_p$ is `pos_ci95` in metres), and $\mathbf{b}_i$ is a systematic bias along or across the aircraft's track.

The conflict detection (CD) system uses the **relative** observed position between ownship $i$ and intruder $j$:

$$\Delta\mathbf{p}^{\text{obs}} = \Delta\mathbf{p}^{\text{true}} + (\boldsymbol{\epsilon}_j - \boldsymbol{\epsilon}_i) + \underbrace{(\mathbf{b}_j - \mathbf{b}_i)}_{\Delta\mathbf{b}}$$

The relative position error distribution is:

$$\mathbf{e} \sim \mathcal{N}(\Delta\mathbf{b},\; 2\sigma^2 \mathbf{I})$$

The effects of $\Delta\mathbf{b}$ on DCPA and $t_{\text{CPA}}$ depend entirely on its orientation relative to the relative velocity $\Delta\mathbf{v} = \mathbf{v}_j - \mathbf{v}_i$.

---

## Along-track bias (ADS-B latency)

ADS-B latency $\lambda$ causes each aircraft to report where it was $\lambda$ seconds ago. The along-track lag for aircraft $i$ is:

$$\mathbf{b}_i^{\text{at}} = -\lambda\, v_i \begin{pmatrix}\sin\psi_i \\ \cos\psi_i\end{pmatrix} = -\lambda\,\mathbf{v}_i$$

### Key identity

The relative along-track bias simplifies to:

$$\Delta\mathbf{b}^{\text{at}} = \mathbf{b}_j^{\text{at}} - \mathbf{b}_i^{\text{at}} = -\lambda(\mathbf{v}_j - \mathbf{v}_i) = -\lambda\,\Delta\mathbf{v}$$

The bias is **always parallel to** $\Delta\mathbf{v}$, regardless of individual speeds or crossing angle.

### Effect on $t_{\text{CPA}}$

$$t_{\text{CPA}}^{\text{obs}} = -\frac{\Delta\mathbf{p}^{\text{obs}} \cdot \Delta\mathbf{v}}{|\Delta\mathbf{v}|^2} = t_{\text{CPA}}^{\text{true}} + \lambda + \mathcal{N}\!\left(0,\; \frac{2\sigma^2}{|\Delta\mathbf{v}|^2}\right)$$

The conflict always appears exactly $\lambda$ seconds further in the future than it is. This holds for any speeds or crossing angle.

### Effect on DCPA

Since $\Delta\mathbf{b}^{\text{at}} \parallel \Delta\mathbf{v}$, it has zero perpendicular component. Expanding the CPA position shows the latency term cancels exactly:

$$\Delta\mathbf{p}_{\text{CPA}}^{\text{obs}} = \Delta\mathbf{p}_{\text{CPA}}^{\text{true}} + (\boldsymbol{\epsilon}_j - \boldsymbol{\epsilon}_i) + \epsilon_t\,\Delta\mathbf{v}$$

where $\epsilon_t$ is the noise component of $t_{\text{CPA}}$. Therefore:

$$\text{DCPA}^{\text{obs}} = \left|\,\text{DCPA}^{\text{true}} + (\boldsymbol{\epsilon}_j - \boldsymbol{\epsilon}_i)_\perp\,\right|, \qquad (\boldsymbol{\epsilon}_j - \boldsymbol{\epsilon}_i)_\perp \sim \mathcal{N}(0,\; 2\sigma^2)$$

**Latency does not bias the DCPA estimate.** The CD system still perceives the correct conflict distance -- it just perceives it $\lambda$ seconds too late.

---

## Cross-track bias

A cross-track bias $b_{\text{ct}}$ shifts each aircraft's reported position 90 degrees left of its track:

$$\mathbf{b}_i^{\text{ct}} = b_{\text{ct}} \begin{pmatrix}-\cos\psi_i \\ \sin\psi_i\end{pmatrix}$$

The relative cross-track bias is:

$$\Delta\mathbf{b}^{\text{ct}} = b_{\text{ct}} \begin{pmatrix}\cos\psi_i - \cos\psi_j \\ \sin\psi_j - \sin\psi_i\end{pmatrix}$$

This is **not** generally parallel to $\Delta\mathbf{v}$, so it generally has a non-zero perpendicular component and biases DCPA.

### Case 1: 90-degree crossing, equal speeds

With $\psi_i = 0^\circ$, $\psi_j = 90^\circ$, $v_i = v_j = v$:

$$\Delta\mathbf{b}^{\text{ct}} = b_{\text{ct}}\begin{pmatrix}1 \\ 1\end{pmatrix}, \qquad \Delta\mathbf{v} = v\begin{pmatrix}1 \\ -1\end{pmatrix}$$

The perpendicular direction to $\Delta\mathbf{v}$ is $(1,\,1)/\sqrt{2}$, giving:

$$\text{DCPA bias} = \Delta\mathbf{b}^{\text{ct}} \cdot \frac{(1,1)}{\sqrt{2}} = \sqrt{2}\; b_{\text{ct}}$$

The $t_{\text{CPA}}$ shift is zero because $\Delta\mathbf{b}^{\text{ct}} \cdot \Delta\mathbf{v} = b_{\text{ct}}(1,1) \cdot v(1,-1) = 0$.

### Case 2: 90-degree crossing, heterogeneous speeds (10 vs 30 kts)

With $\psi_i = 0^\circ$, $\psi_j = 90^\circ$, $v_i = 10$ kts, $v_j = 30$ kts.

The cross-track bias vectors depend only on heading, not speed, so $\Delta\mathbf{b}^{\text{ct}}$ is unchanged:

$$\Delta\mathbf{b}^{\text{ct}} = b_{\text{ct}}\begin{pmatrix}1 \\ 1\end{pmatrix}$$

But $\Delta\mathbf{v}$ is now asymmetric:

$$\Delta\mathbf{v} = \begin{pmatrix}30 \\ -10\end{pmatrix} \text{ kts}, \qquad |\Delta\mathbf{v}| = \sqrt{1000} = 10\sqrt{10} \text{ kts}$$

The unit vectors along and perpendicular to $\Delta\mathbf{v}$ are:

$$\hat{\Delta\mathbf{v}} = \frac{1}{\sqrt{10}}\begin{pmatrix}3 \\ -1\end{pmatrix}, \qquad \hat{\Delta\mathbf{v}}_\perp = \frac{1}{\sqrt{10}}\begin{pmatrix}1 \\ 3\end{pmatrix}$$

**DCPA bias** (perpendicular component):

$$\text{DCPA bias} = \Delta\mathbf{b}^{\text{ct}} \cdot \hat{\Delta\mathbf{v}}_\perp = b_{\text{ct}}(1,1) \cdot \frac{(1,3)}{\sqrt{10}} = \frac{4\,b_{\text{ct}}}{\sqrt{10}} \approx 1.265\;b_{\text{ct}}$$

**$t_{\text{CPA}}$ bias** (parallel component) -- non-zero this time because $v_i \neq v_j$:

$$t_{\text{CPA}} \text{ bias} = -\frac{\Delta\mathbf{b}^{\text{ct}} \cdot \Delta\mathbf{v}}{|\Delta\mathbf{v}|^2} = -\frac{b_{\text{ct}}(v_j - v_i)}{v_i^2 + v_j^2} = -\frac{20\,b_{\text{ct}}}{1000} = -\frac{b_{\text{ct}}}{50} \text{ s/kts}$$

For $b_{\text{ct}} = 50$ m this gives approximately $-1$ s.

### Comparison across cases

| | Equal speeds | 10 vs 30 kts |
|---|---|---|
| $\Delta\mathbf{b}^{\text{ct}}$ | $b_{\text{ct}}(1,\,1)$ | $b_{\text{ct}}(1,\,1)$ |
| DCPA bias | $\sqrt{2}\;b_{\text{ct}} \approx 1.41\;b_{\text{ct}}$ | $\dfrac{4}{\sqrt{10}}\;b_{\text{ct}} \approx 1.27\;b_{\text{ct}}$ |
| $t_{\text{CPA}}$ bias | $0$ | $-\dfrac{b_{\text{ct}}(v_j - v_i)}{v_i^2 + v_j^2}$ |

The DCPA bias is slightly smaller for heterogeneous speeds because the skewed $\Delta\mathbf{v}$ rotates the perpendicular direction away from $(1,1)$. However, a non-zero $t_{\text{CPA}}$ bias now appears, proportional to the speed difference. This does not occur in the equal-speed case.

---

## Summary

| Bias type | DCPA effect | $t_{\text{CPA}}$ effect |
|---|---|---|
| Along-track (latency $\lambda$) | **None** -- bias is parallel to $\Delta\mathbf{v}$ by construction | $+\lambda$ always, independent of speed or crossing angle |
| Cross-track ($b_{\text{ct}}$), equal speeds | Non-zero: $\sqrt{2}\;b_{\text{ct}}$ at $90^\circ$ | Zero at $90^\circ$ (geometry-dependent in general) |
| Cross-track ($b_{\text{ct}}$), unequal speeds | Non-zero: $\frac{4}{\sqrt{10}}\;b_{\text{ct}}$ at $90^\circ$ | Non-zero: $-\frac{b_{\text{ct}}(v_j-v_i)}{v_i^2+v_j^2}$ |

The fundamental asymmetry is structural: the latency bias has the special form $-\lambda\Delta\mathbf{v}$, which is guaranteed to be parallel to $\Delta\mathbf{v}$ by construction. Cross-track biases carry no such guarantee and will generally corrupt both DCPA and $t_{\text{CPA}}$ estimates, with the $t_{\text{CPA}}$ corruption appearing only when speeds differ.

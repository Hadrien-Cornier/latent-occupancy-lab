# 1) Objective: what we are building and how we judge it

### 1.1 Product goal (one sentence)

Estimate **aggregate person-exposure to indoor PM2.5** over time using only indoor sensors, **without occupancy labels**.

### 1.2 The key correction: occupancy is not binary

The room can contain **0, 1, 2, …, N people**. We treat **occupant count** as the latent variable:

* Latent: (N_t \in {0,1,2,\dots,N_{\max}})

We do **not** treat occupancy as a simple “occupied vs empty” bit.

### 1.3 What “exposure” means here (precise definition)

We target **aggregate person-exposure** (think “person-minutes weighted by concentration”):

* **True exposure (synthetic only; from the simulator):**
  [
  E_{\text{true}}=\sum_t PM_{\text{true}}(t)\cdot N_t \cdot \Delta t
  ]

* **Estimated exposure (works in synthetic and real deployments):**
  [
  E_{\text{est}}=\sum_t PM_{\text{obs}}(t)\cdot \mathbb{E}[N_t\mid \text{sensors}] \cdot \Delta t
  ]
  where
  [
  \mathbb{E}[N_t\mid \text{sensors}] = \sum_{n=0}^{N_{\max}} n\cdot P(N_t=n\mid \text{sensors})
  ]

This is the project’s scoreboard. The model that wins is the one with the **lowest exposure error**, not the best occupant counting accuracy. 

### 1.4 Secondary (diagnostic) goals

These are useful for debugging in simulation, but they are not the optimization target:

* Accuracy of ( \mathbb{E}[N_t] ) vs (N_t)
* Calibration of (P(N_t=n))
* Ventilation proxy / inferred air-exchange ( \lambda(t) ) plausibility
* Event flags (cooking, ventilation, filtration, HVAC)

---

# 2) First principles: why this is solvable without labels

### 2.1 Shared driver: ventilation / air-exchange

Many indoor signals are coupled through a shared latent driver: **ventilation / air-exchange rate** ( \lambda(t) ). CO₂, humidity (in absolute units), and temperature are all “pulled” toward outdoor conditions by ventilation. This is the backbone of multi-sensor fusion.  

### 2.2 PM2.5 is special (and annoying)

PM2.5 is not just “ventilation + source.” It also has **extra sinks** (deposition, filtration), so PM2.5 decay is not a pure ventilation tracer. We must model PM dynamics with **swappable sink terms**, and we must **not let PM force ventilation inference** unless explicitly enabled and correctly modeled.  

### 2.3 Why we need a digital twin

Real homes rarely have occupant-count labels, and we need a way to compare models objectively. We build a physics-based simulator (“digital twin”) that generates:

* Multi-sensor time series (CO₂, PM2.5, T, RH)
* Ground truth (N_t), ( \lambda(t) ), events
* Ground truth exposure (E_{\text{true}})

Then we choose models by exposure correctness under stress tests.  

---

# 3) Inputs, outputs, and hard interfaces

### 3.1 Required input time series (1-minute resolution or resampled)

Indoor:

* `co2_ppm`
* `pm25_ugm3`
* `temp_c`
* `rh_pct`
* `timestamp`

Optional outdoor (toggle-driven; not assumed by default):

* `outdoor_temp_c`
* `outdoor_rh_pct` 

### 3.2 Required model outputs

Every model must output:

1. **Count posterior**:
   `p_N[t, n] = P(N_t = n | sensors)` for `n = 0..N_max`
2. **Expected count** (derived):
   `E_N[t] = Σ_n n * p_N[t,n]`
3. **Exposure estimate** (library computes this from PM and `E_N`):

   * cumulative `E_est` for an interval
   * optionally a running exposure time series

Optional model outputs (useful for analysis):

* `lambda_hat[t]` (ventilation proxy)
* event posteriors/flags (cooking, ventilation, filtration, HVAC)
* confidence/uncertainty metrics

### 3.3 Data contract: fail fast

Implement strict validation before modeling:

* timestamps monotonic, no duplicates
* resample to 1-minute grid (with clear rules)
* unit sanity checks + plausible bounds
* missingness report per channel
* optional outdoor channels aligned or resampled

Plan B explicitly calls this out; do it early so you don’t debug ghosts. 

---

# 4) System architecture (modules and responsibilities)

We build a reusable Python package with a clean separation:

1. **Digital Twin** (physics + scenarios + sensor noise)
2. **Features** (signal processing + conversions + signature detectors)
3. **Models** (ladder of unsupervised methods; all output count posteriors)
4. **Evaluation** (exposure-first metrics + ablations + stress sweeps)
5. **Notebooks** only orchestrate; no core logic inside notebooks

This matches the modular “lab” architecture and model ladder from Plan A, the exposure-first focus from Plan B, and the dependency-driven build spine from Plan C.   

### 4.1 Suggested repo layout

```
latent_exposure_lab/
  pyproject.toml
  README.md

  latent_exposure_lab/
    data_contract/
      schema.py
      validate.py

    digital_twin/
      room.py
      weather.py
      scenarios.py
      sensors.py
      engine.py
      dynamics/
        co2.py
        humidity.py
        temperature.py
        pollutant_base.py
        pollutant_vent_only.py
        pollutant_deposition.py
        pollutant_deposition_filtration.py

    features/
      smooth.py
      derivatives.py
      time_features.py
      psychrometrics.py
      signatures.py
      spikes.py

    models/
      base.py
      heuristic_counts.py
      clustering_counts.py
      hmm_counts.py
      skf_counts.py

    eval/
      exposure.py
      metrics.py
      ablations.py
      stress_sweeps.py
      report.py

  notebooks/
    01_generate_synth_data.ipynb
    02_eda_signatures.ipynb
    03_baseline_exposure.ipynb
    04_model_ladder.ipynb
    05_ablations_stress_report.ipynb
```

---

# 5) Digital Twin: generate truth, confounders, and “annoying realism”

## 5.1 Latents produced by the simulator

At each minute t, the simulator produces:

* (N_t) occupant count (integer)
* ( \lambda(t) ) ventilation / air-exchange (time-varying, bursty)
* event flags: cooking, window open, HVAC, filtration
* “true” (noise-free) signals, plus observed (noisy) sensor signals

Plan A already uses scenarios explicitly like “Sleep (2 ppl) → Empty → Evening (2 ppl)” and Plan C defines modes as ((N_{people}, VentLevel)); we make that explicit and machine-readable.  

## 5.2 Core dynamics (simple, extensible ODE/discrete approximation)

Use one-zone mass-balance style dynamics.

### CO₂ dynamics (human source scales with N)

[
\dot C = \frac{S_{co2}(N_t)}{V} - \lambda(t)\cdot (C - C_{bg})
]

* (S_{co2}(N_t) = N_t \cdot \alpha_{co2})
* If outdoor CO₂ is not available (default), treat (C_{bg}) as a parameter/latent drift (Plan B constraint). 

### Humidity: compute and simulate in absolute humidity

Convert RH→absolute humidity (H) using psychrometrics. Ventilation becomes identifiable as a pull toward (H_{out}). 

[
\dot H = \frac{S_{H}(N_t, events)}{V} - \lambda(t)\cdot (H - H_{out})
]

Outdoor forcing is a toggle:

* v1 default: (H_{out}) treated as constant or slowly varying latent
* v2: (H_{out}) computed from outdoor T/RH in `weather.py`

### Temperature: ventilation pull + building heat loss + HVAC confound

Include a UA heat-loss term (Plan C) and allow HVAC events that change T without meaningful ventilation changes.  

[
\dot T = \text{internal gains}(N_t, cooking) - \lambda(t)(T - T_{out}) - UA(T - T_{out}) + \text{HVAC}(t)
]

### PM2.5 dynamics: pluggable sinks (must be swappable)

PM is implemented as a plugin class so assumptions can change by swapping one file/class (your explicit requirement).

General form:
[
\dot P = \frac{S_{pm}(events)}{V} - \lambda(t)(P - P_{out}) - k_{dep}P - k_{filt}(t)P
]

Implementations:

* `PMVentOnly` (minimal, for debugging)
* `PMWithDeposition` (adds ( -k_{dep}P ); Plan C realism) 
* `PMWithDepositionAndFiltration` (adds filtration sink; Plan B realism) 

**Critical toggle:** `pm_infers_lambda` default **False** so PM does not “force” ventilation inference unless explicitly enabled with a trusted PM model. 

## 5.3 Scenario scripting (multi-day “life stories”)

Scenarios must be multi-day to learn time-of-day priors.

Include:

* weekday pattern: sleep → leave → return → cooking → sleep
* weekend variability
* window open (λ spike)
* HVAC event (T changes, λ unchanged)
* filtration on/off (PM decay without CO₂/H/T changes)
* cooking spikes (PM + CO₂ + heat + moisture)  

## 5.4 Sensor observation layer

Simulate:

* measurement noise
* drift
* missingness/dropouts
* response lag (esp. PM sensors)

Goal: models that only work in perfect toy data should fail here.

---

# 6) Feature layer: convert raw sensors into model-friendly signals

## 6.1 Standard preprocessing

* resample to 1-minute grid
* robust smoothing (median/EMA)
* derivatives: (dCO₂/dt), (d^2CO₂/dt^2) (and optionally for others)  

## 6.2 Time-of-day encoding (circular)

Compute:

* `hour_sin = sin(2π * hour/24)`
* `hour_cos = cos(2π * hour/24)`

This avoids midnight discontinuity and improves unsupervised identifiability. 

## 6.3 Psychrometrics (RH → absolute humidity)

Implement a reliable conversion utility and store both RH and absolute humidity if helpful, but use **absolute humidity** for physics and most features. 

## 6.4 Signature detectors (cheap confound handling)

Implement signature flags used as features or gating signals:

* **HVAC signature**: large |dT| while |dCO₂| and |dH| are small
  (temperature changing independently)  
* **Ventilation signature**: CO₂ and H both move toward outdoor baselines (and T toward T_out if informative)
* **Cooking signature**: PM spike + CO₂ rise (+ heat/moisture)
* **Filtration signature**: PM drops without matching CO₂/H changes

These allow robust behavior without fully modeling HVAC controllers (your stated preference). 

---

# 7) Model layer: multiple unsupervised models, unified “count posterior” API

## 7.1 The base interface (non-negotiable)

Every model implements:

* `fit(df_or_list_of_df) -> self`
* `predict_count_posterior(df) -> p_N`
  returns array shape `[T, N_max+1]`
* `predict_expected_count(df) -> E_N`
  default implementation: `E_N[t] = Σ_n n * p_N[t,n]`
* `predict_aux(df) -> dict` (optional: lambda_hat, event probs)
* `get_params() -> dict`

This enforces comparability and enables the ablation runner.

## 7.2 Choose a practical state representation: exact counts vs bins

Add configuration:

* `count_state_mode = "binned" | "exact"`
* `N_max` (integer maximum count considered)

Recommended default:

* `"binned"` with bins like:

  * `0`, `1`, `2`, `3+` (mapped to representative counts or a distribution across {3..N_max})
    This improves stability and prevents state explosion.

Binary occupancy is just the special case `N_max=1`.

---

## 7.3 Model 1: Heuristic baseline (count-aware state machine)

Purpose: strong baseline + interpretability + sanity check.

Core ideas to include (from Plan A’s heuristic phase space and equilibrium handling): 

* Use smoothed `dCO₂`, `d²CO₂`, CO₂ level, plus `dH`, `dT`, PM spikes.
* **Equilibrium disambiguation**: “flat” doesn’t mean empty; use **CO₂ level** to distinguish low vs high plateaus. 
* Extend that idea from empty/occupied to **count bins**:

  * “flat at moderately high CO₂” → likely low/med count (depending on ventilation evidence)
  * “flat very high” → likely higher count bin

Output:

* a soft posterior over count bins → converted to `p_N[t,n]`

---

## 7.4 Model 2: Clustering (GMM/DBSCAN) → count bins

Purpose: fast iteration and signature discovery.  

* Feature vector examples:

  * `[dCO₂, d²CO₂, CO₂_level, dH, dT, PM_spike, hour_sin, hour_cos]`
* Fit clustering model; map clusters to count bins based on centroid characteristics.
* Output `p_N` via soft cluster membership / responsibilities.

---

## 7.5 Model 3: Time-inhomogeneous HMM (count states + circadian prior)

Purpose: explainable probabilistic model with time-of-day-dependent transitions.  

* Hidden state: count bins or exact counts
* Transition matrix depends on hour-of-day:

  * high probability of changes during “commute hours”
  * high stability during sleep hours
* Train unsupervised via EM (Baum–Welch variant).
* Output posterior over states → `p_N`.

---

## 7.6 Model 4: Switching Kalman Filter / EKF (gold standard, physics-informed)

Purpose: best performance via multi-sensor fusion and shared ventilation latent.  

Core design:

* Discrete mode includes occupant count + ventilation regime (and optionally HVAC/filtration):

  * Mode example: `(N=n, VentLevel=low/high, HVAC=on/off, Filtration=on/off)`
    Plan C explicitly structures modes this way. 
* Continuous state includes sensor states and (optionally) a ventilation proxy/latent ( \lambda(t) ).
* Sensor fusion: CO₂ + absolute humidity (+ temperature when not HVAC-confounded) provide multiple “views” of ventilation. 

Realism constraints (Plan B):

* PM2.5 includes deposition/filtration sinks; PM does not bias λ unless enabled. 
* HVAC confound handled via signature detection gating or explicit HVAC mode. 

Output:

* `p_mode[t, mode]` from SKF → marginalize to get `p_N[t,n]`

---

# 8) Exposure computation (central library function)

Implement in `eval/exposure.py`:

1. `E_N[t] = Σ_n n * p_N[t,n]`
2. `Exposure = Σ_t pm_obs[t] * E_N[t] * dt_seconds`

Optional but valuable:

* Exposure uncertainty quantiles via sampling count trajectories from `p_N` (Monte Carlo), especially helpful when counts are ambiguous.

---

# 9) Evaluation: exposure-first metrics + diagnostics

## 9.1 Primary metrics (synthetic truth)

* Absolute exposure error: `|E_est - E_true|`
* Relative exposure error: `|E_est - E_true| / max(E_true, eps)`
* Scenario-specific exposure error:

  * cooking spikes
  * post-ventilation windows
  * filtration periods

This is Plan B’s “exposure is the objective” principle, generalized to counts. 

## 9.2 Secondary diagnostics (simulation only)

* RMSE/MAE of `E[N_t]` vs `N_true(t)`
* Confusion matrix across count bins
* Calibration plots for `p_N` (optional)
* λ(t) recovery plausibility (when applicable)

---

# 10) Ablation framework: prove what helps exposure (and what’s cargo cult)

Implement an ablation runner that can enumerate configs, fit models, and produce a report.

## 10.1 Standard ablation grid

Sensors:

* CO₂ only
* CO₂ + abs humidity
* * temperature (with HVAC gating)
* * PM features
* (optional) allow PM to influence λ inference (`pm_infers_lambda=True`)

Physics toggles:

* humidity external mode: fixed vs weather-driven
* PM dynamics: vent-only vs +deposition vs +deposition+filtration
* HVAC confound: off vs on
* filtration events: off vs on

Priors:

* with vs without time-of-day transitions/features 

## 10.2 Stress sweeps (robustness)

Vary:

* ventilation burstiness and baseline leakage
* filtration strength
* outdoor seasonality (T_out close to T_in makes temperature less informative)
* sensor noise/drift/missingness

Output:

* ranked model table by exposure error
* plots of error distributions across scenarios
* short markdown/HTML report

Plan A and Plan B both explicitly want ablations + robustness tests; implement them as a first-class module, not ad-hoc notebook code.  

---

# 11) Configuration: make assumptions swappable, not hard-coded

This project will live or die by its ability to evolve assumptions without refactoring the world.

## 11.1 Required config knobs

* `dt_seconds` (default 60)
* `N_max`
* `count_state_mode` ("binned" default)
* `humidity_external_mode` ("fixed" default; "weather" later)
* `pm_dynamics_class` (vent-only vs deposition vs deposition+filtration)
* `pm_infers_lambda` (default False)
* `use_temperature_in_lambda` (default True but gated by HVAC signature)
* `time_of_day_prior` on/off

## 11.2 “Swappable classes” (explicit extension points)

* `PollutantDynamicsBase`
* `WeatherProviderBase` (optional)
* model classes implementing the count posterior API
* signature detectors as feature plugins

This preserves your ability to revise pollutant/humidity modeling by swapping a class, exactly as requested. 

---

# 12) Implementation roadmap (milestones with acceptance tests)

This follows Plan C’s dependency spine, upgraded to exposure-first and count-aware. 

## Milestone 0 — Project skeleton + data contract

Deliver:

* repo layout
* validation utilities
* one script/notebook that loads a CSV, validates, resamples, plots channels

Acceptance:

* invalid inputs fail fast with helpful errors

## Milestone 1 — Digital twin v1 (multi-day count truth)

Deliver:

* simulator generating multi-day sequences with:

  * `N_true[t]`, `lambda_true[t]`, event flags
  * observed sensors with noise/missingness
* default scenarios: weekday/weekend/cooking/window-open/HVAC/filtration

Acceptance:

* plots clearly show intended signatures and count changes

## Milestone 2 — Exposure pipeline end-to-end

Deliver:

* `compute_exposure(pm_obs, p_N, dt)` and `compute_exposure_from_expected(pm_obs, E_N, dt)`
* baseline “dumb” estimator to validate plumbing (e.g., constant expected count)

Acceptance:

* exposure numbers are reproducible and match synthetic truth when fed `N_true`

## Milestone 3 — Heuristic baseline (count-aware)

Deliver:

* heuristic model producing `p_N`
* uses derivatives + CO₂ level equilibrium disambiguation + multi-sensor confirmation + HVAC signature gating 

Acceptance:

* beats dumb estimator on exposure error across standard scenarios

## Milestone 4 — Clustering + HMM (count states)

Deliver:

* clustering model → `p_N`
* time-inhomogeneous HMM → `p_N`

Acceptance:

* improves exposure error in schedule-driven multi-day scenarios

## Milestone 5 — SKF/EKF (gold standard)

Deliver:

* switching filter model with modes including count and ventilation regime
* supports toggles for humidity external mode and PM sink model
* PM does not force λ unless enabled  

Acceptance:

* best exposure performance in messy scenarios (cooking + ventilation + HVAC + filtration)

## Milestone 6 — Ablations + stress sweeps + report generator

Deliver:

* one command/notebook runs:

  * data generation
  * model fits
  * ablations and stress sweeps
  * report outputs (CSV + plots + markdown)

Acceptance:

* produces a concise ranking of “what complexity actually buys exposure accuracy”

---

# 13) Definition of “done” (strong acceptance criteria)

The project is “complete” when:

1. **Exposure-first loop works end-to-end**: sensors → `p_N` → `E_N` → exposure estimate.
2. **Models are comparable**: all implement the same count posterior API.
3. **Digital twin is credible enough to stress-test**: multi-day, confounders, missingness, sinks.
4. **Ablation harness exists**: you can quantify contribution of each sensor and physics toggle to exposure accuracy.
5. **Assumptions are swappable**: changing PM sinks or humidity external forcing does not require rewriting inference code.
6. **Documentation is usable**: README explains inputs/outputs, configs, and how to run the evaluation report.

---

## A final implementation mantra (so the implementation doesn’t drift)

**If a change doesn’t improve exposure error (or explainably trade it for robustness/uncertainty), it’s not progress.**

Everything else—occupant counting accuracy, pretty state plots, elegant physics—exists to serve that one number.

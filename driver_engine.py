Run echo "===== DRIVER ENGINE START ====="
===== DRIVER ENGINE START =====
==============================================
 UNIVERSAL DRIVER ENGINE v4
==============================================
symbols       : 3
lookback      : 10y
window        : 40
forward       : 40
min_history   : 80
max_drivers   : 3
cluster_sim   : 0.9
event_spacing : 40
cycle_min     : 25
==============================================

===== DRIVER: RKLB =====
[OK] RKLB: history=1460 events=35 drivers=2 phase=DOWNTREND price=64.46
  DRIVER_1 | 추세하락→반등시도 | ACTIVE | sim=94.9% | +5=77.8% | +10=77.8% | +20=55.6% | -5=88.9% | cycle=87.0d | T1=77.84 | T2=81.46 | INV=56.22
  DRIVER_2 | 추세상승→확장 | ACTIVE | sim=87.2% | +5=87.5% | +10=75.0% | +20=75.0% | -5=50.0% | cycle=117.0d | T1=82.99 | T2=102.64 | INV=55.04

===== DRIVER: UBER =====
[OK] UBER: history=1850 events=45 drivers=2 phase=TRANSITION price=70.44
  DRIVER_1 | 박스→반복회귀 | ACTIVE | sim=94.0% | +5=65.5% | +10=44.8% | +20=20.7% | -5=82.8% | cycle=58.5d | T1=74.58 | T2=79.94 | INV=60.70
  DRIVER_2 | 박스→반복회귀 | ACTIVE | sim=83.6% | +5=66.7% | +10=0.0% | +20=0.0% | -5=100.0% | cycle=581.0d | T1=74.23 | T2=74.95 | INV=62.50

===== DRIVER: CBRS =====
[OK] CBRS: history=88 events=1 drivers=1 PROVISIONAL phase=TRANSITION price=198.80
  DRIVER_1 | 박스→반복회귀 | PROVISIONAL | sim=90.8% | success=N/A | T1=204.76 | T2=208.74 | INV=182.20

===== OUTPUT =====
profile : 03_RESULTS/daily/driver_profile.csv
current : 03_RESULTS/daily/driver_current.csv
history : 03_RESULTS/daily/driver_history.csv
rows    : profile=5, current=3, history=50

===== CURRENT DRIVER SUMMARY =====
symbol      as_of  current_price active_driver   pattern       state      phase  similarity_pct  success_rate_pct  hit_5_pct  hit_10_pct  hit_20_pct  stop_5_pct  stop_10_pct  target1  target2  invalidation  occurrences         cycle_status  median_cycle_days
  RKLB 2026-09-18         64.465      DRIVER_1 추세하락→반등시도      ACTIVE  DOWNTREND           94.88             77.78      77.78       77.78       55.56       88.89        66.67  77.8391  81.4629       56.2202            9            VALIDATED               87.0
  UBER 2026-09-18         70.438      DRIVER_1   박스→반복회귀      ACTIVE TRANSITION           94.01             65.52      65.52       44.83       20.69       82.76        51.72  74.5764  79.9424       60.7019           29            VALIDATED               58.5
  CBRS 2026-09-18        198.800      DRIVER_1   박스→반복회귀 PROVISIONAL TRANSITION           90.83               NaN        NaN         NaN         NaN         NaN          NaN 204.7640 208.7400      182.2010            1 INSUFFICIENT_HISTORY                NaN
===== DRIVER ENGINE END =====

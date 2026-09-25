import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ضبط إعدادات الصفحة
st.set_page_config(
    page_title="Flight Delay Intelligence Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# تخصيص المظهر بـ CSS
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .metric-card {
        background-color: #1e222b;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #00d26a;
        margin-bottom: 10px;
    }
    .metric-card-danger {
        background-color: #1e222b;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_app_artifacts():
    models = joblib.load("lightgbm_ensemble_model.pkl")
    artifacts = joblib.load("model_artifacts.pkl")
    return models, artifacts


models, artifacts = load_app_artifacts()

# الهيدر الرئيسي
st.title("✈️ نظام التنبؤ الذكي بتأخير الرحلات الجوية")
st.caption("مبني باستخدام LightGBM 5-Fold Ensemble Model & Advanced Feature Engineering")
st.markdown("---")

# القائمة الجانبية لإدخال البيانات
st.sidebar.header("⚙️ إعدادات الرحلة")

airline = st.sidebar.selectbox("شركة الطيران (Airline)", artifacts["airlines_list"], index=0)
origin = st.sidebar.selectbox("مطار المغادرة (Origin)", artifacts["origins_list"], index=0)
dest_options = [d for d in artifacts["dests_list"] if d != origin]
destination = st.sidebar.selectbox("مطار الوصول (Destination)", dest_options, index=0)

col_s1, col_s2 = st.sidebar.columns(2)
month = col_s1.slider("الشهر", 1, 12, 6)
day = col_s2.slider("اليوم", 1, 31, 15)
day_of_week = st.sidebar.selectbox("يوم الأسبوع (1=الإثنين)", list(range(1, 8)), index=3)

sched_dep_hour = st.sidebar.slider("ساعة المغادرة المجدولة", 0, 23, 14)
sched_arr_hour = st.sidebar.slider("ساعة الوصول المجدولة", 0, 23, 18)
distance = st.sidebar.number_input("المسافة الميلية (Distance)", min_value=50, max_value=5000, value=1200)
scheduled_time = st.sidebar.number_input("زمن الرحلة المجدول (دقائق)", min_value=20, max_value=1000, value=180)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 التشغيل ومؤشرات المطار")
turnaround_min = st.sidebar.number_input("زمن تجهيز الطائرة (Turnaround Min)", 0, 360, 45)
aircraft_daily_leg = st.sidebar.number_input("ترتيب الرحلة اليومي للطائرة (Leg #)", 1, 12, 3)
aircraft_daily_legs = st.sidebar.number_input("إجمالي رحلات الطائرة اليومية", 1, 12, 6)
origin_hourly_flights = st.sidebar.slider("عدد الرحلات بالمطار خلال الساعة", 0, 150, 30)
origin_peak_pressure = st.sidebar.slider("مؤشر ضغط الازدحام بمطار المغادرة", 0.0, 1.0, 0.35)

# معالجة بيانات الإدخال
route = f"{origin}_{destination}"
input_dict = {
    "MONTH": month,
    "DAY": day,
    "DAY_OF_WEEK": day_of_week,
    "AIRLINE": airline,
    "ORIGIN_AIRPORT": origin,
    "DESTINATION_AIRPORT": destination,
    "SCHEDULED_TIME": scheduled_time,
    "DISTANCE": distance,
    "SCHED_DEP_HOUR": sched_dep_hour,
    "SCHED_ARR_HOUR": sched_arr_hour,
    "ORIGIN_HOURLY_FLIGHTS": origin_hourly_flights,
    "AIRCRAFT_DAILY_LEG": aircraft_daily_leg,
    "AIRCRAFT_DAILY_LEGS": aircraft_daily_legs,
    "IS_WEEKEND": 1 if day_of_week in [6, 7] else 0,
    "IS_PEAK_HOUR": 1 if sched_dep_hour in [7, 8, 9, 16, 17, 18, 19] else 0,
    "IS_RED_EYE": 1 if sched_dep_hour in [23, 0, 1, 2, 3, 4] else 0,
    "IS_OVERNIGHT_ARRIVAL": 1 if sched_arr_hour < sched_dep_hour else 0,
    "HOUR_SIN": np.sin(2 * np.pi * sched_dep_hour / 24.0),
    "HOUR_COS": np.cos(2 * np.pi * sched_dep_hour / 24.0),
    "DOW_SIN": np.sin(2 * np.pi * day_of_week / 7.0),
    "DOW_COS": np.cos(2 * np.pi * day_of_week / 7.0),
    "ROUTE": route,
    "DISTANCE_BAND": int(distance // 500),
    "IS_LONG_HAUL": 1 if distance > 1500 else 0,
    "SCHED_SPEED_MPH": (distance / (scheduled_time + 1e-5)) * 60,
    "IS_FIRST_LEG": 1 if aircraft_daily_leg == 1 else 0,
    "TURNAROUND_MIN": turnaround_min,
    "IS_TIGHT_TURNAROUND": 1 if turnaround_min < 30 else 0,
    "LEG_PROGRESS": aircraft_daily_leg / (aircraft_daily_legs + 1e-5),
    "ROTATION_RISK": (aircraft_daily_leg / (aircraft_daily_legs + 1e-5)) * (1.0 / (turnaround_min + 1e-5)) * 100,
    "ORIGIN_PEAK_PRESSURE": origin_peak_pressure,
    "TURNAROUND_PER_PROGRESS": turnaround_min / ((aircraft_daily_leg / (aircraft_daily_legs + 1e-5)) + 1e-5),
    "TURNAROUND_PER_LEG": turnaround_min / (aircraft_daily_leg + 1e-5),
    "CONGESTION_PRESSURE_PROD": origin_hourly_flights * origin_peak_pressure,
    "LEG_COMPLETION_RATIO": aircraft_daily_leg / (aircraft_daily_legs + 1e-5),
    "SPEED_DISTANCE_RATIO": ((distance / (scheduled_time + 1e-5)) * 60) / (distance + 1e-5),
}

# إضافة Target Encoding و Historical Delay Rates
gm = artifacts["global_mean"]
input_dict["ORIGIN_HIST_DELAY_RATE"] = artifacts["origin_delay_rates"].get(origin, gm)
input_dict["AIRLINE_HIST_DELAY_RATE"] = artifacts["airline_delay_rates"].get(airline, gm)
input_dict["ROUTE_HIST_DELAY_RATE"] = artifacts["route_delay_rates"].get(route, gm)

for col in ["ROUTE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "AIRLINE"]:
    val = input_dict[col]
    input_dict[f"{col}_TE"] = artifacts["te_maps"][col].get(val, gm)

df_infer = pd.DataFrame([input_dict])
for col in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ROUTE"]:
    df_infer[col] = df_infer[col].astype("category")

# التأكد من ترتيب الأعمدة المتطابق مع التدريب
X_single = df_infer[artifacts["features"]]

# حساب التوقعات عبر الـ 5 نماذج
fold_probs = [m.predict_proba(X_single)[:, 1][0] for m in models]
avg_prob = np.mean(fold_probs)
threshold = artifacts["best_threshold"]
is_delayed = avg_prob >= threshold

# عرض العرض الرئيسي في 3 تبويبات
tab1, tab2, tab3 = st.tabs(["🎯 التنبؤ بالرحلة", "📈 تحليل النماذج والـ 5-Folds", "🔍 مؤشرات الخطر التفاعلية"])

with tab1:
    c1, c2, c3 = st.columns([1.5, 1, 1])

    with c1:
        st.subheader("نتيجة التقييم اللحظي")
        if is_delayed:
            st.error(f"⚠️ **الرحلة معرضة للتأخير!**\n\nاحتمالية التأخير: **{avg_prob * 100:.1f}%**")
        else:
            st.success(f"✅ **الرحلة في موعدها المتوقع**\n\nنسبة الانتظام: **{(1 - avg_prob) * 100:.1f}%**")

        st.progress(float(avg_prob))

    with c2:
        st.metric("العتبة المحسّنة (Optimal Threshold)", f"{threshold:.3f}")
        st.metric("درجة الخطر (Risk Level)", "مرتفع 🔴" if avg_prob > 0.65 else ("متوسط 🟡" if avg_prob >= 0.4 else "منخفض 🟢"))

    with c3:
        st.metric("مسار الرحلة (Route)", route)
        st.metric("معدل التأخير التاريخي للمسار", f"{artifacts['route_delay_rates'].get(route, gm)*100:.1f}%")

    st.markdown("---")
    st.subheader("📊 رسم بياني لاحتمالية التأخير مقارنة بالعتبة")

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=avg_prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "نسبة احتمال التأخير (%)", "font": {"size": 20}},
            delta={"reference": threshold * 100, "increasing": {"color": "red"}, "decreasing": {"color": "green"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff4b4b" if is_delayed else "#00d26a"},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#1e222b"},
                    {"range": [threshold * 100, 100], "color": "#3a1c1c"},
                ],
                "threshold": {
                    "line": {"color": "yellow", "width": 4},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with tab2:
    st.subheader("توزيع توقعات نماذج الـ 5-Folds")
    st.write("يعتمد النظام على أخذ متوسط التوقعات عبر 5 نماذج LightGBM لتقليل الانحراف والتشتت:")

    fold_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(fold_probs))], "Probability": [p * 100 for p in fold_probs]})
    fig_folds = px.bar(
        fold_df,
        x="Fold",
        y="Probability",
        text_auto=".1f",
        color="Probability",
        color_continuous_scale="Reds" if is_delayed else "Greens",
        title="الاحتمالية المحسوبة من كل Fold",
    )
    fig_folds.add_hline(y=threshold * 100, line_dash="dash", line_color="yellow", annotation_text="Optimal Threshold")
    st.plotly_chart(fig_folds, use_container_width=True)

with tab3:
    st.subheader("العوامل المؤثرة المباشرة (Live Feature Stress Test)")

    metrics_df = pd.DataFrame(
        [
            {"العامل": "الازدحام التراكمي بالمطار", "القيمة": f"{input_dict['CONGESTION_PRESSURE_PROD']:.2f}"},
            {"العامل": "زمن التجهيز لكل مرحلة (Turnaround/Progress)", "القيمة": f"{input_dict['TURNAROUND_PER_PROGRESS']:.1f} min"},
            {"العامل": "مخاطر التدوير (Rotation Risk)", "القيمة": f"{input_dict['ROTATION_RISK']:.2f}"},
            {"العامل": "معدل تأخير شركة الطيران التاريخي", "القيمة": f"{input_dict['AIRLINE_HIST_DELAY_RATE']*100:.1f}%"},
        ]
    )
    st.table(metrics_df)
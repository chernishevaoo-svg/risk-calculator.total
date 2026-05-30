import streamlit as st
import json
import numpy as np
import pandas as pd

# Настройка страницы
st.set_page_config(page_title="Калькулятор риска", page_icon="🏥", layout="wide")

@st.cache_data
def load_params():
    with open('model_params.json', 'r', encoding='utf-8') as f:
        return json.load(f)

params = load_params()

st.title("🏥 Оценка риска летального исхода")
st.markdown("Введите показатели пациента для расчета вероятности неблагоприятного исхода и определения группы риска.")
st.markdown("---")

# Извлекаем параметры
feature_names = params['feature_names']
coefficients = np.array(params['coefficients'])
intercept = params['intercept']
orig_medians = params['orig_medians']
inverted_features = params['inverted_features']
orig_min_max = params['orig_min_max']
prob_low = params['prob_low']
prob_high = params['prob_high']
base_lp = params['base_lp']
nom_scale = params['nom_scale']

# Боковая панель с справкой
with st.sidebar:
    st.header("Справка")
    st.info(f"""
    **Группы риска:**
    - 🟢 Низкий риск: P < {prob_low:.1%}
    - 🟡 Умеренный риск: {prob_low:.1%} ≤ P < {prob_high:.1%}
    - 🔴 Высокий риск: P ≥ {prob_high:.1%}
    """)

# Основная форма ввода
st.subheader("Ввод данных пациента")
input_values = {}

# Разбиваем на колонки для красоты
cols = st.columns(min(len(feature_names), 3))

for i, feat in enumerate(feature_names):
    col = cols[i % len(cols)]
    
    min_val = orig_min_max[feat][0]
    max_val = orig_min_max[feat][1]
    median_val = orig_medians[feat]
    step = 0.1 if not float(median_val).is_integer() else 1
    
    with col:
        input_values[feat] = st.number_input(
            f"{feat}", 
            min_value=float(min_val), 
            max_value=float(max_val), 
            value=float(median_val), 
            step=float(step),
            format="%.2f",
            help=f"Диапазон в выборке: [{min_val} - {max_val}]"
        )

# Кнопка расчета
if st.button("Рассчитать риск", type="primary", use_container_width=True):
    # Подготовка вектора признаков
    x_vals = []
    
    for i, feat in enumerate(feature_names):
        val = input_values[feat]
        # Инверсия признака точно по формуле обучения (X_transformed = -X)
        if feat in inverted_features:
            val = -val
        x_vals.append(val)
    
    X_input = np.array(x_vals)
    
    # Расчет логита и вероятности
    linear_predictor = np.dot(X_input, coefficients) + intercept
    probability = float(1 / (1 + np.exp(-linear_predictor)))
    
    # Расчет баллов по номограмме
    points = (linear_predictor - base_lp) * nom_scale
    points = max(0, points)
    
    # Определение группы риска
    if probability < prob_low:
        risk_group = "НИЗКИЙ"
        risk_color = "green"
        risk_emoji = "🟢"
    elif probability < prob_high:
        risk_group = "УМЕРЕННЫЙ"
        risk_color = "orange"
        risk_emoji = "🟡"
    else:
        risk_group = "ВЫСОКИЙ"
        risk_color = "red"
        risk_emoji = "🔴"
        
    # Вывод результатов
    st.markdown("---")
    st.subheader("Результат")
    
    res_cols = st.columns(3)
    
    with res_cols[0]:
        st.metric("Вероятность летальности", value=f"{probability*100:.2f}%")
    
    with res_cols[1]:
        st.metric("Сумма баллов (номограмма)", value=f"{points:.0f}")
        
    with res_cols[2]:
        st.markdown(f"""
        <div style="text-align:center; padding:20px; background-color: {risk_color}33; border-radius:10px; border:2px solid {risk_color}">
            <h3 style="color:{risk_color}; margin:0;">{risk_emoji} {risk_group} РИСК</h3>
        </div>
        """, unsafe_allow_html=True)

    st.caption("⚠️ Внимание: данный калькулятор является вспомогательным инструментом и не заменяет клинического суждения врача.")

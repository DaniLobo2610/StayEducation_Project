import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# REGLAS DE NEGOCIO

# Las siguientes condiciones NO forman parte del modelode Machine Learning:

# Son reglas definidas por el equipo del proyecto para
# generar recomendaciones institucionales posteriores
# a la predicción realizada por el modelo.

# Los umbrales utilizados representan criterios de
# seguimiento académico y pueden modificarse según
# las políticas de cada institución educativa.
# =====================================================

# ==========================================
# REGLAS INSTITUCIONALES
REGLAS = {
    "RIESGO_BAJO":0.30,
    "RIESGO_MEDIO":0.70,

    "NOTA_MINIMA":10,

    "MIN_APROBADAS":2,

    "MIN_APROBADAS_SEGUIMIENTO":4
}

# Configuración inicial de la página
st.set_page_config(
    page_title="StayEdu | Panel Inteligente",
    page_icon="🎓",
    layout="wide"
)

# Estilos CSS Avanzados para modo oscuro premium
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #f3f4f6;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.75rem;
        border: none;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        transform: translateY(-1px);
    }
    div[data-testid="stForm"] {
        background-color: #161b22;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #30363d;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Carga del modelo y objetos con caché
@st.cache_resource
def cargar_recursos():
    modelo = joblib.load("stayedu_model.pkl")
    
    scaler = joblib.load("stayedu_scaler.pkl")
    label_encoder = joblib.load("stayedu_label_encoder.pkl")
    selected_features = joblib.load("stayedu_features.pkl")
    return modelo, scaler, label_encoder, selected_features

modelo, scaler, label_encoder, selected_features = cargar_recursos()

# Diccionario de etiquetas en español
etiquetas = {
    "Curricular units 2nd sem (grade)": "Nota promedio 2do semestre",
    "Curricular units 1st sem (grade)": "Nota promedio 1er semestre",
    "Curricular units 2nd sem (approved)": "Materias aprobadas 2do semestre",
    "Curricular units 1st sem (approved)": "Materias aprobadas 1er semestre",
    "Curricular units 2nd sem (enrolled)": "Materias matriculadas 2do semestre",
    "Curricular units 1st sem (enrolled)": "Materias matriculadas 1er semestre",
    "Tuition fees up to date": "Matrícula al día (1 = Sí, 0 = No)",
    "Scholarship holder": "Tiene beca (1 = Sí, 0 = No)",
    "Debtor": "Tiene deudas pendientes (1 = Sí, 0 = No)",
    "Age at enrollment": "Edad al momento de inscribirse",
    "Admission grade": "Nota de admisión",
    "Gender": "Género (0 = Femenino, 1 = Masculino)",
    "Displaced": "Estudiante desplazado (1 = Sí, 0 = No)",
    "Curricular units 1st sem (evaluations)": "Evaluaciones 1er semestre",
    "Course": "Curso / Carrera",
    "Father's occupation": "Ocupación del padre",
    "Mother's occupation": "Ocupación de la madre",
    "Unemployment rate": "Tasa de desempleo",
    "Mother's qualification": "Nivel educativo de la madre",
    "Father's qualification": "Nivel educativo del padre",
    "GDP": "PIB (GDP)",
}

CURSOS = {
    "33 - Tecnologías de Producción de Biocombustibles": 33,
    "171 - Animación y Diseño Multimedia": 171,
    "8014 - Servicio Social (Horario Nocturno)": 8014,
    "9003 - Agronomía": 9003,
    "9070 - Diseño de Comunicación": 9070,
    "9085 - Enfermería Veterinaria": 9085,
    "9119 - Ingeniería Informática": 9119,
    "9130 - Equinocultura": 9130,
    "9147 - Administración de Empresas": 9147,
    "9238 - Servicio Social": 9238,
    "9254 - Turismo": 9254,
    "9500 - Enfermería": 9500,
    "9556 - Higiene Oral": 9556,
    "9670 - Gestión de Publicidad y Marketing": 9670,
    "9773 - Periodismo y Comunicación": 9773,
    "9853 - Educación Básica": 9853,
    "9991 - Administración de Empresas (Horario Nocturno)": 9991
}

PADRE_ESTUDIOS = {
    "1 - Educación Secundaria (12.º año)": 1,
    "2 - Educación Superior - Licenciatura": 2,
    "3 - Educación Superior - Título Universitario": 3,
    "4 - Educación Superior - Maestría": 4,
    "5 - Educación Superior - Doctorado": 5,
    "6 - Estudios Universitarios Incompletos": 6,
    "9 - 12.º año de escolaridad incompleto": 9,
    "10 - 11.º año de escolaridad incompleto": 10,
    "12 - Otro tipo de 11.º año de escolaridad": 12,
    "14 - 10.º año de escolaridad": 14,
    "18 - Curso General de Comercio": 18,
    "19 - Educación Básica - Tercer Ciclo (9.º, 10.º y 11.º año)": 19,
    "22 - Curso Técnico Profesional": 22,
    "26 - 7.º año de escolaridad": 26,
    "27 - Segundo Ciclo de Educación Secundaria": 27,
    "29 - 9.º año de escolaridad incompleto": 29,
    "30 - 8.º año de escolaridad": 30,
    "34 - Desconocido": 34,
    "35 - No sabe leer ni escribir": 35,
    "36 - Sabe leer pero no completó 4.º grado": 36,
    "37 - Educación Básica - Primer Ciclo (4.º grado)": 37,
    "38 - Educación Básica - Segundo Ciclo (6.º grado)": 38,
    "39 - Curso de Especialización Tecnológica": 39,
    "40 - Educación Superior - Grado (Primer Ciclo)": 40,
    "41 - Curso Superior Especializado": 41,
    "42 - Curso Técnico Superior Profesional": 42,
    "43 - Educación Superior - Maestría (Segundo Ciclo)": 43,
    "44 - Educación Superior - Doctorado (Tercer Ciclo)": 44
}

OCUPACIONES = {
    "0 - Estudiante": 0,
    "1 - Representantes del Poder Legislativo, Directores y Gerentes Ejecutivos": 1,
    "2 - Especialistas en Actividades Intelectuales y Científicas": 2,
    "3 - Técnicos y Profesionales de Nivel Intermedio": 3,
    "4 - Personal Administrativo": 4,
    "5 - Trabajadores de Servicios Personales, Seguridad y Vendedores": 5,
    "6 - Agricultores y Trabajadores Calificados de Agricultura, Pesca y Silvicultura": 6,
    "7 - Trabajadores Calificados de la Industria, Construcción y Artesanos": 7,
    "8 - Operadores de Instalaciones, Máquinas y Ensambladores": 8,
    "9 - Trabajadores No Calificados": 9,
    "10 - Profesiones de las Fuerzas Armadas": 10,
    "90 - Otra situación": 90,
    "99 - Sin información": 99,

    "101 - Oficiales de las Fuerzas Armadas": 101,
    "102 - Sargentos de las Fuerzas Armadas": 102,
    "103 - Otro personal de las Fuerzas Armadas": 103,

    "112 - Directores de Servicios Administrativos y Comerciales": 112,
    "114 - Directores de Hoteles, Restaurantes, Comercio y Otros Servicios": 114,

    "121 - Especialistas en Ciencias Físicas, Matemáticas e Ingeniería": 121,
    "122 - Profesionales de la Salud": 122,
    "123 - Profesores": 123,
    "124 - Especialistas en Finanzas, Contabilidad, Organización Administrativa y Relaciones Públicas": 124,

    "131 - Técnicos de Ciencias e Ingeniería": 131,
    "132 - Técnicos de Nivel Intermedio del Área de la Salud": 132,
    "134 - Técnicos Jurídicos, Sociales, Deportivos y Culturales": 134,
    "135 - Técnicos en Tecnologías de la Información y Comunicación": 135,

    "141 - Oficinistas, Secretarios y Operadores de Procesamiento de Datos": 141,
    "143 - Operadores de Contabilidad, Estadística, Finanzas y Registros": 143,
    "144 - Otro Personal de Apoyo Administrativo": 144,

    "151 - Trabajadores de Servicios Personales": 151,
    "152 - Vendedores": 152,
    "153 - Cuidadores y Asistentes Personales": 153,
    "154 - Personal de Protección y Seguridad": 154,

    "161 - Agricultores Orientados al Mercado y Productores Agropecuarios": 161,
    "163 - Agricultores, Ganaderos, Pescadores y Cazadores de Subsistencia": 163,

    "171 - Trabajadores Calificados de la Construcción (excepto Electricistas)": 171,
    "172 - Trabajadores Calificados de Metalurgia y Metalmecánica": 172,
    "174 - Trabajadores Calificados en Electricidad y Electrónica": 174,
    "175 - Trabajadores de Alimentos, Madera, Textil y Otras Industrias": 175,

    "181 - Operadores de Plantas y Máquinas Fijas": 181,
    "182 - Trabajadores de Ensamblaje": 182,
    "183 - Conductores de Vehículos y Operadores de Equipos Móviles": 183,

    "191 - Personal de Limpieza": 191,
    "192 - Trabajadores No Calificados de Agricultura, Ganadería, Pesca y Silvicultura": 192,
    "193 - Trabajadores No Calificados de Minería, Construcción, Manufactura y Transporte": 193,
    "194 - Auxiliares en Preparación de Alimentos": 194,
    "195 - Vendedores Ambulantes y Prestadores de Servicios en la Vía Pública": 195
}

# --- ENCABEZADO ---
st.title("🎓 StayEdu: Panel Inteligente de Retención Estudiantil")
st.markdown("Plataforma analítica avanzada con Machine Learning Explicable (XAI) para la detección y mitigación del abandono escolar.")
st.markdown("Estudiantes de la clase de IA - Alberto Daniel Lobo, Robí Williams Mejia, Hector Josue Fortín y Joksan Jared Zavala.")
st.markdown("---")

col_form, col_res = st.columns([1.2, 0.8], gap="large")

with col_form:
    st.subheader("📋 Parámetros Académicos y Demográficos")

    #entrada = {}
    #with st.form("formulario_estudiante"):
    #    for var in selected_features:
    #        etiqueta = etiquetas.get(var, var)
    #        entrada[var] = st.number_input(etiqueta, value=0.0, step=1.0, format="%.2f")

    #    st.markdown("")
    #    enviado = st.form_submit_button("🔍 Ejecutar Análisis de Riesgo")
    entrada = {}

    with st.form("formulario_estudiante"):

        # ============================
        # PRIMERA FILA
        # ============================

        col_academica, espacio, col_personal = st.columns([5, 1, 3.5])

        with col_academica:

            st.subheader("📚 Información Académica")

            entrada["Admission grade"] = st.number_input(
                "Nota de admisión",
                min_value=0.0,
                max_value=200.0,
                value=120.0,
                step=1.0
            )

            st.divider()

            st.markdown("**Primer semestre**")

            entrada["Curricular units 1st sem (enrolled)"] = st.number_input(
                "Materias matriculadas",
                min_value=0,
                max_value=20,
                value=5
            )

            entrada["Curricular units 1st sem (evaluations)"] = st.number_input(
                "Evaluaciones realizadas",
                min_value=0,
                max_value=30,
                value=2,
                step=1
            )

            entrada["Curricular units 1st sem (approved)"] = st.number_input(
                "Materias aprobadas",
                min_value=0,
                max_value=20,
                value=5
            )

            entrada["Curricular units 1st sem (grade)"] = st.number_input(
                "Nota promedio",
                min_value=0.0,
                max_value=20.0,
                value=12.0,
                step=0.5
            )

            st.divider()

            st.markdown("**Segundo semestre**")

            entrada["Curricular units 2nd sem (enrolled)"] = st.number_input(
                "Materias matriculadas ",
                min_value=0,
                max_value=20,
                value=5,
                key="mat2"
            )

            entrada["Curricular units 2nd sem (evaluations)"] = st.number_input(
                "Evaluaciones realizadas ",
                min_value=0,
                max_value=30,
                value=2,
                step=1,
                key="eval2"
            )

            entrada["Curricular units 2nd sem (approved)"] = st.number_input(
                "Materias aprobadas ",
                min_value=0,
                max_value=20,
                value=5,
                key="apr2"
            )

            entrada["Curricular units 2nd sem (grade)"] = st.number_input(
                "Nota promedio ",
                min_value=0.0,
                max_value=20.0,
                value=12.0,
                step=0.5,
                key="nota2"
            )

        with col_personal:

            st.subheader("👤 Información Personal")

            entrada["Age at enrollment"] = st.number_input(
                "Edad al ingresar",
                min_value=16,
                max_value=100,
                value=19,
                step=1
            )

            genero = st.selectbox(
                "Género",
                ["Masculino", "Femenino"]
            )

            entrada["Gender"] = 1 if genero == "Masculino" else 0

            desplazado = st.selectbox(
                "¿Es estudiante desplazado?",
                ["No", "Sí"]
            )

            entrada["Displaced"] = 1 if desplazado == "Sí" else 0

        col_economica, espacio2, col_contexto = st.columns([5, 1, 3.5])

        with col_economica:

            st.subheader("💰 Información Económica")

            beca = st.selectbox(
                "¿Posee beca?",
                ["No", "Sí"]
            )
            entrada["Scholarship holder"] = 1 if beca == "Sí" else 0

            matricula = st.selectbox(
                "¿Tiene la matrícula al día?",
                ["Sí", "No"]
            )
            entrada["Tuition fees up to date"] = 1 if matricula == "Sí" else 0

            deudor = st.selectbox(
                "¿Posee deudas con la institución?",
                ["No", "Sí"]
            )
            entrada["Debtor"] = 1 if deudor == "Sí" else 0

            entrada["Unemployment rate"] = st.number_input(
                "Tasa de desempleo (%)",
                min_value=0.0,
                max_value=30.0,
                value=10.0,
                step=0.1
            )

            entrada["GDP"] = st.number_input(
                "PIB",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.1
            )

        with col_contexto:

            st.subheader("🏛 Información Institucional")

            curso = st.selectbox(
                "Carrera",
                options=list(CURSOS.keys())
            )

            entrada["Course"] = CURSOS[curso]

            estudio_padre = st.selectbox(
                "Nivel educativo del padre",
                options=list(PADRE_ESTUDIOS.keys())
            )

            entrada["Father's qualification"] = PADRE_ESTUDIOS[estudio_padre]

            estudio_madre = st.selectbox(
                "Nivel de estudios de la madre",
                options=list(PADRE_ESTUDIOS.keys())
            )

            entrada["Mother's qualification"] = PADRE_ESTUDIOS[estudio_madre]

            ocupacion_padre = st.selectbox(
                "Ocupación del padre",
                options=list(OCUPACIONES.keys())
            )

            entrada["Father's occupation"] = OCUPACIONES[ocupacion_padre]

            ocupacion_madre = st.selectbox(
                "Ocupación de la madre",
                options=list(OCUPACIONES.keys())
            )

            entrada["Mother's occupation"] = OCUPACIONES[ocupacion_madre]


        st.divider()

        enviado = st.form_submit_button(
            "🔍 Ejecutar Análisis de Riesgo",
            use_container_width=True
        )
    st.subheader("📊 Diagnóstico del Estudiante")

    if enviado:
        datos = pd.DataFrame([entrada])[selected_features]


        datos_escalados = scaler.transform(datos)

        prediccion = modelo.predict(datos_escalados)[0]
        probabilidades = modelo.predict_proba(datos_escalados)[0]
        clase_predicha = label_encoder.inverse_transform([prediccion])[0]

        if "Dropout" in label_encoder.classes_:
            idx_dropout = list(label_encoder.classes_).index("Dropout")
            prob_riesgo = probabilidades[idx_dropout]
        else:
            prob_riesgo = max(probabilidades)

        if prob_riesgo < REGLAS["RIESGO_BAJO"]:
            nivel = "Bajo"
            color_badge = "🟢"

        elif prob_riesgo < REGLAS["RIESGO_MEDIO"]:
            nivel = "Medio"
            color_badge = "🟠"

        else:
            nivel = "Alto"
            color_badge = "🔴"

        st.markdown(
            f"<div style='padding: 18px; border-radius: 12px; background-color: #161b22; border: 1px solid #30363d; margin-bottom: 15px;'>"
            f"<p style='margin:0; color: #8b949e; font-size: 14px; font-weight: 500;'>ESTADO PREDICTIVO</p>"
            f"<p style='font-size: 22px; font-weight: 700; color: #58a6ff; margin: 4px 0 0 0;'>{clase_predicha}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Probabilidad de Abandono", value=f"{prob_riesgo * 100:.1f}%")
        with col_m2:
            st.metric(label="Nivel de Riesgo", value=f"{color_badge} {nivel}")

        st.markdown("---")

        st.subheader("💡 Plan de Acción Sugerido")

        # Variables críticas evaluadas directamente desde el diccionario de entrada
        aprobadas_2do = entrada.get('Curricular units 2nd sem (approved)', 5)
        nota_2do = entrada.get('Curricular units 2nd sem (grade)', 12)
        aprobadas_1er = entrada.get('Curricular units 1st sem (approved)', 5)
        nota_1er = entrada.get('Curricular units 1st sem (grade)', 12)
        matricula = entrada.get('Tuition fees up to date', 1)

        if nivel == "Alto" or clase_predicha == "Dropout":
            st.error("⚠️ **Alerta: Alto Riesgo de Abandono (Dropout)**")

            if (
                aprobadas_2do <= REGLAS["MIN_APROBADAS"]
                or nota_2do < REGLAS["NOTA_MINIMA"]
            ):
                st.markdown(f"🔴 **Alerta Académica Crítica (2do Semestre):** El estudiante aprobó pocas materias ({aprobadas_2do}) o su nota promedio es baja ({nota_2do}). **Recomendación:** Asignación urgente a tutorías de refuerzo.")

            if (
                aprobadas_1er <= REGLAS["MIN_APROBADAS"]
                or nota_1er < REGLAS["NOTA_MINIMA"]
            ):
                st.markdown(f"🔴 **Alerta de Arrastre (1er Semestre):** Se detectan deficiencias académicas desde el primer periodo ({aprobadas_1er} materias aprobadas, nota {nota_1er}).")

            if matricula == 0:
                st.markdown("🔴 **Alerta Financiera:** La matrícula no se encuentra al día. **Recomendación:** Contactar a administración para revisar planes de pago o becas.")

            st.markdown("""
            **Plan de Acción Institucional:**
            * Derivar a orientación psicológica y consejería estudiantil.
            * Realizar una revisión de la carga académica para evitar la saturación.
            """)

        elif nivel == "Medio" or clase_predicha == "Enrolled":
            st.warning("⚠️ **Estudiante en Estado de Seguimiento (En curso / Retrasado)**")

            if aprobadas_2do < REGLAS["MIN_APROBADAS_SEGUIMIENTO"]:
                st.markdown(f"🟡 **Seguimiento:** El avance en el segundo semestre es moderado ({aprobadas_2do} materias aprobadas).")

            st.markdown("""
            **Plan de Acción Institucional:**
            * Monitorear de cerca las evaluaciones pendientes del semestre actual.
            * Brindar orientación sobre gestión del tiempo y hábitos de estudio.
            """)

        else:
            st.success("🎉 **Pronóstico Favorable (Graduación)**")
            st.markdown("""
            **Estado Óptimo:**
            * Los indicadores clave (calificaciones, materias aprobadas y situación financiera) muestran estabilidad y éxito académico.
            * **Sugerencia:** Mantener el ritmo de estudio e invitar al estudiante a participar en programas de excelencia o prácticas profesionales.
            """)

        # --- IMPLEMENTACIÓN DE SHAP (Explicabilidad) ---
        st.markdown("---")
        st.subheader("🧠 Explicabilidad del Modelo (SHAP)")
        st.markdown("Factores específicos que impulsaron este resultado para el estudiante:")

        try:
            # Explicador específico para LightGBM
            explainer = shap.TreeExplainer(modelo)

            # SHAP para el registro actual
            shap_values = explainer.shap_values(datos_escalados)

            # Índice de la clase Dropout
            clase_idx = idx_dropout

            # Modelos multiclase
            # Obtener únicamente los valores SHAP de la clase Dropout
            if isinstance(shap_values, list):
                # Versiones antiguas de SHAP
                values_to_plot = shap_values[idx_dropout][0]

            elif hasattr(shap_values, "values"):
                # Versiones nuevas (Explanation)
                values_to_plot = shap_values.values[0, :, idx_dropout]

            else:
                values_to_plot = shap_values[0, :, idx_dropout]

            # Etiquetas en español
            features_labels = [etiquetas.get(f, f) for f in selected_features]

            # Ordenar por impacto absoluto
            indices = np.argsort(np.abs(values_to_plot))[::-1][:10]

            valores = values_to_plot[indices]
            nombres = np.array(features_labels)[indices]

            fig, ax = plt.subplots(figsize=(9, 5))

            colores = ["#d62728" if v > 0 else "#2ca02c" for v in valores]

            ax.barh(nombres, valores, color=colores)

            ax.invert_yaxis()

            ax.axvline(0, color="white", linestyle="--")

            ax.set_facecolor("#161b22")
            fig.patch.set_facecolor("#0e1117")

            ax.tick_params(colors="white")

            st.pyplot(fig)

            st.info("""
                🔴 **Rojo:** factores que aumentan la probabilidad de abandono.

                🟢 **Verde:** factores que disminuyen la probabilidad de abandono y favorecen la graduación.
            """)

        except Exception as e:
            st.error("Error en SHAP:")
            st.exception(e)

    else:
        st.info("Ingrese los datos requeridos en el formulario del panel superior y presione **'Ejecutar Análisis de Riesgo'** para visualizar el diagnóstico.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #8b949e; font-size: 13px;'>StayEdu Analytics Engine — Diseñado para la Gestión y Retención Académica</p>", unsafe_allow_html=True)

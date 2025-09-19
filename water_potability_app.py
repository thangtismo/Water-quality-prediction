# File: water_potability_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import plotly.express as px

# Tiêu đề ứng dụng
st.title("Ứng Dụng Dự Đoán Chất Lượng Nước Uống")

# Mô tả
st.markdown(
    """
Nhập các chỉ số chất lượng nước để dự đoán xem nước có uống được hay không.
Dự đoán được thực hiện bằng mô hình LSTM đã được huấn luyện.
"""
)

# Định nghĩa các đặc trưng
features = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

# Tạo form nhập liệu
st.header("Nhập Chỉ Số Nước")
input_data = {}
with st.form(key="water_form"):
    for feature in features:
        # Cho phép nhập với độ chính xác cao, không giới hạn min_value
        input_data[feature] = st.number_input(
            f"{feature}",
            value=0.0,
            step=0.000000000000001,  # Step rất nhỏ để có thể nhập chính xác
            format="%.15f",  # Hiển thị đến 15 chữ số thập phân
        )

    submit_button = st.form_submit_button(label="Dự đoán")


# Tải pipeline tiền xử lý và mô hình
@st.cache_resource
def load_model_and_pipeline():
    try:
        pipeline = joblib.load(
            "num_pipeline.pkl"
        )  # Đường dẫn tới file pipeline đã lưu
        model = tf.keras.models.load_model(
            "lstm_model.h5"
        )  # Đường dẫn tới file mô hình LSTM
        return pipeline, model
    except FileNotFoundError as e:
        st.error(f"Không tìm thấy file mô hình: {e}")
        return None, None


# Hàm dự đoán
def predict_potability(data, pipeline, model):
    # Chuyển dữ liệu thành DataFrame
    df = pd.DataFrame([data])

    # Tiền xử lý dữ liệu
    processed_data = pipeline.transform(df)

    # Định dạng lại dữ liệu cho LSTM (thêm chiều time step)
    processed_data = processed_data.reshape(
        (processed_data.shape[0], 1, processed_data.shape[1])
    )

    # Dự đoán
    prediction = model.predict(processed_data)
    probability = prediction[0][0]

    # Chuyển đổi xác suất thành nhãn
    label = "Có thể uống" if probability >= 0.5 else "Không thể uống"
    confidence = probability if probability >= 0.5 else 1 - probability

    return label, confidence, probability


# Xử lý khi người dùng nhấn nút "Dự đoán"
if submit_button:
    try:
        # Tải pipeline và mô hình
        pipeline, model = load_model_and_pipeline()

        if pipeline is not None and model is not None:
            # Dự đoán
            label, confidence, probability = predict_potability(
                input_data, pipeline, model
            )

            # Hiển thị kết quả
            st.header("Kết Quả Dự Đoán")
            st.write(f"**Chất lượng nước**: {label}")
            st.write(f"**Độ tin cậy**: {confidence:.2%}")
            st.write(f"**Xác suất có thể uống**: {probability:.2%}")

            # Hiển thị biểu đồ xác suất bằng Plotly
            st.subheader("Biểu đồ Xác Suất")

            # Tạo dữ liệu cho biểu đồ
            chart_data = pd.DataFrame(
                {
                    "Kết quả": ["Có thể uống", "Không thể uống"],
                    "Xác suất": [probability, 1 - probability],
                }
            )

            # Tạo biểu đồ cột
            fig = px.bar(
                chart_data,
                x="Kết quả",
                y="Xác suất",
                title="Phân bố Xác suất Dự đoán",
                color="Kết quả",
                color_discrete_map={
                    "Có thể uống": "#36A2EB",
                    "Không thể uống": "#FF6384",
                },
            )

            # Cập nhật layout
            fig.update_layout(
                yaxis=dict(range=[0, 1], title="Xác suất"),
                xaxis=dict(title="Kết quả"),
                showlegend=False,
            )

            # Hiển thị biểu đồ
            st.plotly_chart(fig, use_container_width=True)

            # Thêm thông tin chi tiết
            st.subheader("Thông tin Chi tiết")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Xác suất Có thể uống", f"{probability:.2%}")
            with col2:
                st.metric("Xác suất Không thể uống", f"{1-probability:.2%}")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")
        st.write("Vui lòng kiểm tra lại dữ liệu đầu vào và các file mô hình.")

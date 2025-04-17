# Embedded Medical Inference System (STM32 + FreeRTOS)

This project implements a basic health risk classification system on an STM32 microcontroller using FreeRTOS. It supports three machine learning models—**Decision Tree**, **Logistic Regression**, and **K-Nearest Neighbors (KNN)**—to classify physiological measurements into predefined risk levels: **Low**, **Medium**, and **High**.

## Features

- Real-time inference using embedded machine learning models
- FreeRTOS-based multi-threading on STM32
- UART output triggered by user button press
- Parallel LED tasks to demonstrate RTOS scheduling
- Predictive models include:
  - Decision Tree
  - Logistic Regression (with softmax)
  - KNN (K = 3)

## Input Format

Each input sample is a 3-dimensional feature vector:

- Systolic Blood Pressure (e.g., 130 mmHg)
- Diastolic Blood Pressure (e.g., 75 mmHg)
- Body Temperature (e.g., 37.0 °C)

These values are hardcoded as `sample_features` and can be replaced with real-time sensor readings for deployment.

## Hardware Requirements

- STM32F4 Series microcontroller board (e.g., STM32F412ZGTx)
- User button (e.g., USER_Btn_Pin) for input trigger
- UART interface (e.g., USART3) for output
- On-board LEDs (LD1, LD2, LD3)

## Toolchain Requirements

- STM32CubeIDE or STM32CubeMX (with FreeRTOS middleware enabled)
- UART set to 115200 baud rate (8N1 configuration)
- ST-Link USB debugger (for flashing and serial communication)

## UART Usage

To monitor UART output on Linux, use:

```bash
sudo minicom -b 115200 -D /dev/ttyACM0
```

## Inference Behavior

On boot:
- LD1 blinks every 500 ms
- LD2 blinks every 1000 ms
- LD3 blinks every 2000 ms

On user button press:
- The system runs inference on a predefined input vector using all three models.
- Results are transmitted via UART in the following format:

```
DT Prediction: Class 1 (medium risk)
LR Prediction: Class 0 (low risk)
KNN Prediction: Class 1 (medium risk)
```

## Implementation Notes

- **Decision Tree**:
  - Hardcoded rules based on medical thresholds

- **Logistic Regression**:
  - Implemented using matrix multiplication and softmax
  - Fixed weights and biases defined in the code

- **KNN**:
  - Distance-based classifier with Euclidean metric
  - Uses a fixed set of five labeled training samples

## Future Work

- Replace hardcoded inputs with sensor interfacing (e.g., temperature, BP sensors)
- Expand training data and refactor ML models into modular components
- Add support for dynamic UART command-based inference

## License

This project is based on STMicroelectronics firmware and is provided under terms described in the [LICENSE](LICENSE) file.

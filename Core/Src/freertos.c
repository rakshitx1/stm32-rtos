/*

This code is part of a project that implements a simple health monitoring system using
machine learning models (Logistic Regression, Decision Tree, and KNN) on an STM32 microcontroller using FreeRTOS.

Google Colab link for models: https://colab.research.google.com/drive/1ZQqohtJSYoM6jttgKan2qIDlMB0ZFQ_j?usp=sharing

*/

/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * File Name          : freertos.c
 * Description        : Code for freertos applications
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "main.h"
#include "cmsis_os.h"
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "semphr.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#define NUM_FEATURES 3
#define NUM_CLASSES 3
#define NUM_TRAINING_SAMPLES 5 // Example number of training samples

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN Variables */

SemaphoreHandle_t buttonSemaphore;
extern UART_HandleTypeDef huart3;

// Pre-trained Logistic Regression weights and biases
float lr_weights[NUM_CLASSES][NUM_FEATURES] = {
	{0.1f, -0.05f, 0.02f},	 // Class 0 weights (e.g., Low Risk)
	{-0.03f, 0.08f, -0.01f}, // Class 1 weights (e.g., Medium Risk)
	{-0.07f, 0.12f, 0.05f}	 // Class 2 weights (e.g., High Risk)
};

float lr_biases[NUM_CLASSES] = {-0.5f, 0.2f, 0.8f};

// Pre-fed training data for KNN (example)
typedef struct
{
	float features[NUM_FEATURES];
	int label;
} KNN_Sample;

KNN_Sample knn_training_data[NUM_TRAINING_SAMPLES] = {
	{{120.0f, 70.0f, 36.5f}, 0},
	{{140.0f, 85.0f, 37.2f}, 1},
	{{160.0f, 95.0f, 38.0f}, 2},
	{{110.0f, 65.0f, 36.0f}, 0},
	{{135.0f, 80.0f, 37.0f}, 1}};

#define KNN_K 3 // Number of neighbors for KNN

/* USER CODE END Variables */
osThreadId LED1Handle;
osThreadId LED2Handle;
osThreadId LED3Handle;
osThreadId InferencingHandle;

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN FunctionPrototypes */

// Function prototypes for Decision Tree
int predict_dt(float features[NUM_FEATURES]);

// Function prototypes for Logistic Regression
int predict_lr(float features[NUM_FEATURES]);
void softmax(float *logits, float *probabilities, int num_classes);

// Function prototypes for KNN
int predict_knn(float features[NUM_FEATURES]);
float euclidean_distance(float *v1, float *v2, int length);

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
	if (GPIO_Pin == USER_Btn_Pin)
	{
		BaseType_t xHigherPriorityTaskWoken = pdFALSE;
		xSemaphoreGiveFromISR(buttonSemaphore, &xHigherPriorityTaskWoken);
		portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
	}
}

/* USER CODE END FunctionPrototypes */

void LED1_init(void const *argument);
void LED2_init(void const *argument);
void LED3_init(void const *argument);
void InferencingTask(void const *argument);

void MX_FREERTOS_Init(void); /* (MISRA C 2004 rule 8.1) */

/* GetIdleTaskMemory prototype (linked to static allocation support) */
void vApplicationGetIdleTaskMemory(StaticTask_t **ppxIdleTaskTCBBuffer, StackType_t **ppxIdleTaskStackBuffer, uint32_t *pulIdleTaskStackSize);

/* USER CODE BEGIN GET_IDLE_TASK_MEMORY */
static StaticTask_t xIdleTaskTCBBuffer;
static StackType_t xIdleStack[configMINIMAL_STACK_SIZE];

void vApplicationGetIdleTaskMemory(StaticTask_t **ppxIdleTaskTCBBuffer, StackType_t **ppxIdleTaskStackBuffer, uint32_t *pulIdleTaskStackSize)
{
	*ppxIdleTaskTCBBuffer = &xIdleTaskTCBBuffer;
	*ppxIdleTaskStackBuffer = &xIdleStack[0];
	*pulIdleTaskStackSize = configMINIMAL_STACK_SIZE;
	/* place for user code */
}
/* USER CODE END GET_IDLE_TASK_MEMORY */

/**
 * @brief  FreeRTOS initialization
 * @param  None
 * @retval None
 */
void MX_FREERTOS_Init(void)
{
	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* USER CODE BEGIN RTOS_MUTEX */
	/* add mutexes, ... */
	/* USER CODE END RTOS_MUTEX */

	/* USER CODE BEGIN RTOS_SEMAPHORES */
	/* add semaphores, ... */
	/* Create the binary semaphore */
	buttonSemaphore = xSemaphoreCreateBinary();
	/* USER CODE END RTOS_SEMAPHORES */

	/* USER CODE BEGIN RTOS_TIMERS */
	/* start timers, add new ones, ... */
	/* USER CODE END RTOS_TIMERS */

	/* USER CODE BEGIN RTOS_QUEUES */
	/* add queues, ... */
	/* USER CODE END RTOS_QUEUES */

	/* Create the thread(s) */
	/* definition and creation of LED1 */
	osThreadDef(LED1, LED1_init, osPriorityNormal, 0, 128);
	LED1Handle = osThreadCreate(osThread(LED1), NULL);

	/* definition and creation of LED2 */
	osThreadDef(LED2, LED2_init, osPriorityNormal, 0, 128);
	LED2Handle = osThreadCreate(osThread(LED2), NULL);

	/* definition and creation of LED3 */
	osThreadDef(LED3, LED3_init, osPriorityNormal, 0, 128);
	LED3Handle = osThreadCreate(osThread(LED3), NULL);

	/* definition and creation of Inferencing */
	osThreadDef(Inferencing, InferencingTask, osPriorityHigh, 0, 256);
	InferencingHandle = osThreadCreate(osThread(Inferencing), NULL);

	/* USER CODE BEGIN RTOS_THREADS */
	/* add threads, ... */
	/* USER CODE END RTOS_THREADS */
}

/* USER CODE BEGIN Header_LED1_init */
/**
 * @brief Function implementing the LED1 thread.
 * @param argument: Not used
 * @retval None
 */
/* USER CODE END Header_LED1_init */
void LED1_init(void const *argument)
{
	/* USER CODE BEGIN LED1_init */
	/* Infinite loop */
	for (;;)
	{
		HAL_GPIO_TogglePin(LD1_GPIO_Port, LD1_Pin); // Toggle LED1
		osDelay(500);
	}
	/* USER CODE END LED1_init */
}

/* USER CODE BEGIN Header_LED2_init */
void LED2_init(void const *argument)
{
	/* USER CODE BEGIN LED2_init */
	/* Infinite loop */
	for (;;)
	{
		HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin); // Toggle LED2
		osDelay(1000);
	}
	/* USER CODE END LED2_init */
}

/* USER CODE BEGIN Header_LED3_init */
void LED3_init(void const *argument)
{
	/* USER CODE BEGIN LED3_init */
	/* Infinite loop */
	for (;;)
	{
		HAL_GPIO_TogglePin(LD3_GPIO_Port, LD3_Pin); // Toggle LED3
		osDelay(2000);
	}
	/* USER CODE END LED3_init */
}

void InferencingTask(void const *argument)
{
	char msg[64];
	const char *risk_levels[] = {"low risk", "medium risk", "high risk"};
	float sample_features[NUM_FEATURES] = {130.0f, 75.0f, 37.0f}; // Example pre-fed data

	for (;;)
	{
		if (xSemaphoreTake(buttonSemaphore, portMAX_DELAY) == pdTRUE)
		{
			// Display Input Features
			snprintf(msg, sizeof(msg), "Sample Data:\r\n");
			HAL_UART_Transmit(&huart3, (uint8_t *)msg, strlen(msg), HAL_MAX_DELAY);
			snprintf(msg, sizeof(msg), "Systolic: 130, Diastolic: 75, Temperature: 37\r\n");
			HAL_UART_Transmit(&huart3, (uint8_t *)msg, strlen(msg), HAL_MAX_DELAY);

			// Choose which model to run: Logistic Regression, Decision Tree or KNN
			int predicted_class_dt = predict_dt(sample_features);
			// Map the prediction to a risk level string and send it via UART
			snprintf(msg, sizeof(msg), "DT Prediction: Class %d (%s)\r\n", predicted_class_dt, risk_levels[predicted_class_dt]);
			HAL_UART_Transmit(&huart3, (uint8_t *)msg, strlen(msg), HAL_MAX_DELAY);
			osDelay(1000);

			int predicted_class_lr = predict_lr(sample_features);
			// Map the prediction to a risk level string and send it via UART
			snprintf(msg, sizeof(msg), "LR Prediction: Class %d (%s)\r\n", predicted_class_lr, risk_levels[predicted_class_lr]);
			HAL_UART_Transmit(&huart3, (uint8_t *)msg, strlen(msg), HAL_MAX_DELAY);
			osDelay(1000);

			int predicted_class_knn = predict_knn(sample_features);
			// Map the prediction to a risk level string and send it via UART
			snprintf(msg, sizeof(msg), "KNN Prediction: Class %d (%s)\r\n\r\n", predicted_class_knn, risk_levels[predicted_class_knn]);
			HAL_UART_Transmit(&huart3, (uint8_t *)msg, strlen(msg), HAL_MAX_DELAY);
			osDelay(1000);

			/*
			command to see UART output in terminal:
			$ sudo minicom -b 115200 -D ${serial_port}
			$ sudo minicom -b 115200 -D /dev/ttyACM0
			*/
		}
	}
}

// Decision Tree Prediction Function
int predict_dt(float sample_features[NUM_FEATURES])
{
	float systolic = sample_features[0];
	float diastolic = sample_features[1];
	float temperature = sample_features[2];

	if (systolic < 125.0f)
	{
		if (temperature < 37.2f)
		{
			return 0; // Low Risk
		}
		else
		{
			return 1; // Medium Risk due to temperature
		}
	}
	else if (systolic < 145.0f)
	{
		if (diastolic < 85.0f)
		{
			return 1; // Medium Risk
		}
		else
		{
			if (temperature >= 38.0f)
			{
				return 2; // High Risk due to temp and pressure
			}
			else
			{
				return 1; // Medium Risk
			}
		}
	}
	else
	{
		if (diastolic >= 90.0f || temperature >= 38.0f)
		{
			return 2; // High Risk
		}
		else
		{
			return 1; // Still Medium Risk
		}
	}
}

// Logistic Regression Prediction Function
int predict_lr(float features[NUM_FEATURES])
{
	float logits[NUM_CLASSES];
	float probabilities[NUM_CLASSES];

	for (int i = 0; i < NUM_CLASSES; i++)
	{
		logits[i] = lr_biases[i];
		for (int j = 0; j < NUM_FEATURES; j++)
		{
			logits[i] += lr_weights[i][j] * features[j];
		}
	}

	softmax(logits, probabilities, NUM_CLASSES);

	int predicted_class = 0;
	float max_prob = probabilities[0];
	for (int i = 1; i < NUM_CLASSES; i++)
	{
		if (probabilities[i] > max_prob)
		{
			max_prob = probabilities[i];
			predicted_class = i;
		}
	}
	return predicted_class;
}

// Softmax Function
void softmax(float *logits, float *probabilities, int num_classes)
{
	float sum_exp = 0.0f;
	float max_logit = logits[0];
	for (int i = 1; i < num_classes; i++)
	{
		if (logits[i] > max_logit)
		{
			max_logit = logits[i];
		}
	}
	for (int i = 0; i < num_classes; i++)
	{
		probabilities[i] = expf(logits[i] - max_logit);
		sum_exp += probabilities[i];
	}
	for (int i = 0; i < num_classes; i++)
	{
		probabilities[i] /= sum_exp;
	}
}

// KNN Prediction Function
int predict_knn(float features[NUM_FEATURES])
{
	float distances[NUM_TRAINING_SAMPLES];
	for (int i = 0; i < NUM_TRAINING_SAMPLES; i++)
	{
		distances[i] = euclidean_distance(features, knn_training_data[i].features, NUM_FEATURES);
	}

	// Find the indices of the k smallest distances
	int nearest_neighbors_labels[KNN_K];
	float sorted_distances[KNN_K];
	int sorted_indices[KNN_K];

	// Initialize with a large value
	for (int i = 0; i < KNN_K; i++)
	{
		sorted_distances[i] = FLT_MAX;
		sorted_indices[i] = -1;
	}

	for (int i = 0; i < NUM_TRAINING_SAMPLES; i++)
	{
		for (int j = 0; j < KNN_K; j++)
		{
			if (distances[i] < sorted_distances[j])
			{
				// Shift larger distances to the right
				for (int k = KNN_K - 1; k > j; k--)
				{
					sorted_distances[k] = sorted_distances[k - 1];
					sorted_indices[k] = sorted_indices[k - 1];
				}
				sorted_distances[j] = distances[i];
				sorted_indices[j] = i;
				break;
			}
		}
	}

	// Count the occurrences of each class among the nearest neighbors
	int class_counts[NUM_CLASSES] = {0};
	for (int i = 0; i < KNN_K; i++)
	{
		if (sorted_indices[i] != -1)
		{
			class_counts[knn_training_data[sorted_indices[i]].label]++;
		}
	}

	// Find the class with the maximum count
	int predicted_class = 0;
	int max_count = class_counts[0];
	for (int i = 1; i < NUM_CLASSES; i++)
	{
		if (class_counts[i] > max_count)
		{
			max_count = class_counts[i];
			predicted_class = i;
		}
	}
	return predicted_class;
}

// Euclidean Distance Function
float euclidean_distance(float *v1, float *v2, int length)
{
	float sum = 0.0f;
	for (int i = 0; i < length; i++)
	{
		sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	}
	return sqrtf(sum);
}

/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

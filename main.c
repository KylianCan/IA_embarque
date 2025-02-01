#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cjson/cJSON.h>
#include <math.h>
#include "Lecture/Bmp2Matrix.h"
#include "Lecture/Bmp2Matrix.c"

// Structure pour contenir les poids et biais
typedef struct {
    float* weights;
    int weights_size;
    float* biases;
    int biases_size;
} LayerParams;

// Charger les poids et biais pour une couche
LayerParams load_layer_params(const char* filename, const char* weight_key, const char* bias_key) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* data = (char*)malloc(length + 1);
    fread(data, 1, length, file);
    fclose(file);

    cJSON* json = cJSON_Parse(data);
    free(data);

    if (!json) {
        printf("Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        exit(EXIT_FAILURE);
    }

    // Lire les poids
    cJSON* weights_json = cJSON_GetObjectItemCaseSensitive(json, weight_key);
    int weights_count = cJSON_GetArraySize(weights_json);
    float* weights = (float*)malloc(weights_count * sizeof(float));
    for (int i = 0; i < weights_count; i++) {
        weights[i] = (float)cJSON_GetArrayItem(weights_json, i)->valuedouble;
    }

    // Lire les biais
    cJSON* biases_json = cJSON_GetObjectItemCaseSensitive(json, bias_key);
    int biases_count = cJSON_GetArraySize(biases_json);
    float* biases = (float*)malloc(biases_count * sizeof(float));
    for (int i = 0; i < biases_count; i++) {
        biases[i] = (float)cJSON_GetArrayItem(biases_json, i)->valuedouble;
    }

    cJSON_Delete(json);

    LayerParams params = {weights, weights_count, biases, biases_count};
    return params;
}

// Fonction d'activation ReLU
float relu(float x) {
    return x > 0 ? x : 0;
}

// Fonction softmax
void softmax(float* input, int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i]);
        sum += input[i];
    }
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

// Calcul d'une couche dense
void dense_layer(char** input, char* output, char* weights, float* biases, int input_size, int output_size) {
    for (int i = 0; i < output_size; i++) {
        output[i] = biases[i];
        for (int j = 0; j < input_size; j++) {
            output[i] += input[j] * weights[i * input_size + j];
        }
        output[i] = relu(output[i]);
    }
}

int main() {
    // Charger les paramètres de la couche 1 du MLP
    LayerParams layer1 = load_layer_params("MLP_weights.json", "layers.1.weight", "layers.1.bias");

    // Charger les paramètres de la couche 2 du MLP
    LayerParams layer2 = load_layer_params("MLP_weights.json", "layers.3.weight", "layers.3.bias");

    // Exemple d'entrée
    /*char* image_path = "./KylianProcessed/0/0_2.bmp";
    float input[784] = Bmp2Matrix(image_path);
    float output1[64];  // Sortie de la première couche
    float output2[10];  // Sortie finale*/

    FILE* path = "./KylianProcessed/0/0_2.bmp";
    BMP* input_bmp;
    LireBitmap(path, input_bmp);
    ConvertRGB2Gray(input_bmp);
    char** output1;
    char** output2;
    // Appliquer la première couche dense
    dense_layer(input_bmp->mPixelsGray, output1, layer1.weights, layer1.biases, 784, 64);

    // Appliquer la seconde couche dense
    dense_layer(output1, output2, layer2.weights, layer2.biases, 64, 10);

    // Appliquer softmax
    softmax(output2, 10);

    // Afficher la sortie
    for (int i = 0; i < 10; i++) {
        printf("Output[%d]: %f\n", i, output2[i]);
    }

    // Libérer la mémoire
    free(layer1.weights);
    free(layer1.biases);
    free(layer2.weights);
    free(layer2.biases);

    return 0;
}


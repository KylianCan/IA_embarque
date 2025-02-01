/*******************************************************
Nom ......... : main.c
Role ........ : Programme principal executant la lecture
                d'une image bitmap
Auteur ...... : Frédéric CHATRIE
Version ..... : V1.1 du 1/2/2021
Licence ..... : /

Compilation :
make veryclean
make
Pour exécuter, tapez : ./all
********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Bmp2Matrix.h"

int main(int argc, char* argv[]){

    FILE* file1 = fopen("W1.txt", "r");
    if (file1 == NULL) {
        perror("Erreur lors de l'ouverture du fichier");
        return 1;
    }
    
    float W1[64][784];

    for (int i = 0; i < 784; i++) { 
        for (int j = 0; j < 64; j++) {
            fscanf(file1, "%f", &W1[j][i]);
        }
    }
    
    fclose(file1);

    FILE* file2 = fopen("W2.txt", "r");
    if (file2 == NULL) {
        perror("Erreur lors de l'ouverture du fichier");
        return 1;
    }
    float W2[10][64];
    int rows = 0;
    int cols = 0;
    for (int i = 0; i < 64; i++) { 
        for (int j = 0; j < 10; j++) {
            fscanf(file1, "%f", &W2[j][i]);
        }
    }

    fclose(file2);
    FILE* file3 = fopen("B1.txt", "r");
    if (file3 == NULL) {
        perror("Erreur lors de l'ouverture du fichier");
        return 1;
    }
    
    float B1[64];
    rows = 0;
    while (fscanf(file3, "%f", &B1[rows]) == 1) {
        rows++;
    }
    fclose(file3);
    
    // Calculer la taille du vecteur en octets
    size_t taille = sizeof(W1)/sizeof(W1[0]);

    // Afficher la taille du vecteur
    // printf("taille: %zu \n", taille);


    FILE* file4 = fopen("B2.txt", "r");
    if (file4 == NULL) {
        perror("Erreur lors de l'ouverture du fichier");
        return 1;
    }
    float B2[10];
    rows = 0;
    while (fscanf(file4, "%f", &B2[rows]) == 1) {
        rows++;
    }
    fclose(file4);
    
  
   BMP bitmap;
   FILE* pFichier=NULL;
   pFichier=fopen(argv[1], "rb");     //Ouverture du fichier contenant l'image
   if (pFichier==NULL) {
       printf("Erreur dans la lecture du fichier\n");
   }

   LireBitmap(pFichier, &bitmap);
   fclose(pFichier);               //Fermeture du fichier contenant l'image
   ConvertRGB2Gray(&bitmap);

   //Model application
   float input_flatten[28*28];
   flatten(bitmap, input_flatten);
   float X1[64];
   linear_function1(X1, W1,input_flatten, B1);
    relu(X1);
   float X2[10];
   linear_function2(X2, W2,X1, B2);
   softmax(X2, 10);
   for (int i = 0; i < 10; i++) {
        printf("| ");
        printf("%0.8f ", X2[i]);
    }
    printf("\n");
   printf("le chiffre predit est : %d \n", argmax(X2,10));
   DesallouerBMP(&bitmap);

   return 0;
}

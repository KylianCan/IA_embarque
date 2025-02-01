
#include <stdio.h>
#ifndef BMP2MATRIX_H
#define BMP2MATRIX_H
#include <math.h>

typedef struct{
char signature[3];            //Signature "BM"
unsigned int tailleFichier;   //Taille du fichier
unsigned int reserved;        //Non utilise
unsigned int offset;          //Nombre de caracteres à parcourir avant d'arriver aux donnees du fichier
}Header;

typedef struct{
unsigned int tailleHeader;    //Taille de l'en-tete du fichier
int largeur;                  //Nombre de colonnes de l'image
int hauteur;                  //Nombre de lignes de l'image
unsigned short trames;        //Nombre de plans dans le cas d'un tenseur
unsigned short bits;          //Nombre de bits par pixel
unsigned int compression;     //Type de compression utilise
unsigned int tailleImage;     //Taille de l'image compresse ou non
int xResolution;              //Resolution des pixels sur les colonnes en pixels par metre
int yResolution;              //Resolution des pixels sur les lignes en pixels par metre
unsigned int nbCouleurs;      //Nombre de couleurs importantes
unsigned int couleursImportantes;      //Nombre de couleurs importantes
}InfoHeader;

typedef struct{
unsigned char rouge;        //Composante rouge du pixel
unsigned char vert;         //Composante verte du pixel
unsigned char bleu;         //Composante bleu du pixel
}Pixels;

typedef struct{
Header header;              //En-tete
InfoHeader infoHeader;      //Info de l'en-tete
Pixels** mPixels;            //Matrice de pixels RGB
unsigned char** mPixelsGray;            //Matrice de pixels en niveau de gris

}BMP;

//Prototypes des fonctions
void AllouerBMP(BMP* pBitmap);

void relu(float* X);

void softmax(float* input, int length);

void flatten(BMP input, float output[]);

void linear_function1(float* X1, float W[64][784],float* X, float* B);

void linear_function2(float* X2, float W[10][64],float* X, float* B);

int argmax(float arr[], int size);

void DesallouerBMP(BMP* pBitmap);

void LireBitmap(FILE* pFichier,BMP* pBitmap);
long LireHeaderBitmap(FILE* pFichier,BMP* pBitmap);
void LireInfoHeaderBitmap(FILE* pFichier,BMP* pBitmap,long curseur);
void LirePixelsBitmap(FILE* pFichier,BMP* pBitmap);

void ConvertRGB2Gray(BMP* pBitmap);


#endif

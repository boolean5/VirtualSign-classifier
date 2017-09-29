/*---------------------------------------------------------------------*/
// A simple console application to get glove sensor data. To save a line
// of data press "s". The file where the data will be saved must be 
// passed in as the first argument.
// Created by Georgia :)
/*---------------------------------------------------------------------*/

#include "fglove.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {

	char    *szPort    = NULL;
        fdGlove *pGloveA    = NULL;
        int      glovetype = FD_GLOVENONE;
	char 	*outputFile = NULL;
	FILE *fp;
	char 	c = 'a';

	// Check that the arguments are correct
	if (argc<3) {
		printf("\nUsage: datasetcreator <devicename> <outputfile>\n");
                printf("Example: sudo ./datasetcreator /dev/usb/hiddev0 dataset.txt\n\n");
                return 0;
	} else {
		szPort = argv[1];
		outputFile = argv[2];
	}

	// Initialize glove
	printf("\nAttempting to open glove A on %s .. ", szPort );
	pGloveA = fdOpen(szPort);
	if (pGloveA == NULL) {
                printf("failed.\n");
                return -1;
        }
        printf("succeeded.\n");	

	// Opening file for writing
	fp = fopen(outputFile, "w");
	if (fp == NULL) {
		printf("failed to open file\n");
		return -1;
	}

	float gloveA_scaled[18];
	// Save the data everytime "s" is pressed and quit when "q" is pressed
	while (c != 'q') {
		c = getchar();
		if (c == 's') {
			fdGetSensorScaledAll(pGloveA, gloveA_scaled);

			fprintf(fp, "A:%0.1f %0.1f||%0.1f||%0.1f %0.1f||%0.1f||%0.1f %0.1f||%0.1f||%0.1f %0.1f||%0.1f||%0.1f %0.1f",
                        gloveA_scaled[FD_THUMBNEAR],
                        gloveA_scaled[FD_THUMBFAR],
                        gloveA_scaled[FD_THUMBINDEX],
                        gloveA_scaled[FD_INDEXNEAR],
                        gloveA_scaled[FD_INDEXFAR],
                        gloveA_scaled[FD_INDEXMIDDLE],
                        gloveA_scaled[FD_MIDDLENEAR],
                        gloveA_scaled[FD_MIDDLEFAR],
                        gloveA_scaled[FD_MIDDLERING],
                        gloveA_scaled[FD_RINGNEAR],
                        gloveA_scaled[FD_RINGFAR],
                        gloveA_scaled[FD_RINGLITTLE],
                        gloveA_scaled[FD_LITTLENEAR],
                        gloveA_scaled[FD_LITTLEFAR]);			

			fprintf(fp, " >> %d\n", fdGetGesture(pGloveA));
			// reset c
			c = 'a';
		}
	}

	// Closing the gloves
	printf("\nClosing glove(s)...\n" );
        fdClose(pGloveA);
        printf("Glove(s) closed.\n\n");

	// Closing the file
	fclose(fp);

	return 0;
}




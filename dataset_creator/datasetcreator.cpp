/*---------------------------------------------------------------------*/
// A simple console application to get glove sensor data. To save a line
// of data press "s". The file where the data will be saved must be 
// passed in as the first argument.
// Created by Georgia :)
/*---------------------------------------------------------------------*/

#define DOTS_10 "**********"
//#define DOTS_10 "▮▮▮▮▮▮▮▮▮▮"
#define SPAC_10 "          "

#include "fglove.h"
#include <stdio.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

int main(int argc, char** argv) {

	char    *szPort    = NULL;
        fdGlove *pGloveA    = NULL;
        int      glovetype = FD_GLOVENONE;
	char 	*outputFile = NULL;
	FILE *fp;
	char 	c = 'a';
	static struct termios oldt, newt;
	int repetitions = 10;
	int i = 0;
	int j = 0;

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
	printf("\nAttempting to open glove A on %s ... ", szPort );
	pGloveA = fdOpen(szPort);
	if (pGloveA == NULL) {
                printf("failed.\n");
                return -1;
        }
        printf("succeeded.\n");	

	// Open output file for writing
	fp = fopen(outputFile, "w");
	if (fp == NULL) {
		printf("failed to open file\n");
		return -1;
	}

	// Change the default behavior of the terminal
	// https://stackoverflow.com/questions/1798511/how-to-avoid-press-enter-with-any-getchar
	/*tcgetattr gets the parameters of the current terminal
	STDIN_FILENO will tell tcgetattr that it should write the settings
	of stdin to oldt*/
	tcgetattr( STDIN_FILENO, &oldt);
	/*now the settings will be copied*/
	newt = oldt;

	/*ICANON normally takes care that one line at a time will be processed
	that means it will return if it sees a "\n" or an EOF or an EOL*/
	newt.c_lflag &= ~(ICANON);          

	/*Those new settings will be set to STDIN
	TCSANOW tells tcsetattr to change attributes immediately. */
	tcsetattr( STDIN_FILENO, TCSANOW, &newt);

	float gloveA_scaled[18];
	printf("\nPlease, input gesture %d:\n ", i);
	printf("\r[%1.*s", j, DOTS_10);
        printf("%1.*s] %d% \r", 10-j, SPAC_10, j*10);
	
	// Save the data everytime "s" is pressed and quit when "q" is pressed
	while (c != 'q') {
		c = getchar();
		if (c == 's') {
			j++;
			printf("\r[%1.*s", j, DOTS_10);
			if (j != 10){
			        printf("%1.*s] %d% \r", 10-j, SPAC_10, j*10); 
			} else {
                                printf("] %d% \r", j*10);
                        }

			fdGetSensorScaledAll(pGloveA, gloveA_scaled);

		        fprintf(fp, "%0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f",
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

		        // fprintf(fp, " >> %d\n", fdGetGesture(pGloveA));
		        fprintf(fp, " %d\n", i);

			if (j==10) {
				j = 0;
                                // update the progress bar
                                i++;
                                if (i == 42) {
                                        break;
                                }
                                printf("\nPlease, input gesture %d:\n ", i);
                                printf("\r[%1.*s", j, DOTS_10);
                                printf("%1.*s] %d% \r", 10-j, SPAC_10, j*10);
			}
			// reset c
			c = 'a';
		}
	}

	// Close the gloves
	printf("\nClosing glove(s)...\n" );
        fdClose(pGloveA);
        printf("Glove(s) closed.\n\n");

	// Close the file
	fclose(fp);

	// Restore the old terminal settings
	tcsetattr( STDIN_FILENO, TCSANOW, &oldt);

	return 0;
}

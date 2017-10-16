/*---------------------------------------------------------------------*/
// A simple console application to get glove sensor data. First input 
// the identifier of the hand configuration you wish to input. To save 
// a line of data press "s". The file where the data will be saved must 
// be passed in as the first argument and the number of repetitions for 
// each hand as the second argument. To exit press "q".
/*---------------------------------------------------------------------*/

#define DOTS_10 "**********"
#define SPAC_10 "          "

#include "fglove.h"
#include <stdio.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char** argv) {

        char    *szPort    = NULL;
        fdGlove *pGloveA    = NULL;
        int      glovetype = FD_GLOVENONE;
	char	*outputFile = NULL;
	char outputFileScaled[100];
	char outputFileRaw[100];
        FILE *fpscaled, *fpraw;
        char    c = 'a';
        static struct termios oldt, newt;
        int repetitions;
        int i = 0;
        int j = 0;
	int handconfid;

        // Check that the arguments are correct
        if (argc<4) {
                printf("\nUsage: datasetcreator <devicename> <outputfile> <numberofrepetitions>\n");
                printf("Example: sudo ./datasetcreator /dev/usb/hiddev0 dataset 10\n\n");
                return 0;
        } else {
                szPort = argv[1];
                outputFile = argv[2];
		repetitions = atoi(argv[3]);
        }

	strcpy(outputFileScaled, outputFile);
	strcpy(outputFileRaw, outputFile);
	strcat(outputFileScaled, "-scaled.txt");
        strcat(outputFileRaw, "-raw.txt");

        // Initialize glove
        printf("\nAttempting to open glove A on %s ... ", szPort );
        pGloveA = fdOpen(szPort);
        if (pGloveA == NULL) {
                printf("failed.\n");
                return -1;
        }
        printf("succeeded.\n");

        // Open output files for writing
        fpscaled = fopen(outputFileScaled, "w");
        if (fpscaled == NULL) {
                printf("failed to open file\n");
                return -1;
        }
	fpraw = fopen(outputFileRaw, "w");
        if (fpraw == NULL) {
                printf("failed to open file\n");
                return -1;
        }

        // Calibration instructions
        printf("\nPlease calibrate the glove by following the instructions below.\n");
	printf("\nPress ENTER when you are ready to begin the creation of the dataset.\n");
	printf("\n1. Hold your hand in a relaxed, open position.\n");
	printf("\n2. Open all of your fingers as wide as possible.\n");
	printf("\n3. Close all of your fingers except your thumb.\n");
	printf("\n3. Close your thumb too.\n");
	getchar();

	// Print instructions
        printf("\nPress 's' to record sensor values, 'q' to quit.\n");


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
	unsigned short gloveA_raw[18];
	printf("\nPlease, input the ID of the hand configuration you wish to capture:\n");
	scanf("%d", &handconfid);
        printf("\nPlease, input gesture %d:\n ", handconfid);
        printf("\r[%1.*s", j, DOTS_10);
        printf("%1.*s] %d% \r", repetitions-j, SPAC_10, j*100/repetitions);

        // Save the data everytime "s" is pressed and quit when "q" is pressed
        while (c != 'q') {
                c = getchar();
                if (c == 's') {
                        j++;
                        printf("\r[%1.*s", j, DOTS_10);
                        if (j != repetitions){
                                printf("%1.*s] %d% \r", repetitions-j, SPAC_10, j*100/repetitions);
                        } else {
                                printf("] %d% \r", j*100/repetitions);
                        }

			// Get scaled Data
                        fdGetSensorScaledAll(pGloveA, gloveA_scaled);

                        fprintf(fpscaled, "%0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f",
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
			fprintf(fpscaled, " %d\n", handconfid);

			// Get raw data
			fdGetSensorRawAll(pGloveA, gloveA_raw);

                        fprintf(fpraw, "%hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu",
                        gloveA_raw[FD_THUMBNEAR],
                        gloveA_raw[FD_THUMBFAR],
                        gloveA_raw[FD_THUMBINDEX],
                        gloveA_raw[FD_INDEXNEAR],
                        gloveA_raw[FD_INDEXFAR],
                        gloveA_raw[FD_INDEXMIDDLE],
                        gloveA_raw[FD_MIDDLENEAR],
                        gloveA_raw[FD_MIDDLEFAR],
                        gloveA_raw[FD_MIDDLERING],
                        gloveA_raw[FD_RINGNEAR],
                        gloveA_raw[FD_RINGFAR],
                        gloveA_raw[FD_RINGLITTLE],
                        gloveA_raw[FD_LITTLENEAR],
                        gloveA_raw[FD_LITTLEFAR]);

                        // fprintf(fp, " >> %d\n", fdGetGesture(pGloveA));
                        fprintf(fpraw, " %d\n", handconfid);

                        if (j==repetitions) {
                                j = 0;
				printf("\nPlease, input the ID of the hand configuration you wish to capture:\n");
		        	scanf("%d", &handconfid);
			        printf("\nPlease, input gesture %d:\n ", handconfid);
                                printf("\r[%1.*s", j, DOTS_10);
                                printf("%1.*s] %d% \r", repetitions-j, SPAC_10, j*100/repetitions);
                        }
                        // reset c
                        c = 'a';
                }
        }

        // Close the gloves
        printf("\nClosing glove(s)...\n" );
        fdClose(pGloveA);
        printf("Glove(s) closed.\n\n");

        // Close the files
        fclose(fpscaled);
	fclose(fpraw);

        // Restore the old terminal settings
        tcsetattr( STDIN_FILENO, TCSANOW, &oldt);

        return 0;
}

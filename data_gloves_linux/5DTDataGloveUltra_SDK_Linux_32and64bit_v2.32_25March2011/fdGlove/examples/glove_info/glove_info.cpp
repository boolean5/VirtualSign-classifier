/*--------------------------------------------------------------------------*/
// A simple console application to get glove info
//
// Copyright (C) 2010, 5DT <Fifth Dimension Technologies>
//
/*--------------------------------------------------------------------------*/

#include "fglove.h"
#include <stdio.h>
#include <unistd.h>  // for usleep

int main( int argc, char** argv )
{
	char    *szPort    = NULL;
	fdGlove *pGloveA    = NULL;
	fdGlove *pGloveB    = NULL;
	int      glovetype = FD_GLOVENONE;

	if (argc<2)
	{
		printf("\nUsage: glove_info <devicename>\n");
		printf("Example: sudo ./glove_info /dev/usb/hiddev0\n\n");
		return 0;
	}
	else
	{
		szPort = argv[1];
	}

	// Initialize glove
	printf("\nAttempting to open glove A on %s .. ", szPort );
	pGloveA = fdOpen(szPort);
	if (pGloveA == NULL)
	{
		printf("failed.\n");
		return -1;
	}
	printf("succeeded.\n");

	// if GloveA is of type wireless, then there could also be a GloveB connected to the same wireless device
	glovetype = fdGetGloveType(pGloveA);
	switch (glovetype)
	{
		case FD_GLOVE5UW:
		case FD_GLOVE14UW: {
			printf( "Attempting to open glove B on %s .. ", szPort );
			// We open Glove B on exactly the same port...
			if (NULL == (pGloveB = fdOpen(szPort)))
			{
				printf( "failed.\n" );
				printf("Using only Glove A...\n");
			} 
			else
			{
				printf("succeeded.\n");
				printf(" Using Glove A and Glove B...\n");
			}
		} break;
		
		default:
			printf("Glove A isn't wireless. Not opening Glove B.\n");
			
	}

	//---------------------------------------------------------------------------------------
	// Display Glove A data
	printf("\nGlove A:\n");
	
	char *szType = "?";
	glovetype = fdGetGloveType(pGloveA);
	switch (glovetype) 
	{
		case FD_GLOVENONE:    szType = "None"; break;
		case FD_GLOVE5U:      szType = "Data Glove 5 Ultra"; break;
		case FD_GLOVE5UW:     szType = "Data Glove 5 Ultra W"; break;
		case FD_GLOVE5U_USB:  szType = "Data Glove 5 Ultra USB"; break;
		case FD_GLOVE7:       szType = "Data Glove 5"; break;
		case FD_GLOVE7W:      szType = "Data Glove 5W"; break;
		case FD_GLOVE16:      szType = "Data Glove 16"; break;
		case FD_GLOVE16W:     szType = "Data Glove 16W"; break;
		case FD_GLOVE14U:     szType = "DG14 Ultra serial"; break;
		case FD_GLOVE14UW:    szType = "DG14 Ultra W"; break;
		case FD_GLOVE14U_USB: szType = "DG14 Ultra USB"; break;
	}
	
	printf("Glove type: %s\n", szType );
	printf("Glove handedness: %s\n", fdGetGloveHand(pGloveA)==FD_HAND_RIGHT?"Right":"Left" );
	printf("Data rate: %iHz\n", fdGetPacketRate(pGloveA));
	
	//---------------------------------------------------------------------------------------
	// Display Glove B data, if the glove is available

	if (pGloveB != NULL)
	{
		printf("\nGlove B:\n");
		glovetype = fdGetGloveType(pGloveB);
		switch (glovetype)
		{
			case FD_GLOVENONE:    szType = "None"; break;
			case FD_GLOVE5U:      szType = "Data Glove 5 Ultra"; break;
			case FD_GLOVE5UW:     szType = "Data Glove 5 Ultra W"; break;
			case FD_GLOVE5U_USB:  szType = "Data Glove 5 Ultra USB"; break;
			case FD_GLOVE7:       szType = "Data Glove 5"; break;
			case FD_GLOVE7W:      szType = "Data Glove 5W"; break;
			case FD_GLOVE16:      szType = "Data Glove 16"; break;
			case FD_GLOVE16W:     szType = "Data Glove 16W"; break;
			case FD_GLOVE14U:     szType = "DG14 Ultra serial"; break;
			case FD_GLOVE14UW:    szType = "DG14 Ultra W"; break;
			case FD_GLOVE14U_USB: szType = "DG14 Ultra USB"; break;
		}
	
		printf("Glove type: %s\n", szType );
		printf("Glove handedness: %s\n", fdGetGloveHand(pGloveB)==FD_HAND_RIGHT? "Right":"Left" );	
	}


	// Close gloves
	printf("\nClosing glove(s)...\n" );
	fdClose(pGloveA);
	fdClose(pGloveB);
	printf("Glove(s) closed.\n\n");

	return 0;
}
/*--------------------------------------------------------------------------*/

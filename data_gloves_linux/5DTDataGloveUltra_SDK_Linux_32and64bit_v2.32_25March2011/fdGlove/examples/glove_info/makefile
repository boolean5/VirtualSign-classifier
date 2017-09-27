#
# Makefile for glove_info, a simple console app to demonstrate reading glove info
#
# Copyright (C) 2010, 5DT <Fifth Dimension Technologies>
#

CPP = g++
CFLAGS = -w
BIN = glove_info
LIBS = -lfglove
OBJFILES = glove_info.o

default: $(OBJFILES)
	$(CPP) -o $(BIN) $(OBJFILES) $(LIBS)

clean:
	rm -f $(BIN) $(OBJFILES) *~ core #*

%.o: %.cpp
	$(CPP) $(CFLAGS) $(CFG_INC) -c $< -o $@


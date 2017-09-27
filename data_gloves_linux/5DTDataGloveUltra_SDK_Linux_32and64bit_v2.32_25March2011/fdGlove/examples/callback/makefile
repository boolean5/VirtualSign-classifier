#
# Makefile for testglove, a simple console app to demonstrate reading the glove using a callback method
#
# Copyright (C) 2010, 5DT <Fifth Dimension Technologies>
#

CPP = g++
CFLAGS = -w
BIN = callback
LIBS = -lfglove
OBJFILES = callback.o

default: $(OBJFILES)
	$(CPP) -o $(BIN) $(OBJFILES) $(LIBS)

clean:
	rm -f $(BIN) $(OBJFILES) *~ core #*

%.o: %.cpp
	$(CPP) $(CFLAGS) $(CFG_INC) -c $< -o $@


# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2
LIBS = -lm

# Target executable name
TARGET = cpu_model

# Source files
SRCS = cpu_model.c utilities.c
# Header files 
HEADERS = cpu_model.h

# Default rule: build the target
all: $(TARGET)

# Rule to link the object files into the final executable
$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LIBS)

# Rule to clean up the build files
clean:
	rm -f $(TARGET)

# Rule to run the simulation
run: all
	./$(TARGET)



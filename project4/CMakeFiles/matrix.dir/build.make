# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lin_junjie/C++_project/project/project4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lin_junjie/C++_project/project/project4

# Include any dependencies generated for this target.
include CMakeFiles/matrix.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/matrix.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/matrix.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matrix.dir/flags.make

CMakeFiles/matrix.dir/matrix.c.o: CMakeFiles/matrix.dir/flags.make
CMakeFiles/matrix.dir/matrix.c.o: matrix.c
CMakeFiles/matrix.dir/matrix.c.o: CMakeFiles/matrix.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lin_junjie/C++_project/project/project4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/matrix.dir/matrix.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/matrix.dir/matrix.c.o -MF CMakeFiles/matrix.dir/matrix.c.o.d -o CMakeFiles/matrix.dir/matrix.c.o -c /home/lin_junjie/C++_project/project/project4/matrix.c

CMakeFiles/matrix.dir/matrix.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/matrix.dir/matrix.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lin_junjie/C++_project/project/project4/matrix.c > CMakeFiles/matrix.dir/matrix.c.i

CMakeFiles/matrix.dir/matrix.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/matrix.dir/matrix.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lin_junjie/C++_project/project/project4/matrix.c -o CMakeFiles/matrix.dir/matrix.c.s

CMakeFiles/matrix.dir/matrixmultiplication.c.o: CMakeFiles/matrix.dir/flags.make
CMakeFiles/matrix.dir/matrixmultiplication.c.o: matrixmultiplication.c
CMakeFiles/matrix.dir/matrixmultiplication.c.o: CMakeFiles/matrix.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lin_junjie/C++_project/project/project4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/matrix.dir/matrixmultiplication.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/matrix.dir/matrixmultiplication.c.o -MF CMakeFiles/matrix.dir/matrixmultiplication.c.o.d -o CMakeFiles/matrix.dir/matrixmultiplication.c.o -c /home/lin_junjie/C++_project/project/project4/matrixmultiplication.c

CMakeFiles/matrix.dir/matrixmultiplication.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/matrix.dir/matrixmultiplication.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lin_junjie/C++_project/project/project4/matrixmultiplication.c > CMakeFiles/matrix.dir/matrixmultiplication.c.i

CMakeFiles/matrix.dir/matrixmultiplication.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/matrix.dir/matrixmultiplication.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lin_junjie/C++_project/project/project4/matrixmultiplication.c -o CMakeFiles/matrix.dir/matrixmultiplication.c.s

# Object files for target matrix
matrix_OBJECTS = \
"CMakeFiles/matrix.dir/matrix.c.o" \
"CMakeFiles/matrix.dir/matrixmultiplication.c.o"

# External object files for target matrix
matrix_EXTERNAL_OBJECTS =

matrix: CMakeFiles/matrix.dir/matrix.c.o
matrix: CMakeFiles/matrix.dir/matrixmultiplication.c.o
matrix: CMakeFiles/matrix.dir/build.make
matrix: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
matrix: /usr/lib/x86_64-linux-gnu/libpthread.a
matrix: CMakeFiles/matrix.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lin_junjie/C++_project/project/project4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable matrix"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matrix.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matrix.dir/build: matrix
.PHONY : CMakeFiles/matrix.dir/build

CMakeFiles/matrix.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/matrix.dir/cmake_clean.cmake
.PHONY : CMakeFiles/matrix.dir/clean

CMakeFiles/matrix.dir/depend:
	cd /home/lin_junjie/C++_project/project/project4 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lin_junjie/C++_project/project/project4 /home/lin_junjie/C++_project/project/project4 /home/lin_junjie/C++_project/project/project4 /home/lin_junjie/C++_project/project/project4 /home/lin_junjie/C++_project/project/project4/CMakeFiles/matrix.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/matrix.dir/depend

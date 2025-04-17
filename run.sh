#!/bin/bash

# Setting up trap to handle exit status
trap 'EXIT_STATUS=$?; if [ $EXIT_STATUS -eq 0 ]; then echo -e "\033[32mProgram exited successfully.\033[0m"; else echo -e "\033[31mProgram exited with status $EXIT_STATUS.\033[0m"; fi' EXIT

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Navigate to the project directory
cd "$SCRIPT_DIR" || { echo -e "\033[31mFailed to navigate to the script directory.\033[0m"; exit 1; }

# Initialize variables
NINJA=true
BUILD_TYPE="Debug"
CLEAN=false
VERBOSE=false
PARALLEL=true
FLASH=false

# Parse arguments
for arg in "$@"; do
	case "${arg,,}" in
		("n")
			NINJA=false
			;;
		("r")
			BUILD_TYPE="Release"
			;;
		("mr")
			BUILD_TYPE="MinSizeRel"
			;;
		("rd")
			BUILD_TYPE="RelWithDebInfo"
			;;
		("p")
			PARALLEL=false
			;;
		("v")
			VERBOSE=true
			;;
		("c")
			CLEAN=true
			;;
		("f")
			FLASH=true
			;;
		(*)
			echo -e "\033[31mInvalid argument: $arg\033[0m"
			exit 1
			;;
	esac
done

# Print properties
echo " "
echo -e "\033[34m#############################################\033[0m"
# echo -e "\033[34mProject Name: $PROJECT_NAME\033[0m"
echo -e "\033[34mRunning with the following properties:\033[0m"

# Check if Ninja is available
if [ "$NINJA" = true ]; then
	if ! ninja --version &> /dev/null; then
		echo -e "\033[31mNinja not found.\033[0m"
		echo -e "\033[31mFalling back to default build system.\033[0m"
		NINJA=false
	else
		echo -e "\033[34mBuild System: Ninja\033[0m"
	fi
fi

echo -e "\033[34mBuild Type: $BUILD_TYPE\033[0m"
echo -e "\033[34mParallel Build: $PARALLEL\033[0m"
echo -e "\033[34mVerbose: $VERBOSE\033[0m"
echo -e "\033[34mClean Build: $CLEAN\033[0m"

echo -e "\033[34m#############################################\033[0m"
echo " "

# Remove existing build directory if it exists and the clean flag is set
if [ "$CLEAN" = true ]; then
	if [ -d "build" ]; then
		rm -rf build || { echo -e "\033[31mFailed to remove build directory.\033[0m"; exit 1; }
	fi
fi

# Create build directory and configure project
if [ "$NINJA" = true ]; then
	cmake -S . -B build/ -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -G Ninja || exit $?
else
	cmake -S . -B build/ -DCMAKE_BUILD_TYPE="$BUILD_TYPE" || exit $?
fi

# Build the project
BUILD_FLAGS=""
if [ "$VERBOSE" = true ]; then
	if [ "$NINJA" = true ]; then
		BUILD_FLAGS="-v"
	else
		BUILD_FLAGS="-- VERBOSE=1"
	fi
fi

if [ "$PARALLEL" = true ]; then
	cmake --build build/ --config "$BUILD_TYPE" $BUILD_FLAGS --parallel || exit $?
else
	cmake --build build/ --config "$BUILD_TYPE" $BUILD_FLAGS || exit $?
fi

# Source the project name from the file
if [ -f "build/project_name.txt" ]; then
	PROJECT_NAME=$(cat build/project_name.txt)
else
	echo -e "\033[31mProject name file not found.\033[0m"
	exit 1
fi

# Convert ELF to BIN
arm-none-eabi-objcopy -O binary build/$PROJECT_NAME.elf build/$PROJECT_NAME.bin

# Flash the BIN file
if [ "$FLASH" = true ]; then
	st-flash --format binary write build/$PROJECT_NAME.bin 0x08000000
fi

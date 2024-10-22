#!/bin/bash

# Define the directories to be cleared
DIR1="./data"
DIR2="./frames"
DIR3="./backup_resume"
DIR4="./main_resume"


# Check if DIR1 exists and is a directory
if [ -d "$DIR1" ]; then
    echo "Clearing directory: $DIR1"
    rm -rf "$DIR1"/*
else
    echo "Directory $DIR1 does not exist."
fi

# Check if DIR2 exists and is a directory
if [ -d "$DIR2" ]; then
    echo "Clearing directory: $DIR2"
    rm -rf "$DIR2"/*
else
    echo "Directory $DIR2 does not exist."
fi

# Check if DIR2 exists and is a directory
if [ -d "$DIR3" ]; then
    echo "Clearing directory: $DIR3"
    rm -rf "$DIR3"/*
else
    echo "Directory $DIR3 does not exist."
fi

# Check if DIR2 exists and is a directory
if [ -d "$DIR4" ]; then
    echo "Clearing directory: $DIR4"
    rm -rf "$DIR4"/*
else
    echo "Directory $DIR4 does not exist."
fi

rm data.zip

echo "Directories cleared."

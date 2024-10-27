#!/bin/bash

# Function to run file in a new terminal with custom title
run_file_in_terminal() {
  subfolder=$1
  file=$2
  title=$3
  
  # Open a new terminal window in the specified subfolder and run the file with custom title
  gnome-terminal --working-directory="$PWD/$subfolder" --title="$title" -- sh -c "./$file; exit"
}

# Loop to run the script in 20 different folders
for i in {1..20}
do
  folder="run_$i"
  title="Terminal $i"
  
  # Run the file in a new terminal window for each folder
  run_file_in_terminal "$folder" "./do_all.sh" "$title" &
done

